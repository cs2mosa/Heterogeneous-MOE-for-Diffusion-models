from typing import Any
from models.model_config1 import preconditioned_HDMOEM
import torch
from utils import EDM_LOSS, sample_sigma,ZetaScheduler,MaskGenerator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from VAE_CLIP import StabilityVAE ,CLIP_EMBED
import json
from configs import model_configs,mask_configs,zeta_configs,loss_configs,optim_configs
from graphs.logger import Logger

def training_HDMOE(model_config: dict[str,Any],
                   Optim_config: dict[str,Any],
                   loss_config: dict[str,Any],
                   mask_config: dict[str,Any],
                   zeta_config: dict[str,Any]
                   ):

    dataloader = get_dataloader(model_config)
    data_iter = iter(dataloader)
    logger = Logger(
        log_dir="./logs",
        run_name="hdmoem_cifar10_v1",
        log_interval=10  # Log every 10 steps
    )
    vae = StabilityVAE(batch_size=model_config["batch_size"])
    vae.init(device=model_config["device"])
    clip = CLIP_EMBED(device=model_config["device"])
    clip.init()
    model = preconditioned_HDMOEM(
        IN_in_channels=model_config["img_channels"],
        IN_img_resolution=model_config['img_resolution'],
        internal_channels=model_config['internal_channels'],
        time_emb_dim=model_config["time_emb_dim"],
        text_emb_dim=model_config["text_emb_dim"],
        num_experts=model_config["num_experts"],
        top_k=model_config["top_k"],
        Fourier_bandwidth=model_config["fourier_bandwidth"],
        VIT_num_blocks=model_config['VIT_num_blocks'],  # Shallow ViT for demo
        VIT_patch_sizes=model_config['VIT_patch_sizes'],  # 4x4 patches for 32x32 img
        VIT_num_groups=model_config['VIT_num_groups'],
        VIT_num_heads=model_config['VIT_num_heads'],
        VIT_emb_size=model_config['VIT_emb_size'],
        Unet_num_blocks=model_config['Unet_num_blocks'],
        Unet_channel_mult=model_config['Unet_channel_mult'],
        Unet_channel_mult_emb=model_config['Unet_channel_mult_emb'],# 2 levels for 32x32 -> 16->8
        Unet_kernel_sizes= model_config['Unet_kernel_sizes'],
        Unet_model_channels=model_config['Unet_model_channels'],
        sigma_data=model_config["sigma_data"],
        log_var_channels= model_config["log_var_channels"],
    ).to(model_config["device"])

    optimizer = torch.optim.AdamW(list(model.parameters()),
                                  lr=Optim_config["lr"]
                                  )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                           T_max=model_config['total_steps'],
                                                           eta_min=Optim_config['eta_min']
                                                           )

    zeta_sched = ZetaScheduler(total_steps=model_config["total_steps"],
                               max_zeta=zeta_config["max_zeta"],
                               min_zeta=zeta_config["min_zeta"],
                               strategy=zeta_config["strategy"],
                               warmup_ratio=zeta_config["warmup_ratio"],
                               )

    Unet_mask_gen = MaskGenerator(expert_attributes=mask_config["unet_attr"],
                                  p_mean= mask_config["p_mean"],
                                  p_std= mask_config["p_std"],
                                  total_steps=model_config['total_steps'],
                                  min_active=mask_config['min_active'],
                                  step_size=mask_config['step_size'],
                                  max_bandwidth=mask_config['max_BW'],
                                  bandwidth=mask_config['BW'],
                                  strat_band=mask_config['strat_band'],
                                  noise_range=mask_config['unet_noise_range']
                                  )

    vit_mask_gen = MaskGenerator(expert_attributes= mask_config["vit_attr"],
                                 p_mean= mask_config["p_mean"],
                                 p_std= mask_config["p_std"],
                                 total_steps=model_config['total_steps'],
                                 min_active=mask_config['min_active'],
                                 step_size=mask_config['step_size'],
                                 max_bandwidth=mask_config['max_BW'],
                                 bandwidth=mask_config['BW'],
                                 strat_band=mask_config['strat_band'],
                                 noise_range=mask_config['vit_noise_range']
                                 )

    criterion = EDM_LOSS(num_experts=model_config["num_experts"],
                         sigma_data=model_config["sigma_data"],
                         Unet_bal= loss_config["unet_bal"],
                         vit_bal=loss_config['vit_bal'],
                         z_bal=loss_config['z_bal'],
                         prior_bal=loss_config['prior_bal'],
                         transition_sigma=loss_config['transition_sigma'],
                         sharpness=loss_config['sharpness']
                         )

    json_log_path = os.path.join(model_config["save_dir_stats"], "training_stats.jsonl")
    if os.path.exists(json_log_path):
        os.remove(json_log_path)

    print(f"Logging metrics to: {json_log_path}")

    model.train()
    current_mse = float('inf')
    for step in range(model_config["total_steps"]):
        try:
            images_rgb, _ = next(data_iter)  # Ignore labels, use fixed prompt
        except StopIteration:
            data_iter = iter(dataloader)
            images_rgb, _ = next(data_iter)

        images = images_rgb.to(model_config["device"])
        images = (images + 1) / 2
        images = images * 255
        latent_images = vae.encode(images)
        sigma = sample_sigma(batch_size=model_config['batch_size'],
                             uniform=False,
                             p_mean=mask_config["p_mean"],
                             p_std=mask_config["p_std"],
                             sigma_max=model_config['sigma_max'],
                             sigma_min=model_config['sigma_min']
                             ).to(model_config["device"])

        noise = torch.randn_like(latent_images).to(model_config["device"]) * sigma
        images_noised = latent_images + noise
        cur_zeta = zeta_sched.get_zeta(step = step)
        Unet_mask = Unet_mask_gen(sigma = sigma,
                                  step = step)

        vit_mask = vit_mask_gen(sigma=sigma,
                                  step=step)

        text_emb = clip.encode_text([model_config["fixed_prompt"]] * model_config["batch_size"])
        out_model = model(
            x=images_noised,
            sigma=sigma,
            text_emb=text_emb,
            Unet_router_mask=Unet_mask,
            Vit_router_mask=vit_mask,
            zeta=cur_zeta,
            return_log_var=False
        )

        loss = criterion(sigma_vec=sigma,
                         x= latent_images,
                         sigma = sigma,
                         out_model=out_model
                         )

        logger.log_training_step(
            step=step,
            loss_dict=loss,
            zeta=cur_zeta,
            log_var=out_model['log_var'].mean(),
            lr=optimizer.param_groups[0]['lr'],
            sigma=sigma
        )

        # Log router statistics
        logger.log_router_statistics(
            step=step,
            unet_probs=out_model["Unet_router_loss"],
            vit_probs=out_model["vit_router_loss"],
            sigma=sigma
        )

        # Log scaling and gating
        gate_weights = model.out_gate  # You'll need to expose this
        logger.log_scaling_gating(
            step=step,
            scaling_factors=out_model["scaling_net_out"],
            gate_weights=gate_weights,
            sigma=sigma
        )

        optimizer.zero_grad()
        loss["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #loggers
        logger.log_gradients(step=step, model=model)
        logger.log_weight_statistics(step=step, model=model)

        optimizer.step()
        scheduler.step()
        current_mse = loss["denoising"].item()

        if step % 100 == 0:
            print(f"Step {step}/{model_config['total_steps']} | "
                  f"total_Loss: {loss["loss"]:.4f} | "
                  f"MSE: {loss["denoising"]:.4f} | "
                  f"Z_loss: {loss["z_loss"]:.4f} | "
                  f"Routers_loss: {loss["balance"]:.4f} | "
                  f"entropy: {loss["entropy"]:.4f} | "
                  f"pure_loss: {loss['pure_loss']:.4f} | "
                  f"LogVar: {out_model["log_var"].mean().item() if out_model["log_var"] is not None else 0.0 :.3f}")

        if step % model_config.get("save_interval", 1000) == 0 and step > 0:
            save_checkpoint(model, optimizer, step, current_mse, {"model_configs":model_config, "Optim_config":Optim_config,
                   "loss_config": loss_config,
                   "mask_config": mask_config,
                   "zeta_config": zeta_config},
                            f"ckpt_{step}.pt")

    save_checkpoint(model, optimizer, model_config["total_steps"], current_mse,
                    {"model_configs":model_config, "Optim_config":Optim_config,
                   "loss_config": loss_config,
                   "mask_config": mask_config,
                   "zeta_config": zeta_config}, "final_model1.pt")

    print("Training Complete.")

# will be transferred to data_collector.py, but we try here for now
def get_dataloader(cfg):
    transform = transforms.Compose([
        transforms.Resize((cfg["data_img_res"], cfg["data_img_res"])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Map to [-1, 1]
    ])

    # Downloads automatically (~300MB)
    dataset = datasets.Flowers102(root='./data', split='train', download=True, transform=transform)
    # Augment with validation set to get more data
    val_set = datasets.Flowers102(root='./data', split='val', download=True, transform=transform)
    full_dataset = torch.utils.data.ConcatDataset([dataset, val_set])

    return DataLoader(full_dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)

#will be saved in utils.py
def save_checkpoint(model,
                    optimizer,
                    step,
                    mse_score,
                    configs,
                    filename
                    ):
    if "save_dir" in configs:
        save_path = configs["save_dir"]
    elif "model_configs" in configs and "save_dir" in configs["model_configs"]:
        save_path = configs["model_configs"]["save_dir"]
    else:
        save_path = "./checkpoints"

    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'mse': mse_score,
        'config': configs
    }
    torch.save(checkpoint, str(full_path))
    print(f"   [Save] Checkpoint saved: {full_path}")

#will be saved in utils.py
def log_step_json(filepath,
                  step,
                  loss_dict,
                  zeta,
                  log_var,
                  lr
                  ):
    """
    Appends a single step's data as a JSON object to the file.
    """
    record = {
        "step": step,
        "zeta": round(float(zeta), 6),
        "log_var": round(float(log_var), 6),
        # Main Losses
        "total_loss": round(loss_dict["loss"].item(), 6),
        "raw_mse": round(loss_dict["denoising"].item(), 6), # Or 'raw_mse' if you updated the class
        # Aux Losses (Use .get() in case you disable them later)
        "balance_loss": round(loss_dict["balance"].item(), 6),
        "z_loss": round(loss_dict["z_loss"].item(), 6),
        "entropy_loss": round(loss_dict["entropy"].item(), 6),
        "edm_loss": round(loss_dict["pure_loss"].item(),6),
        "lr":round(lr,6)
    }
    with open(filepath, "a") as f:
        f.write(json.dumps(record) + "\n")

if __name__ == '__main__':
    training_HDMOE(model_configs,
                   optim_configs,
                   loss_configs,
                   mask_configs,
                   zeta_configs)
