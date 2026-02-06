from typing import Any
from models.model_config2 import preconditioned_HDMOEM
import torch
from utils import EDM_LOSS, sample_sigma_hybrid,ZetaScheduler,MaskGenerator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from VAE_CLIP import StabilityVAE ,CLIP_EMBED
from configs import model_configs,mask_configs,zeta_configs,loss_configs,optim_configs
from graphs.logger import Logger
from EDM_sampler import EDM_Sampler
from torchvision.utils import save_image

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
        run_name="hdmoem_flower102_v1",
        log_interval=10  # Log every 10 steps
    )
    vae = StabilityVAE(batch_size=model_config["batch_size"],target_std=model_config['sigma_data'])
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

    optimizer = torch.optim.AdamW([
                                {'params': model.net.Unet_experts.parameters(), 'lr': optim_configs['lr_unet']},
                                {'params': model.net.VIT_experts.parameters(), 'lr': optim_configs['lr_vit']}, # 2x Boost
                                {'params': model.net.cross_attn.parameters(), 'lr': optim_configs['lr_attn']},  # Boost the fusion bridge
                                {'params': model.net.routers.parameters(), 'lr': optim_configs['lr_router']}],
                                )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                           T_max=Optim_config['total_schedule_steps'],
                                                           eta_min=Optim_config['eta_min']
                                                           )

    zeta_sched = ZetaScheduler(total_steps=zeta_config["total_schedule_steps"],
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
        sigma = sample_sigma_hybrid(batch_size=model_config['batch_size'],
                             sigma_max=model_config['sigma_max'],
                             sigma_min=model_config['sigma_min'],
                             p_mean= mask_config['p_mean'],
                             p_std= mask_config['p_std'],
                             extreme_prob= 0.5,
                             device=model_config["device"]
                             )

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
            transition_point = mask_config["p_mean"],
            softness = mask_config["p_std"],
            return_log_var=True
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
            log_var=out_model['log_var'].mean().item() if out_model["log_var"] is not None else 0.0,
            lr=optimizer.param_groups[0]['lr'],
            sigma=sigma,
            p_mean=mask_config["p_mean"],
            p_std=mask_config["p_std"]
        )

        # Log router statistics
        logger.log_router_statistics(
            step=step,
            unet_probs=out_model["Unet_router_loss"],
            vit_probs=out_model["vit_router_loss"],
            sigma=sigma,
            p_mean=mask_config["p_mean"],
            p_std=mask_config["p_std"]
        )

        # Log scaling and gating
        logger.log_scaling_gating(
            scaling_factors=out_model["scaling_net_out"],
            gate_weights=out_model["out_gate"],
            sigma=sigma,
        )

        optimizer.zero_grad()
        loss["loss"].backward()
        # loggers
        logger.log_gradients(step=step, model=model.net)
        logger.log_weight_statistics(step=step, model=model.net)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

def sample_and_save(CONFIG):
    device = CONFIG["device"]

    # Load Model
    print("Loading model structure...")
    modeli = preconditioned_HDMOEM(
        IN_in_channels=CONFIG["img_channels"],
        IN_img_resolution=CONFIG['img_resolution'],
        internal_channels=CONFIG['internal_channels'],
        time_emb_dim=CONFIG["time_emb_dim"],
        text_emb_dim=CONFIG["text_emb_dim"],
        num_experts=CONFIG["num_experts"],
        top_k=CONFIG["top_k"],
        Fourier_bandwidth=CONFIG["fourier_bandwidth"],
        VIT_num_blocks=CONFIG['VIT_num_blocks'],
        VIT_patch_sizes=CONFIG['VIT_patch_sizes'],
        VIT_num_groups=CONFIG['VIT_num_groups'],
        VIT_num_heads=CONFIG['VIT_num_heads'],
        VIT_emb_size=CONFIG['VIT_emb_size'],
        Unet_num_blocks=CONFIG['Unet_num_blocks'],
        Unet_channel_mult=CONFIG['Unet_channel_mult'],
        Unet_channel_mult_emb=CONFIG['Unet_channel_mult_emb'],
        Unet_kernel_sizes=CONFIG['Unet_kernel_sizes'],
        Unet_model_channels=CONFIG['Unet_model_channels'],
        sigma_data=CONFIG["sigma_data"],
        log_var_channels=CONFIG['log_var_channels']
    ).to(device)

    checkpoint_path = r"C:\Users\Maha Mamdouh\PycharmProjects\Heterogeneous-MOE-for-Diffusion-models\Utils\final_model1.pt"
    print(f"Loading weights from {checkpoint_path}...")
    ckeck = torch.load(map_location=device, f=checkpoint_path)
    modeli.load_state_dict(ckeck['model_state_dict'])
    modeli.eval()

    clip = CLIP_EMBED()
    clip.init()

    vae = StabilityVAE(batch_size=CONFIG["batch_size"])
    vae.init(device=device)

    SAMPLER = EDM_Sampler(modeli, Guide_net=modeli, guidance=1.0, num_solve_steps=40)

    # Prepare conditioning
    fixed_prompts = [CONFIG["fixed_prompt"]] * CONFIG["batch_size"]
    with torch.no_grad():
        base_text_emb = clip.encode_text(fixed_prompts)

    # Generate noise
    latent_noise = torch.randn(
        (CONFIG["batch_size"], CONFIG["img_channels"], CONFIG["img_resolution"], CONFIG["img_resolution"]),
        device=device
    )

    print("Sampling...")
    with torch.no_grad():
        latents_denoised = SAMPLER.sample(noise=latent_noise, text_emb=base_text_emb,transition_mean=-1.2,softness=1.2)
        out_images = vae.decode(latents_denoised)

    # Save
    print("Saving images...")
    os.makedirs("generated_samples_unguided", exist_ok=True)

    images_normalized = out_images.float() / 255.0
    images_normalized = images_normalized.clamp(0, 1)
    #test_diffusion_pipeline(modeli,vae,clip,r"C:\Users\Maha Mamdouh\PycharmProjects\Heterogeneous-MOE-for-Diffusion-models\Utils\data\flowers-102\jpg\image_00009.jpg",)
    # Save grid
    save_image(images_normalized, "generated_samples_unguided/grid_unguided.png", nrow=4)

    # Save individual images
    for i, img in enumerate(images_normalized):
        save_image(img, f"generated_samples_unguided/sample_{i}.png")

    print("Done.")

if __name__ == '__main__':
    training_HDMOE(model_configs,
                   optim_configs,
                   loss_configs,
                   mask_configs,
                   zeta_configs)
    #sample_and_save(model_configs)