from typing import Any
from models.model_config2 import preconditioned_HDMOEM
import torch
from utils import EDM_LOSS, sample_sigma,ZetaScheduler,MaskGenerator
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

    SAMPLER = EDM_Sampler(modeli, Guide_net=modeli, guidance=1.0, num_solve_steps=60)

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
    test_diffusion_pipeline(modeli,vae,clip,r"C:\Users\Maha Mamdouh\PycharmProjects\Heterogeneous-MOE-for-Diffusion-models\Utils\data\flowers-102\jpg\image_00002.jpg",)
    # Save grid
    save_image(images_normalized, "generated_samples_unguided/grid_unguided.png", nrow=4)

    # Save individual images
    for i, img in enumerate(images_normalized):
        save_image(img, f"generated_samples_unguided/sample_{i}.png")

    print("Done.")

import matplotlib.pyplot as plt
from pathlib import Path


def test_diffusion_pipeline(
        model,
        vae,
        clip,
        test_image_path: str,
        output_dir: str = "./pipeline_test",
        device: str = "cuda",
        sigma_test: float = 0.5,
        num_experts: int = 4,
):
    """
    Complete end-to-end test of the diffusion pipeline.

    Args:
        model: Your diffusion model
        vae: Your VAE (StabilityVAE instance)
        clip: Your CLIP embedder
        test_image_path: Path to a test flower image
        output_dir: Where to save results
        device: 'cuda' or 'cpu'
        sigma_test: Noise level to test (default 1.0 = medium noise)
        num_experts: Number of experts in your MoE model

    Returns:
        dict with diagnostic info
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure models are in eval mode
    model.eval()
    vae.init(device)
    clip.init()

    results = {}

    print("\n" + "=" * 80)
    print("DIFFUSION PIPELINE DIAGNOSTIC TEST")
    print("=" * 80)

    # ===== STEP 1: LOAD ORIGINAL IMAGE =====
    print("\n[1/6] Loading original image...")
    from PIL import Image
    import torchvision.transforms as T

    img = Image.open(test_image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    img_tensor = (transform(img).unsqueeze(0) * 255.0).to(device)

    save_image(img_tensor / 255.0, output_dir / "step1_original.png")

    results['original'] = {
        'shape': img_tensor.shape,
        'min': img_tensor.min().item(),
        'max': img_tensor.max().item(),
        'mean': img_tensor.mean().item(),
    }

    print(f"  ✓ Original image: shape={img_tensor.shape}")
    print(f"    Range: [{img_tensor.min():.1f}, {img_tensor.max():.1f}]")
    print(f"    Saved: step1_original.png")

    # ===== STEP 2: VAE ENCODE =====
    print("\n[2/6] Encoding with VAE...")

    with torch.no_grad():
        latent = vae.encode(img_tensor)

    results['encoded_latent'] = {
        'shape': latent.shape,
        'mean': latent.mean().item(),
        'std': latent.std().item(),
        'min': latent.min().item(),
        'max': latent.max().item(),
    }

    print(f"  ✓ Encoded latent: shape={latent.shape}")
    print(f"    Mean: {latent.mean():.3f}, Std: {latent.std():.3f}")
    print(f"    Expected: shape=(1,4,32,32), mean≈0, std≈0.5")

    # Check if latent stats are reasonable
    if abs(latent.mean().item()) > 1.0:
        print(f"  ⚠️  WARNING: Latent mean {latent.mean():.3f} is far from 0!")
    if latent.std().item() < 0.1 or latent.std().item() > 2.0:
        print(f"  ⚠️  WARNING: Latent std {latent.std():.3f} is unusual!")

    # Visualize latent channels
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        axes[i].imshow(latent[0, i].cpu(), cmap='viridis')
        axes[i].set_title(f'Latent Channel {i}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'step2_latent_channels.png', dpi=150)
    plt.close()
    print(f"    Saved: step2_latent_channels.png")

    # ===== STEP 3: VAE DECODE (No diffusion) =====
    print("\n[3/6] Decoding latent (testing VAE round-trip)...")

    with torch.no_grad():
        reconstructed = vae.decode(latent)

    # Handle uint8 vs float
    if reconstructed.dtype == torch.uint8:
        reconstructed_float = reconstructed.float() / 255.0
    else:
        reconstructed_float = reconstructed.clamp(0, 1)

    save_image(reconstructed_float, output_dir / "step3_vae_reconstruction.png")

    results['vae_reconstruction'] = {
        'dtype': str(reconstructed.dtype),
        'min': reconstructed_float.min().item(),
        'max': reconstructed_float.max().item(),
    }

    print(f"  ✓ VAE reconstruction: dtype={reconstructed.dtype}")
    print(f"    Range: [{reconstructed_float.min():.3f}, {reconstructed_float.max():.3f}]")
    print(f"    Saved: step3_vae_reconstruction.png")
    print(f"  ⚠️  CRITICAL CHECK: Does step3_vae_reconstruction.png have grid artifacts?")
    print(f"      If YES → VAE decoder is broken")
    print(f"      If NO → Problem is elsewhere")

    # ===== STEP 4: ADD NOISE =====
    print(f"\n[4/6] Adding noise (σ={sigma_test})...")

    noise = torch.randn_like(latent) * sigma_test
    noisy_latent = latent + noise

    results['noisy_latent'] = {
        'mean': noisy_latent.mean().item(),
        'std': noisy_latent.std().item(),
    }

    print(f"  ✓ Noisy latent: mean={noisy_latent.mean():.3f}, std={noisy_latent.std():.3f}")

    # Decode noisy version for comparison
    with torch.no_grad():
        noisy_decoded = vae.decode(noisy_latent)
    if noisy_decoded.dtype == torch.uint8:
        noisy_decoded = noisy_decoded.float() / 255.0
    else:
        noisy_decoded = noisy_decoded.clamp(0, 1)

    save_image(noisy_decoded, output_dir / "step4_noisy.png")
    print(f"    Saved: step4_noisy.png (should look corrupted)")

    # ===== STEP 5: MODEL DENOISE =====
    print("\n[5/6] Denoising with trained model...")

    # Prepare inputs
    sigma_tensor = torch.full((1, 1, 1, 1), sigma_test, device=device)
    text_emb = clip.encode_text(["a beautiful flower"])

    # Router masks (all experts active)
    unet_mask = torch.ones(1, num_experts, device=device)
    vit_mask = torch.ones(1, num_experts, device=device)

    with torch.no_grad():
        out = model(
            x=noisy_latent,
            sigma=sigma_tensor,
            text_emb=text_emb,
            Unet_router_mask=unet_mask,
            Vit_router_mask=vit_mask,
            zeta=0.0,
            transition_point = -1.2,
            softness = 1.2
        )
        denoised_latent = out["denoised"]

    results['denoised_latent'] = {
        'mean': denoised_latent.mean().item(),
        'std': denoised_latent.std().item(),
        'min': denoised_latent.min().item(),
        'max': denoised_latent.max().item(),
    }

    print(f"  ✓ Denoised latent: mean={denoised_latent.mean():.3f}, std={denoised_latent.std():.3f}")

    # Check if denoising moved toward original
    dist_noisy_to_original = (noisy_latent - latent).pow(2).mean().sqrt().item()
    dist_denoised_to_original = (denoised_latent - latent).pow(2).mean().sqrt().item()

    results['denoising_quality'] = {
        'noisy_to_original_dist': dist_noisy_to_original,
        'denoised_to_original_dist': dist_denoised_to_original,
        'improvement': dist_noisy_to_original - dist_denoised_to_original,
    }

    print(f"    Distance to original:")
    print(f"      Noisy:    {dist_noisy_to_original:.4f}")
    print(f"      Denoised: {dist_denoised_to_original:.4f}")

    if dist_denoised_to_original < dist_noisy_to_original:
        improvement = ((dist_noisy_to_original - dist_denoised_to_original) / dist_noisy_to_original) * 100
        print(f"    ✓ Model improved by {improvement:.1f}%")
    else:
        print(f"    ❌ WARNING: Model made it WORSE! Not learning to denoise!")

    # Decode denoised
    with torch.no_grad():
        denoised_img = vae.decode(denoised_latent)
    if denoised_img.dtype == torch.uint8:
        denoised_img = denoised_img.float() / 255.0
    else:
        denoised_img = denoised_img.clamp(0, 1)

    save_image(denoised_img, output_dir / "step5_denoised.png")
    print(f"    Saved: step5_denoised.png")

    # ===== STEP 6: FULL SAMPLER TEST =====
    print("\n[6/6] Testing full sampler (from pure noise)...")

    from EDM_sampler import EDM_Sampler

    sampler = EDM_Sampler(
        model=model,
        Guide_net=model,
        num_solve_steps=60,  # Fewer steps for quick test
        sigma_min=0.002,
        sigma_max=80.0,
        guidance=1.0,
    )

    # Generate from pure noise
    pure_noise = torch.randn(1, 4, 32, 32, device=device)

    with torch.no_grad():
        sampled_latent = sampler.sample(
            noise=pure_noise,
            text_emb=text_emb,
            transition_mean=-1.2,
            softness=1.2,
            uncond_text_emb=None,
        )

    results['sampled_latent'] = {
        'mean': sampled_latent.mean().item(),
        'std': sampled_latent.std().item(),
    }

    print(f"  ✓ Sampled latent: mean={sampled_latent.mean():.3f}, std={sampled_latent.std():.3f}")

    # Decode sampled
    with torch.no_grad():
        sampled_img = vae.decode(sampled_latent)
    if sampled_img.dtype == torch.uint8:
        sampled_img = sampled_img.float() / 255.0
    else:
        sampled_img = sampled_img.clamp(0, 1)

    save_image(sampled_img, output_dir / "step6_sampled_from_noise.png")
    print(f"    Saved: step6_sampled_from_noise.png")
    print(f"  ⚠️  This should resemble a flower (even if blurry)")

    # ===== CREATE COMPARISON FIGURE =====
    print("\n[7/7] Creating comparison figure...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img_tensor[0].cpu().permute(1, 2, 0) / 255.0)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(reconstructed_float[0].cpu().permute(1, 2, 0))
    axes[0, 1].set_title('VAE Reconstruction\n(No Diffusion)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(noisy_decoded[0].cpu().permute(1, 2, 0))
    axes[0, 2].set_title(f'Noisy (σ={sigma_test})', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(denoised_img[0].cpu().permute(1, 2, 0))
    axes[1, 0].set_title('Model Denoised\n(Single Step)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(sampled_img[0].cpu().permute(1, 2, 0))
    axes[1, 1].set_title('Sampled from Noise\n(Full Diffusion)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Add metrics text
    metrics_text = f"""
    LATENT STATISTICS
    Original: μ={results['encoded_latent']['mean']:.3f}, σ={results['encoded_latent']['std']:.3f}
    Noisy: μ={results['noisy_latent']['mean']:.3f}, σ={results['noisy_latent']['std']:.3f}
    Denoised: μ={results['denoised_latent']['mean']:.3f}, σ={results['denoised_latent']['std']:.3f}
    Sampled: μ={results['sampled_latent']['mean']:.3f}, σ={results['sampled_latent']['std']:.3f}

    DENOISING QUALITY
    Distance (noisy→original): {results['denoising_quality']['noisy_to_original_dist']:.4f}
    Distance (denoised→original): {results['denoising_quality']['denoised_to_original_dist']:.4f}
    Improvement: {results['denoising_quality']['improvement']:.4f}
    """
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                    verticalalignment='center', transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'full_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: full_comparison.png")

    # ===== FINAL REPORT =====
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    print("\n✓ FILES SAVED:")
    print(f"  • step1_original.png")
    print(f"  • step2_latent_channels.png")
    print(f"  • step3_vae_reconstruction.png  ← CHECK FOR GRID ARTIFACTS")
    print(f"  • step4_noisy.png")
    print(f"  • step5_denoised.png")
    print(f"  • step6_sampled_from_noise.png  ← CHECK IF LOOKS LIKE FLOWER")
    print(f"  • full_comparison.png")

    print("\n⚠️  CRITICAL CHECKS:")
    print(f"  1. VAE Reconstruction (step3):")
    if abs(results['encoded_latent']['mean']) > 1.0:
        print(f"     ❌ Latent mean {results['encoded_latent']['mean']:.3f} is far from 0")
    else:
        print(f"     ✓ Latent statistics look reasonable")
    print(f"     → Does step3_vae_reconstruction.png have grid artifacts?")
    print(f"       If YES: VAE decoder is broken")
    print(f"       If NO: Problem is in diffusion model or data")

    print(f"\n  2. Model Denoising (step5):")
    if results['denoising_quality']['improvement'] > 0:
        print(f"     ✓ Model reduced distance by {results['denoising_quality']['improvement']:.4f}")
    else:
        print(f"     ❌ Model INCREASED distance (not learning!)")

    print(f"\n  3. Full Sampling (step6):")
    print(f"     → Does step6_sampled_from_noise.png look like a flower?")
    print(f"       If YES: Model is working, something wrong with your generation script")
    print(f"       If NO: Model not learning to generate")

    print("\n" + "=" * 80)

    return results

if __name__ == '__main__':
    """training_HDMOE(model_configs,
                   optim_configs,
                   loss_configs,
                   mask_configs,
                   zeta_configs)"""
    sample_and_save(model_configs)