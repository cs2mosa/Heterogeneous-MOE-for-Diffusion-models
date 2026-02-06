import torch

model_configs = {
        "save_dir": "./checkpoints/flower_run",
        "save_dir_stats": r"C:\Users\Maha Mamdouh\PycharmProjects\PythonProject1\Utils",
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'img_channels': 4,
        'internal_channels': 32,
        "data_img_res": 256,
        'img_resolution': 32,
        'time_emb_dim': 64,
        'text_emb_dim': 768,
        'num_experts': 4,
        'top_k': 1,
        'fourier_bandwidth': 1.0,
        'VIT_num_blocks': 4,
        'VIT_patch_sizes': [4, 8, 8, 16],
        'VIT_num_groups': 4,
        'VIT_num_heads': 8,
        'VIT_emb_size': 32,
        'Unet_num_blocks': 2,
        'Unet_channel_mult': [1, 2],
        'Unet_kernel_sizes': [(3, 3), (3, 3), (5, 5), (5, 5)],
        'Unet_model_channels': 32,
        'Unet_channel_mult_emb': 2,
        'Unet_label_balance': 0.5,
        'Unet_concat_balance': 0.5,
        'sigma_data': 0.5,
        'log_var_channels': 32,
        'batch_size': 32,
        'total_steps': 5000,
        'sigma_min': 0.002,
        'sigma_max': 80,
        "fixed_prompt": "a photo of a flower"
}
loss_configs = {
    'unet_bal': 0.05,
    'vit_bal': 0.1,
    'z_bal': 0.005,
    'prior_bal': 0.0,
}
optim_configs = {
    'eta_min': 1e-5,
    'lr_vit': 2e-3,
    'lr_unet': 5e-4,
    'lr_attn':1e-3,
    'lr_router': 5e-4,
    'total_schedule_steps': 5000 # steepness calculation depends on this
}
mask_configs ={
    'unet_attr': [3, 3, 5, 5],
    'vit_attr': [4, 8, 8, 16],
    'p_mean': -1.2,  # should be tuned
    'p_std': 1.6,  # should be tuned
    'BW': 0.3,
    'max_BW': 0.8,# should be tuned
    'min_active': 1,
    'step_size': 0.1,
    'strat_band': 'step',
    'unet_noise_range': (0.0 , 0.6),
    'vit_noise_range': (0.4 , 1.0)
}
zeta_configs= {
    'min_zeta': 0.01,
    'max_zeta': 2,
    'warmup_ratio': 0.05,
    "strategy": 'cos',
    'alpha': 4.0,
    'total_schedule_steps': 900 # steepness calculation depends on this
}