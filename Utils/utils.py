from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import math
def sample_sigma(batch_size: int,
                 uniform: bool = False,
                 p_mean: float = -0.4,
                 p_std: float = 1.0,
                 sigma_max: Optional[float] = 80.0,
                 sigma_min: Optional[float] = 0.002
                 )-> torch.Tensor:

    if uniform:
        low = torch.log(torch.tensor(sigma_min))
        high = torch.log(torch.tensor(sigma_max))
        rnd_uniform = torch.rand([batch_size, 1, 1, 1])
        sigma_vec = (rnd_uniform * (high - low) + low).exp()
    else:
        sigma_vec = torch.randn([batch_size, 1, 1, 1])
        sigma_vec = (sigma_vec * p_std + p_mean).exp()
        sigma_vec = sigma_vec.clamp(min=sigma_min, max=sigma_max)

    return sigma_vec

def sample_sigma_hybrid(batch_size,
                        sigma_min=0.002,
                        sigma_max=80.0,
                        p_mean=-0.4,  # Adjusted for Latent Space
                        p_std=1.0,  # Adjusted for Latent Space
                        extreme_prob=0.2,
                        device='cuda'):
    """
    Hybrid Sampler:
    Ensures 'Core' training via Log-Normal (80%)
    and 'Expert Coverage' via Log-Uniform (20%).
    """

    # 1. Calculate counts
    n_lognormal = int(batch_size * (1 - extreme_prob))
    n_uniform = batch_size - n_lognormal

    # 2. Log-Normal Samples (The EDM2 core)
    # Focuses on the most informative noise levels for the average expert
    rnd_normal = torch.randn([n_lognormal, 1, 1, 1], device=device)
    sigma_lognormal = (rnd_normal * p_std + p_mean).exp()

    # 3. Log-Uniform Samples (The MoE safety floor)
    # Ensures Structural and Detail experts get consistent gradients
    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    u = torch.rand([n_uniform, 1, 1, 1], device=device)
    sigma_uniform = (u * (log_max - log_min) + log_min).exp()

    # 4. Combine, Clamp, and Shuffle
    # Shuffling is vital so the batch doesn't have 'easy' and 'hard' halves
    sigma = torch.cat([sigma_lognormal, sigma_uniform], dim=0)
    sigma = sigma.clamp(sigma_min, sigma_max)

    perm = torch.randperm(batch_size, device=device)
    return sigma[perm]

def lr_scheduler()->float:
    pass


class PathPriorLoss(nn.Module):
    def __init__(self,
                 transition_sigma: float = 1.0,
                 sharpness: float = 2.0):
        """
        Enforces Inductive Bias:
        - High Noise (> transition_sigma) -> Prefer ViT (Index 0)
        - Low Noise (< transition_sigma)  -> Prefer U-Net (Index 1)

        Args:
            transition_sigma: The noise level where the preference flips (50/50).
            sharpness: How strictly to enforce the split. Higher = Harder switch.
        """
        super().__init__()
        self.transition = transition_sigma
        self.sharpness = sharpness
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self,
                scaling_factors: torch.Tensor,
                sigma: torch.Tensor):
        """
        Args:
            scaling_factors: (Batch, 2) Output from Scaling_router (Sums to 2).
                             Index 0 = ViT, Index 1 = U-Net.
            sigma: (Batch,) Noise levels.
        """
        probs = scaling_factors / 2.0
        sigma_flat = sigma.flatten()
        log_sigma = torch.log(sigma_flat + 1e-8)
        target_logits = (log_sigma - math.log(self.transition)) * self.sharpness
        target_vit_prob = torch.sigmoid(target_logits)
        target_dist = torch.stack([target_vit_prob, 1.0 - target_vit_prob], dim=1)
        log_probs = torch.log(probs + 1e-8)

        return self.kl_div(log_probs, target_dist)


class EDM_LOSS(nn.Module):
    def __init__(self,
                 num_experts: int,
                 sigma_data:float = 0.5,
                 Unet_bal: float = 0.0005,
                 vit_bal: float = 0.0005,
                 z_bal: float = 0.0001,
                 prior_bal:float = 0.001,
                 transition_sigma: float = 1.0,
                 sharpness: float = 2.0
                 ):

        super().__init__()
        self.num_experts = num_experts
        self.sigma_data = sigma_data
        self.Unet_lambda = Unet_bal
        self.vit_lambda = vit_bal
        self.z_bal = z_bal
        self.prior_bal = prior_bal
        self.prior_path_loss = PathPriorLoss(transition_sigma=transition_sigma,
                                             sharpness=sharpness)

    def __call__(self,
                 sigma_vec: torch.Tensor,
                 x: torch.Tensor,
                 sigma: torch.Tensor,
                 out_model: dict[str,torch.Tensor]
                 )->dict[str,torch.Tensor]:

        lamda = 1
        #lamda = (sigma_vec ** 2 + self.sigma_data ** 2)/ (sigma_vec * self.sigma_data) ** 2
        if out_model["log_var"] is None:
            pure_loss = torch.mean(lamda * ((out_model["denoised"] - x) ** 2))
        else:
            log_var = out_model["log_var"].clamp(min=-10, max=10)
            pure_loss = torch.mean((lamda* ((out_model["denoised"] - x) ** 2)) / log_var.exp() + log_var)
        pure_loss = pure_loss.clamp(max=50)

        denoising_loss = torch.mean((out_model["denoised"] - x) ** 2)
        router_loss = (self.Unet_lambda * self.load_balance(out_model["Unet_router_loss"],self.num_experts) + self.vit_lambda * self.load_balance(out_model["vit_router_loss"],self.num_experts)).clamp(max=50)
        #scaling_loss = (self.prior_bal * self.prior_path_loss(out_model["scaling_net_out"],sigma)).clamp(max=50)
        z_los = (self.z_bal * self.z_loss(out_model["Unet_raw"]) + self.z_bal * self.z_loss(out_model["vit_raw"])).clamp(max=50)
        total_loss = (pure_loss + z_los + router_loss ).clamp(max=50) #+ scaling_loss

        return {
            "loss": total_loss,
            "denoising": denoising_loss,
            "balance": router_loss,
            "z_loss": z_los,
            "entropy": 0.0,
            "pure_loss": pure_loss
        }

    @staticmethod
    def load_balance(gate_probs, num_experts):
        P = gate_probs.mean(dim=0)
        return num_experts * torch.sum(P ** 2)

    @staticmethod
    def entropy_loss(probs):
        return -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))

    @staticmethod
    def z_loss(logits):
        logits = logits.clamp(min=-50, max=50)
        z = torch.logsumexp(logits, dim=-1) ** 2
        z = z.clamp(max=100)
        return torch.mean(z)


class ZetaScheduler:
    def __init__(
            self,
            total_steps: int,
            max_zeta: float,
            min_zeta: float = 0.0,
            strategy: str = 'cos',
            alpha: float = 4.0,  # Controls steepness of exponential
            warmup_ratio: float = 0.05
    ):
        """
        Args:
            total_steps: Total training steps.
            max_zeta: Starting value (Exploration).
            min_zeta: Ending value (Exploitation).
            strategy: 'cosine' or 'exponential' (your formula).
            alpha: Decay rate for exponential strategy.
            warmup_ratio: Percentage of steps (0.0 to 1.0) to hold zeta at max_zeta.
        """
        self.total_steps = total_steps
        self.max_zeta = max_zeta
        self.min_zeta = min_zeta
        self.strategy = strategy
        self.alpha = alpha
        self.warmup_steps = int(total_steps * warmup_ratio)

    def get_zeta(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.max_zeta

        if step >= self.total_steps:
            return self.min_zeta

        curr_step = step - self.warmup_steps
        decay_total = self.total_steps - self.warmup_steps

        if self.strategy == 'cos':
            cosine_val = 0.5 * (1 + np.cos(np.pi * curr_step / decay_total))
            zeta = self.min_zeta + (self.max_zeta - self.min_zeta) * cosine_val

        elif self.strategy == 'exp':
            term = -self.alpha * (curr_step - (self.max_zeta / decay_total))
            #Safety clamp for exp to prevent overflow if alpha is huge
            term = max(min(term, 10), -10)
            decay_factor = np.exp(term)
            zeta = (self.max_zeta - self.min_zeta) * decay_factor + self.min_zeta
            zeta = max(min(zeta, self.max_zeta), self.min_zeta)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return float(zeta)


class MaskGenerator(nn.Module):
    def __init__(
            self,
            expert_attributes: list,  # List of kernel sizes or patch sizes
            p_mean: float = -0.4,
            p_std: float = 1.0,
            bandwidth: float = 0.3,
            max_bandwidth: float = 0.9,
            min_active: int = 1,
            total_steps: int = 5000,
            step_size: float = 0.1,
            noise_range: tuple = (0.0, 1.0),
            strat_band: str = "step",
    ):
        """
        Attributes-Aware Mask Generator (Rank-Based).
        Maps experts to noise levels based on their RANK of physical scale.
        Ensures equidistant spacing even if attributes are identical.
        """
        super().__init__()
        self.num_intervals = len(expert_attributes)
        self.strat_band = strat_band
        self.total_steps = total_steps
        self.max_bw = max_bandwidth
        self.step_size = step_size
        self.p_mean = p_mean
        self.p_std = p_std
        self.bandwidth = bandwidth
        self.min_active = min_active

        # --- NEW INITIALIZATION LOGIC ---
        attrs = torch.tensor(expert_attributes, dtype=torch.float32)
        num_experts = len(attrs)

        # 1. Get the rank of each expert based on attributes
        # stable=True ensures that if two experts have the same kernel (e.g., 3, 3),
        # the one that appears first in the list gets the lower sigma center.
        _, sorted_indices = torch.sort(attrs, stable=True)

        # 2. Create evenly spaced centers from 0.0 to 1.0
        # Example for 4 experts: [0.00, 0.33, 0.66, 1.00]
        min_range, max_range = noise_range
        equidistant_points = torch.linspace(min_range, max_range, steps=num_experts)

        # 3. Map these perfect centers back to the original expert indices
        # We create a placeholder and scatter the points into the sorted positions
        final_centers = torch.zeros_like(attrs)
        final_centers[sorted_indices] = equidistant_points

        self.register_buffer('expert_centers', final_centers)
        # --------------------------------

    @torch.no_grad()
    def __call__(self,
                 sigma: torch.Tensor,
                 step: int
                 ) -> torch.Tensor:
        """
        sigma: (Batch,) - Current noise levels
        Returns mask: (Batch, Num_Experts)
        """
        device = sigma.device
        s = sigma.flatten()
        log_sigma = torch.log(s)
        sigma_percentile = 0.5 * (1 + torch.erf((log_sigma - self.p_mean) / (self.p_std * np.sqrt(2))))
        sigma_percentile = sigma_percentile.clamp(0, 1)  # (Batch,)

        # Reshape for broadcasting: (Batch, 1) - (1, Num_Experts)
        sp_reshaped = sigma_percentile.view(-1, 1)
        experts_reshaped = self.expert_centers.to(device).view(1, -1)

        # Calculate distance
        dist = torch.abs(sp_reshaped - experts_reshaped)

        # Dynamic Bandwidth
        current_bw = self.bandwidth_scheduler(step)
        mask = (dist <= current_bw).float()

        # Safety: Ensure min_active experts are always selected
        _, top_indices = torch.topk(-dist, k=self.min_active, dim=-1)
        mask.scatter_(1, top_indices, 1.0)
        return mask

    def bandwidth_scheduler(self, step: int) -> float:
        if step >= self.total_steps:
            return self.max_bw  # Changed from hardcoded 1.0 to max_bw

        if self.strat_band == 'linear':
            prog = step / float(self.total_steps)
            return self.bandwidth + (self.max_bw - self.bandwidth) * prog

        elif self.strat_band == 'step':  # Fixed variable name typo (self.strategy -> self.strat_band)
            # Calculate interval based on total steps / num_intervals (or step_size fraction)
            # Using step_size as a fraction (e.g. 0.1 for 10%)
            interval_size = self.total_steps * self.step_size
            current_interval = int(step / interval_size)

            # Max progress is 1.0
            total_intervals = int(1.0 / self.step_size)
            progress = min(current_interval / total_intervals, 1.0)

            return self.bandwidth + (self.max_bw - self.bandwidth) * progress
        else:
            return self.bandwidth