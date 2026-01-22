from typing import Optional
import numpy as np
import torch
import torch.nn as nn

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

    return sigma_vec

def load_balance(gate_probs, num_experts):
    P = gate_probs.mean(dim=0)
    return num_experts * torch.sum(P**2)

def entropy_loss(probs):
    return -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))

def z_loss(logits):
    z = torch.logsumexp(logits,dim = -1) ** 2
    return torch.mean(z)

class EDM_LOSS(nn.Module):
    def __init__(self,
                 num_experts: int,
                 sigma_data:float = 0.5,
                 Unet_bal: float = 0.0005,
                 vit_bal: float = 0.0005,
                 scale_bal: float = 0.0001,
                 z_bal: float = 0.0001
                 ):

        super().__init__()
        self.num_experts = num_experts
        self.sigma_data = sigma_data
        self.Unet_lambda = Unet_bal
        self.vit_lambda = vit_bal
        self.scale_bal = scale_bal
        self.z_bal = z_bal

    def __call__(self,
                 sigma_vec: torch.Tensor,
                 x: torch.Tensor,
                 out_model: dict[str,torch.Tensor]
                 )->dict[str,torch.Tensor]:

        lamda = (sigma_vec ** 2 + self.sigma_data ** 2)/ (sigma_vec * self.sigma_data) ** 2
        if out_model["log_var"] is None:
            pure_loss = torch.mean(lamda * ((out_model["denoised"] - x) ** 2))
        else:
            pure_loss = torch.mean((lamda* ((out_model["denoised"] - x) ** 2)/out_model["log_var"].exp()) + out_model["log_var"])

        router_loss = self.Unet_lambda * load_balance(out_model["Unet_router_loss"],self.num_experts) + self.vit_lambda * load_balance(out_model["vit_router_loss"],self.num_experts)
        scaling_loss = self.scale_bal * entropy_loss(out_model["scaling_net_out"])
        z_los = self.z_bal * z_loss(out_model["Unet_raw"]) + self.z_bal * z_loss(out_model["vit_raw"])
        total_loss = pure_loss + z_los+  router_loss - scaling_loss

        return {
            "loss": total_loss,
            "denoising": pure_loss,
            "balance": router_loss,
            "z_loss": z_los,
            "entropy": scaling_loss
        }

class ZetaScheduler:
    def __init__(self,
                 total_steps: int,
                 max_zeta: float,
                 min_zeta: float = 0.0,
                 strategy: str = 'cos',
                 alpha: float = 4.0,  # Controls steepness of exponential
                 warmup_ratio: float = 0.05):
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
            p_mean: float = -0.4,  # Must match  sample_sigma mean
            p_std: float = 1.0,  # Must match sample_sigma std
            bandwidth: float = 0.3,  # Width of the sliding window of experts
            min_active: int = 1
    ):
        """
        Attributes-Aware Mask Generator.
        Maps experts to noise levels based on their physical scale (Kernel/Patch size).
        """
        super().__init__()
        self.p_mean = p_mean
        self.p_std = p_std
        self.bandwidth = bandwidth
        self.min_active = min_active
        attrs = torch.tensor(expert_attributes, dtype=torch.float32)
        a_min, a_max = attrs.min(), attrs.max()
        normalized_centers = (attrs - a_min) / (a_max - a_min + 1e-8)
        self.register_buffer('expert_centers', normalized_centers)

    @torch.no_grad()
    def __call__(self, sigma: torch.Tensor):
        """
        sigma: (Batch,) - Current noise levels
        Returns mask: (Batch, Num_Experts)
        """
        device = sigma.device
        s = sigma.flatten()
        log_sigma = torch.log(s)
        sigma_percentile = 0.5 * (1 + torch.erf((log_sigma - self.p_mean) / (self.p_std * np.sqrt(2))))
        sigma_percentile = sigma_percentile.clamp(0, 1)  # (Batch,)
        dist = torch.abs(sigma_percentile.unsqueeze(1) - self.expert_centers.to(device).unsqueeze(0))
        mask = (dist <= self.bandwidth).float()
        _, top_indices = torch.topk(-dist, k=self.min_active, dim=-1)
        mask.scatter_(1, top_indices, 1.0)
        return mask

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Config
    TOTAL_STEPS = 10000
    MAX_ZETA = 50.0
    MIN_ZETA = 0.1
    ALPHA = 0.005  # <--- TWEAK THIS for your exponential formula

    scheduler_cos = ZetaScheduler(TOTAL_STEPS, MAX_ZETA, MIN_ZETA, strategy='cosine')
    # Note: Your formula is sensitive to the magnitude of `step`.
    # Since step goes 0->1000, alpha needs to be small (e.g. 0.005 or 0.01)
    scheduler_exp = ZetaScheduler(TOTAL_STEPS, MAX_ZETA, MIN_ZETA, strategy='exponential', alpha=ALPHA)

    steps = list(range(TOTAL_STEPS))
    zetas_cos = [scheduler_cos.get_zeta(s) for s in steps]
    zetas_exp = [scheduler_exp.get_zeta(s) for s in steps]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, zetas_cos, label='Cosine (Orange)', color='orange')
    plt.plot(steps, zetas_exp, label=f'Exponential (Green) alpha={ALPHA}', color='teal')
    plt.title(f"Zeta Schedulers (Max: {MAX_ZETA})")
    plt.xlabel("Training Step")
    plt.ylabel("Zeta Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()