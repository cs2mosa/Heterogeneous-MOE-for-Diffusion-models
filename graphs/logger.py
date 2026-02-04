"""
Enhanced Training Logger for Hybrid Diffusion MoE Models
=========================================================

This module provides comprehensive logging utilities for tracking all critical
metrics during training. Compatible with the ProfessionalDiffusionPlotter.

Based on best practices from:
- EDM2 (NVIDIA)
- MoE literature (Switch Transformer, DeepSeek)
- Recent diffusion model papers
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict


class Logger:
    """
    Comprehensive training logger for diffusion MoE models.
    
    Tracks:
    - Core losses (EDM, MSE, uncertainty)
    - MoE auxiliary losses (balance, z-loss, entropy)
    - Router statistics (usage, entropy, collapse metrics)
    - Scaling/gating behavior
    - Gradient norms
    - Weight statistics
    - Learning rate schedule
    """
    
    def __init__(self, 
                 log_dir: str = "./training_logs",
                 run_name: str = "experiment",
                 log_interval: int = 10):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            run_name: Name of this training run
            log_interval: Log every N steps (reduce overhead)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_name = run_name
        self.log_interval = log_interval
        
        # File paths
        self.main_log_file = self.log_dir / f"{run_name}_training.jsonl"
        self.router_log_file = self.log_dir / f"{run_name}_router_stats.jsonl"
        self.gradient_log_file = self.log_dir / f"{run_name}_gradients.jsonl"
        self.weight_log_file = self.log_dir / f"{run_name}_weights.jsonl"
        
        # Accumulators for averaging over interval
        self.accumulators = defaultdict(list)
        
        print(f"✓ Initialized Enhanced Logger: {run_name}")
        print(f"  Main log: {self.main_log_file}")
        print(f"  Router stats: {self.router_log_file}")
        print(f"  Gradients: {self.gradient_log_file}")
        print(f"  Weights: {self.weight_log_file}")
    
    def log_training_step(self,
                         step: int,
                         loss_dict: Dict[str, torch.Tensor],
                         zeta: float,
                         log_var: float,
                         lr: float,
                         p_mean:float,
                         p_std: float,
                         sigma: Optional[torch.Tensor] = None):
        """
        Log main training metrics.
        
        Args:
            step: Current training step
            loss_dict: Dictionary with keys:
                - 'loss': total loss
                - 'denoising': raw MSE/denoising loss
                - 'pure_loss': pure EDM loss (before auxiliary)
                - 'balance': load balancing loss
                - 'z_loss': router z-loss
                - 'entropy': entropy regularization
            zeta: Router exploration noise
            log_var: Predicted log variance
            lr: Current learning rate
            sigma: Noise levels (optional, for percentile tracking)
        """
        # Accumulate for interval averaging
        self.accumulators['step'].append(step)
        
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                self.accumulators[key].append(value.item())
            else:
                self.accumulators[key].append(float(value))
        
        self.accumulators['zeta'].append(float(zeta))
        self.accumulators['log_var'].append(float(log_var))
        self.accumulators['lr'].append(float(lr))
        
        if sigma is not None:
            # Track average sigma percentile
            log_sigma = torch.log(sigma)
            sigma_pct = 0.5 * (1 + torch.erf(
                (log_sigma - p_mean) / (p_std * np.sqrt(2))
            ))
            self.accumulators['avg_sigma_percentile'].append(
                sigma_pct.mean().item()
            )
        
        # Write to file every log_interval steps
        if step % self.log_interval == 0 and len(self.accumulators['step']) > 0:
            self._flush_training_log()
    
    def log_router_statistics(self,
                              step: int,
                              unet_probs: torch.Tensor,
                              vit_probs: torch.Tensor,
                              p_mean: float,
                              p_std: float,
                              sigma: torch.Tensor):
        """
        Log detailed router statistics.
        
        Args:
            step: Current training step
            unet_probs: UNet router probabilities (B, num_experts)
            vit_probs: ViT router probabilities (B, num_experts)
            sigma: Noise levels (B,)
        """
        if step % self.log_interval != 0:
            return
        
        with torch.no_grad():
            # Convert sigma to percentile
            log_sigma = torch.log(sigma)
            sigma_pct = 0.5 * (1 + torch.erf(
                (log_sigma - p_mean) / (p_std * np.sqrt(2))
            ))
            
            # Calculate entropy (diversity metric)
            def calc_entropy(probs):
                # Average across batch
                avg_probs = probs.mean(dim=0)
                avg_probs = avg_probs / (avg_probs.sum() + 1e-10)
                entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
                return entropy.item()
            
            # Calculate Gini coefficient (collapse metric)
            def calc_gini(probs):
                # Average usage per expert
                usage = probs.mean(dim=0).cpu().numpy()
                usage = np.sort(usage)
                n = len(usage)
                cumsum = np.cumsum(usage)
                return (2 * np.sum((np.arange(1, n+1) * usage))) / (n * cumsum[-1]) - (n+1)/n
            
            # Expert usage statistics
            unet_usage = unet_probs.mean(dim=0)  # (num_experts,)
            vit_usage = vit_probs.mean(dim=0)
            
            record = {
                "step": step,
                "avg_sigma_percentile": sigma_pct.mean().item(),
                
                # UNet router stats
                "unet_entropy": calc_entropy(unet_probs),
                "unet_gini": calc_gini(unet_probs),
                "unet_max_usage": unet_usage.max().item(),
                "unet_min_usage": unet_usage.min().item(),
                "unet_dead_experts": int((unet_usage < 0.01).sum().item()),
                "unet_usage_std": unet_usage.std().item(),
                
                # ViT router stats
                "vit_entropy": calc_entropy(vit_probs),
                "vit_gini": calc_gini(vit_probs),
                "vit_max_usage": vit_usage.max().item(),
                "vit_min_usage": vit_usage.min().item(),
                "vit_dead_experts": int((vit_usage < 0.01).sum().item()),
                "vit_usage_std": vit_usage.std().item(),
                
                # Per-expert usage (for detailed analysis)
                "unet_expert_usage": unet_usage.cpu().tolist(),
                "vit_expert_usage": vit_usage.cpu().tolist(),
            }
            
            self._write_jsonl(self.router_log_file, record)
    
    def log_scaling_gating(self,
                           scaling_factors: torch.Tensor,
                           gate_weights: torch.Tensor,
                           sigma: torch.Tensor):
        """
        Log scaling network and gating behavior.
        
        Args:
            p_std:
            p_mean:
            scaling_factors: Scaling network output (B, 2) [vit, unet]
            gate_weights: Final gate weights (B, 2) [wx, wa]
            sigma: Noise levels (B,)
        """
        self.accumulators['scaling_vit'].append(
            scaling_factors[:, 0].mean().item()
        )
        self.accumulators['scaling_unet'].append(
            scaling_factors[:, 1].mean().item()
        )
        self.accumulators['gate_wx'].append(
            gate_weights[:, 0].mean().item()
        )
        self.accumulators['gate_wa'].append(
            gate_weights[:, 1].mean().item()
        )
        self.accumulators['noise_level_min'].append(sigma.min().item())
        self.accumulators['noise_level_max'].append(sigma.max().item())
        self.accumulators['noise_level_std'].append(sigma.std().item())
        self.accumulators['noise_level'].append(
            sigma.mean().item()
        )
    
    def log_gradients(self,
                     step: int,
                     model,
                     component_names: Optional[List[str]] = None):
        """
        Log gradient norms for different model components.
        
        Args:
            step: Current training step
            model: The model
            component_names: List of component names to track
                           Default: ['Unet_experts', 'VIT_experts', 'Unet_router', 
                                    'vit_router', 'scaling_net', 'cross_attn']
        """
        if step % self.log_interval != 0:
            return
        
        if component_names is None:
            component_names = ['Unet_experts', 'VIT_experts', 'Unet_router',
                             'vit_router', 'scaling_net', 'cross_attn']
        
        record = {"step": step}
        
        with torch.no_grad():
            for name in component_names:
                if hasattr(model, name):
                    component = getattr(model, name)
                    grad_norm = self._compute_grad_norm(component.parameters())
                    record[f"{name}_grad_norm"] = grad_norm
        
        self._write_jsonl(self.gradient_log_file, record)

    def log_weight_statistics(self, step: int, model: torch.nn.Module):
        """
        Logs weight statistics (Mean, Std, Min, Max).
        FIXED: Removed torch.quantile and torch.cat to prevent RuntimeError on large models.
        """
        # Log much less frequently (e.g., every 50th log interval)
        if step % (self.log_interval * 50) != 0:
            return

        component_names = ['Unet_experts', 'VIT_experts']
        record = {"step": step}

        with torch.no_grad():
            for name in component_names:
                if hasattr(model, name):
                    component = getattr(model, name)

                    # Accumulators for iterative calculation
                    total_sum = 0.0
                    total_sq_sum = 0.0  # Sum of squares for Std Dev
                    total_count = 0
                    max_val = -float('inf')
                    min_val = float('inf')

                    found_weights = False

                    # Iterate over parameters one by one (Memory Safe)
                    for param in component.parameters():
                        if param.requires_grad and param.ndim > 1:
                            found_weights = True

                            # Get min/max of this specific tensor
                            p_min = param.min().item()
                            p_max = param.max().item()

                            # Update global min/max
                            if p_max > max_val: max_val = p_max
                            if p_min < min_val: min_val = p_min

                            # Update sums for Mean/Std
                            total_sum += param.sum().item()
                            total_sq_sum += param.pow(2).sum().item()
                            total_count += param.numel()

                    if found_weights and total_count > 0:
                        # Calculate statistics from accumulators
                        mean = total_sum / total_count

                        # Variance = E[X^2] - (E[X])^2
                        var = (total_sq_sum / total_count) - (mean ** 2)
                        std = np.sqrt(max(0, var))  # Clamp to 0 to avoid fp errors

                        record[f"{name}_weight_mean"] = round(mean, 6)
                        record[f"{name}_weight_std"] = round(std, 6)
                        record[f"{name}_weight_max"] = round(max_val, 6)
                        record[f"{name}_weight_min"] = round(min_val, 6)
                    else:
                        record[f"{name}_weight_mean"] = None

        self._write_jsonl(self.weight_log_file, record)
    
    def _flush_training_log(self):
        """Write accumulated training metrics to file."""
        if len(self.accumulators['step']) == 0:
            return
        
        # Average over the interval
        record = {"step": int(self.accumulators['step'][-1])}
        
        for key, values in self.accumulators.items():
            if key == 'step':
                continue
            if len(values) > 0:
                record[key] = round(float(np.mean(values)), 6)
        
        self._write_jsonl(self.main_log_file, record)
        
        # Clear accumulators
        self.accumulators.clear()
    @staticmethod
    def _write_jsonl(filepath: Path, record: Dict[str, Any]):
        """Append JSON record to file."""
        with open(filepath, 'a') as f:
            f.write(json.dumps(record) + '\n')
    
    @staticmethod
    def _compute_grad_norm(parameters) -> float:
        """Compute gradient norm for a set of parameters."""
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
