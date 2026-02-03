import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

class Plotter:
    """
    Research-grade visualization toolkit for hybrid diffusion MoE models.
    Based on best practices from EDM2, MoE literature, and CVPR/NeurIPS papers.
    """
    
    def __init__(self, run_name: str = "experiment", output_dir: str = "./analysis_results"):
        self.run_name = run_name
        self.output_dir = os.path.join(output_dir, run_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set publication-quality plotting style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
        # Publication-quality settings
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
        })
        
        print(f"✓ Initialized Professional Plotter for: {run_name}")
        print(f"✓ Output directory: {self.output_dir}")

    # =========================================================
    # 1. COMPREHENSIVE TRAINING DYNAMICS (MULTI-PANEL)
    # =========================================================
    def plot_comprehensive_training_dynamics(self, log_file_path: str, 
                                             smooth_window: int = 50):
        """
        Creates a comprehensive 9-panel training dynamics visualization.
        Based on EDM2 and modern diffusion training best practices.
        Now includes MoE auxiliary losses and learning rate schedule.
        """
        df = self._parse_training_logs(log_file_path)
        
        if df is None or len(df) == 0:
            print("❌ No valid training data found")
            return
        
        # Smooth curves for better visualization
        df_smooth = df.copy()
        for col in ['Loss', 'MSE', 'LogVar', 'EDMLoss', 'BalanceLoss', 'ZLoss', 'EntropyLoss']:
            if col in df.columns and df[col].notna().any():
                df_smooth[col] = gaussian_filter1d(df[col].values, sigma=smooth_window//10)
        
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Panel 1: Total Loss vs Pure EDM Loss (comparison)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(df['Step'], df['Loss'], alpha=0.3, color='tab:blue', linewidth=0.5)
        ax1.plot(df_smooth['Step'], df_smooth['Loss'], 
                color='tab:blue', linewidth=2, label='Total Loss (with aux)')
        
        if 'EDMLoss' in df.columns and df['EDMLoss'].notna().any():
            ax1.plot(df['Step'], df['EDMLoss'], alpha=0.3, color='tab:green', linewidth=0.5)
            ax1.plot(df_smooth['Step'], df_smooth['EDMLoss'],
                    color='tab:green', linewidth=2, label='Pure EDM Loss', linestyle='--')
        
        # Add trend line
        z = np.polyfit(df['Step'], df['Loss'], 3)
        p = np.poly1d(z)
        ax1.plot(df['Step'], p(df['Step']), '--', 
                color='red', linewidth=2, alpha=0.7, label='Trend')
        
        ax1.set_xlabel('Training Step', fontweight='bold')
        ax1.set_ylabel('Loss Value', fontweight='bold')
        ax1.set_title('Training Loss Evolution (Total vs Pure EDM)', fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Loss Convergence Rate (Derivative)
        ax2 = fig.add_subplot(gs[0, 2])
        loss_derivative = np.gradient(df_smooth['Loss'])
        ax2.plot(df['Step'], loss_derivative, color='tab:orange', linewidth=1.5)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax2.set_xlabel('Training Step', fontweight='bold')
        ax2.set_ylabel('Loss Gradient (∂L/∂step)', fontweight='bold')
        ax2.set_title('Loss Convergence Rate', fontweight='bold')
        ax2.fill_between(df['Step'], 0, loss_derivative, 
                         where=(loss_derivative < 0), 
                         color='green', alpha=0.2, label='Improving')
        ax2.fill_between(df['Step'], 0, loss_derivative,
                         where=(loss_derivative > 0),
                         color='red', alpha=0.2, label='Degrading')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: MSE (Reconstruction Quality) - Log Scale
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.semilogy(df['Step'], df['MSE'], alpha=0.3, 
                    color='tab:green', linewidth=0.5)
        ax3.semilogy(df_smooth['Step'], df_smooth['MSE'],
                    color='tab:green', linewidth=2, label='MSE')
        ax3.set_xlabel('Training Step', fontweight='bold')
        ax3.set_ylabel('Mean Squared Error (log)', fontweight='bold')
        ax3.set_title('Reconstruction Quality (MSE)', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, which='both')
        
        # Panel 4: Uncertainty (LogVar) Calibration
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(df['Step'], df['LogVar'], alpha=0.3,
                color='tab:red', linewidth=0.5)
        ax4.plot(df_smooth['Step'], df_smooth['LogVar'],
                color='tab:red', linewidth=2, label='Log Variance')
        
        # Highlight different phases
        if len(df) > 3:
            third = len(df) // 3
            ax4.axvspan(df['Step'].iloc[0], df['Step'].iloc[third], 
                       alpha=0.1, color='yellow', label='Exploration')
            ax4.axvspan(df['Step'].iloc[third], df['Step'].iloc[2*third],
                       alpha=0.1, color='blue', label='Learning')
            ax4.axvspan(df['Step'].iloc[2*third], df['Step'].iloc[-1],
                       alpha=0.1, color='green', label='Refinement')
        
        ax4.set_xlabel('Training Step', fontweight='bold')
        ax4.set_ylabel('Log Variance (Uncertainty)', fontweight='bold')
        ax4.set_title('Model Confidence Evolution', fontweight='bold')
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Zeta Schedule (Router Exploration)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(df['Step'], df['Zeta'], color='tab:purple', 
                linewidth=2, label='Zeta')
        ax5.fill_between(df['Step'], 0, df['Zeta'],
                        alpha=0.3, color='tab:purple')
        ax5.set_xlabel('Training Step', fontweight='bold')
        ax5.set_ylabel('Exploration Noise (ζ)', fontweight='bold')
        ax5.set_title('Router Exploration Schedule', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: MoE Auxiliary Losses (Stacked)
        ax6 = fig.add_subplot(gs[2, 0])
        
        if all(col in df.columns for col in ['BalanceLoss', 'ZLoss', 'EntropyLoss']):
            ax6.plot(df['Step'], df_smooth['BalanceLoss'], 
                    label='Load Balance', linewidth=2, color='tab:cyan')
            ax6.plot(df['Step'], df_smooth['ZLoss'],
                    label='Z-Loss (Router)', linewidth=2, color='tab:orange')
            ax6.plot(df['Step'], df_smooth['EntropyLoss'],
                    label='Entropy Loss', linewidth=2, color='tab:pink')
            
            ax6.set_xlabel('Training Step', fontweight='bold')
            ax6.set_ylabel('Auxiliary Loss Value', fontweight='bold')
            ax6.set_title('MoE Auxiliary Losses', fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_yscale('log')
        else:
            ax6.text(0.5, 0.5, 'Auxiliary losses not logged', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.axis('off')
        
        # Panel 7: Learning Rate Schedule
        ax7 = fig.add_subplot(gs[2, 1])
        if 'LR' in df.columns and df['LR'].notna().any():
            ax7.plot(df['Step'], df['LR'], color='tab:brown', linewidth=2)
            ax7.fill_between(df['Step'], 0, df['LR'], alpha=0.3, color='tab:brown')
            ax7.set_xlabel('Training Step', fontweight='bold')
            ax7.set_ylabel('Learning Rate', fontweight='bold')
            ax7.set_title('Learning Rate Schedule', fontweight='bold')
            ax7.grid(True, alpha=0.3)
            ax7.set_yscale('log')
        else:
            ax7.text(0.5, 0.5, 'Learning rate not logged',
                    ha='center', va='center', transform=ax7.transAxes)
            ax7.axis('off')
        
        # Panel 8: Loss Decomposition (Pie Chart at Final Step)
        ax8 = fig.add_subplot(gs[2, 2])
        
        if all(col in df.columns for col in ['EDMLoss', 'BalanceLoss', 'ZLoss', 'EntropyLoss']):
            final_step = df.iloc[-1]
            
            # Calculate auxiliary loss sum
            aux_total = (final_step['BalanceLoss'] + 
                        final_step['ZLoss'] + 
                        final_step['EntropyLoss'])
            
            # Avoid division by zero
            if final_step['Loss'] > 0:
                edm_fraction = final_step['EDMLoss'] / final_step['Loss'] * 100
                aux_fraction = aux_total / final_step['Loss'] * 100
                
                sizes = [edm_fraction, aux_fraction]
                labels = [f'EDM Loss\n({edm_fraction:.1f}%)', 
                         f'Aux Losses\n({aux_fraction:.1f}%)']
                colors = ['#4CAF50', '#FF9800']
                
                ax8.pie(sizes, labels=labels, colors=colors, autopct='',
                       startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
                ax8.set_title('Final Loss Composition', fontweight='bold')
            else:
                ax8.text(0.5, 0.5, 'Loss is zero',
                        ha='center', va='center', transform=ax8.transAxes)
        else:
            ax8.text(0.5, 0.5, 'Loss decomposition unavailable',
                    ha='center', va='center', transform=ax8.transAxes)
            ax8.axis('off')
        
        # Panel 9: Statistical Summary Table
        ax9 = fig.add_subplot(gs[3, :])
        ax9.axis('off')
        
        # Calculate statistics
        stats_data = [
            ['Metric', 'Initial', 'Final', 'Min', 'Max', 'Improvement', 'Std Dev'],
            ['Total Loss', f"{df['Loss'].iloc[0]:.4f}", f"{df['Loss'].iloc[-1]:.4f}",
             f"{df['Loss'].min():.4f}", f"{df['Loss'].max():.4f}",
             f"{((df['Loss'].iloc[0] - df['Loss'].iloc[-1])/df['Loss'].iloc[0]*100):.1f}%",
             f"{df['Loss'].std():.4f}"],
            ['MSE', f"{df['MSE'].iloc[0]:.4f}", f"{df['MSE'].iloc[-1]:.4f}",
             f"{df['MSE'].min():.4f}", f"{df['MSE'].max():.4f}",
             f"{((df['MSE'].iloc[0] - df['MSE'].iloc[-1])/df['MSE'].iloc[0]*100):.1f}%",
             f"{df['MSE'].std():.4f}"],
        ]
        
        # Add EDM loss row if available
        if 'EDMLoss' in df.columns and df['EDMLoss'].notna().any():
            stats_data.append([
                'EDM Loss', f"{df['EDMLoss'].iloc[0]:.4f}", f"{df['EDMLoss'].iloc[-1]:.4f}",
                f"{df['EDMLoss'].min():.4f}", f"{df['EDMLoss'].max():.4f}",
                f"{((df['EDMLoss'].iloc[0] - df['EDMLoss'].iloc[-1])/df['EDMLoss'].iloc[0]*100):.1f}%",
                f"{df['EDMLoss'].std():.4f}"
            ])
        
        # Add LogVar
        stats_data.append([
            'LogVar', f"{df['LogVar'].iloc[0]:.4f}", f"{df['LogVar'].iloc[-1]:.4f}",
            f"{df['LogVar'].min():.4f}", f"{df['LogVar'].max():.4f}", 'N/A',
            f"{df['LogVar'].std():.4f}"
        ])
        
        table = ax9.table(cellText=stats_data, cellLoc='center',
                         loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(7):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(stats_data)):
            for j in range(7):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')
        
        fig.suptitle(f'Comprehensive Training Dynamics: {self.run_name}',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig(f"{self.output_dir}/01_comprehensive_training_dynamics.png")
        print(f"✓ Saved: 01_comprehensive_training_dynamics.png")
        plt.close()

    # =========================================================
    # 2. EXPERT SPECIALIZATION HEATMAP WITH ANALYSIS
    # =========================================================
    def plot_expert_specialization_advanced(self, 
                                            model,
                                            device='cuda',
                                            num_sigma_points=100,
                                            save_data=True):
        """
        Advanced expert specialization visualization with:
        - Heatmaps showing expert activation across noise levels
        - Load balancing metrics
        - Expert utilization statistics
        - Collapse detection indicators
        """
        print("Analyzing Expert Specialization...")
        model.eval()
        
        # Generate sigma range (log-spaced like EDM)
        sigmas = torch.logspace(np.log10(0.002), np.log10(80), 
                               num_sigma_points).to(device)
        
        # Convert sigma to percentiles for mask generation
        sigma_percentiles = 0.5 * (1 + torch.erf(
            (torch.log(sigmas) - (-0.4)) / (1.0 * np.sqrt(2))
        ))
        sigma_percentiles = sigma_percentiles.clamp(0, 1)
        
        # Dummy inputs
        dummy_x = torch.zeros(1, model.internal_channels, 32, 32).to(device)
        dummy_txt = torch.zeros(1, 1, 768).to(device)
        dummy_time = torch.zeros(1, 256).to(device)  # time_emb_dim
        
        # Storage
        unet_activations = []
        vit_activations = []
        
        num_experts = len(model.Unet_experts)
        
        with torch.no_grad():
            for sigma, sigma_pct in tqdm(zip(sigmas, sigma_percentiles), 
                                        total=len(sigmas),
                                        desc="Processing noise levels"):
                # Allow all experts
                mask = torch.ones(1, num_experts).to(device)
                
                # Get router outputs
                # Route through UNet path
                out_unet_router, unet_probs, _ = model.Unet_router(
                    x=dummy_x,
                    time_emb=dummy_time,
                    zeta=0.0,
                    mask=mask
                )
                
                # Route through ViT path
                out_vit_router, vit_probs, _ = model.vit_router(
                    x=dummy_x,
                    time_emb=dummy_time,
                    zeta=0.0,
                    mask=mask
                )
                
                unet_activations.append(unet_probs.cpu().numpy()[0])
                vit_activations.append(vit_probs.cpu().numpy()[0])
        
        # Stack: (num_experts, num_sigma_points)
        U_act = np.stack(unet_activations).T
        V_act = np.stack(vit_activations).T
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # ===== Panel 1: UNet Expert Heatmap =====
        ax1 = fig.add_subplot(gs[0, :2])
        im1 = ax1.imshow(U_act, aspect='auto', cmap='plasma',
                        interpolation='bilinear', vmin=0, vmax=1)
        
        # X-axis: noise levels
        tick_indices = np.linspace(0, num_sigma_points-1, 10, dtype=int)
        tick_labels = [f"{sigmas[i].item():.2f}" for i in tick_indices]
        ax1.set_xticks(tick_indices)
        ax1.set_xticklabels(tick_labels, rotation=45)
        ax1.set_xlabel('Noise Level (σ) [Low → High]', fontweight='bold')
        ax1.set_ylabel('UNet Expert ID', fontweight='bold')
        ax1.set_title('UNet Path: Expert Specialization Across Noise Levels',
                     fontweight='bold', pad=10)
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Routing Probability', rotation=270, labelpad=20)
        
        # Add grid for better readability
        ax1.set_yticks(np.arange(num_experts))
        ax1.grid(False)
        
        # ===== Panel 2: ViT Expert Heatmap =====
        ax2 = fig.add_subplot(gs[1, :2])
        im2 = ax2.imshow(V_act, aspect='auto', cmap='viridis',
                        interpolation='bilinear', vmin=0, vmax=1)
        
        ax2.set_xticks(tick_indices)
        ax2.set_xticklabels(tick_labels, rotation=45)
        ax2.set_xlabel('Noise Level (σ) [Low → High]', fontweight='bold')
        ax2.set_ylabel('ViT Expert ID', fontweight='bold')
        ax2.set_title('ViT Path: Expert Specialization Across Noise Levels',
                     fontweight='bold', pad=10)
        
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Routing Probability', rotation=270, labelpad=20)
        
        ax2.set_yticks(np.arange(num_experts))
        ax2.grid(False)
        
        # ===== Panel 3: Load Balancing Analysis (UNet) =====
        ax3 = fig.add_subplot(gs[0, 2])
        unet_usage = U_act.mean(axis=1)  # Average usage per expert
        colors = ['red' if u < 0.05 else 'orange' if u < 0.15 else 'green' 
                 for u in unet_usage]
        
        bars = ax3.barh(range(num_experts), unet_usage, color=colors, alpha=0.7)
        ax3.set_xlabel('Average Activation', fontweight='bold')
        ax3.set_ylabel('Expert ID')
        ax3.set_title('UNet Load Balance', fontweight='bold')
        ax3.axvline(x=1.0/num_experts, color='blue', linestyle='--',
                   linewidth=2, label=f'Ideal ({1/num_experts:.2f})')
        ax3.set_xlim(0, max(unet_usage) * 1.2)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, (bar, val) in enumerate(zip(bars, unet_usage)):
            ax3.text(val + 0.01, i, f'{val:.2f}', 
                    va='center', fontsize=8)
        
        # ===== Panel 4: Load Balancing Analysis (ViT) =====
        ax4 = fig.add_subplot(gs[1, 2])
        vit_usage = V_act.mean(axis=1)
        colors = ['red' if u < 0.05 else 'orange' if u < 0.15 else 'green'
                 for u in vit_usage]
        
        bars = ax4.barh(range(num_experts), vit_usage, color=colors, alpha=0.7)
        ax4.set_xlabel('Average Activation', fontweight='bold')
        ax4.set_ylabel('Expert ID')
        ax4.set_title('ViT Load Balance', fontweight='bold')
        ax4.axvline(x=1.0/num_experts, color='blue', linestyle='--',
                   linewidth=2, label=f'Ideal ({1/num_experts:.2f})')
        ax4.set_xlim(0, max(vit_usage) * 1.2)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, val) in enumerate(zip(bars, vit_usage)):
            ax4.text(val + 0.01, i, f'{val:.2f}',
                    va='center', fontsize=8)
        
        # ===== Panel 5: Expert Diversity Metrics =====
        ax5 = fig.add_subplot(gs[2, 0])
        
        # Calculate entropy (high = good diversity, low = collapse)
        def calculate_entropy(activations):
            # Normalize each column (timestep) to get distribution
            normalized = activations / (activations.sum(axis=0, keepdims=True) + 1e-10)
            entropy = -np.sum(normalized * np.log(normalized + 1e-10), axis=0)
            return entropy
        
        unet_entropy = calculate_entropy(U_act)
        vit_entropy = calculate_entropy(V_act)
        
        ax5.plot(sigma_percentiles.cpu().numpy(), unet_entropy, 
                label='UNet', linewidth=2, color='tab:orange')
        ax5.plot(sigma_percentiles.cpu().numpy(), vit_entropy,
                label='ViT', linewidth=2, color='tab:blue')
        ax5.axhline(y=np.log(num_experts), color='green', linestyle='--',
                   label=f'Maximum ({np.log(num_experts):.2f})', linewidth=2)
        ax5.set_xlabel('Noise Percentile', fontweight='bold')
        ax5.set_ylabel('Routing Entropy (bits)', fontweight='bold')
        ax5.set_title('Expert Diversity (Higher = Better)', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.fill_between(sigma_percentiles.cpu().numpy(), 0, unet_entropy,
                        alpha=0.2, color='tab:orange')
        ax5.fill_between(sigma_percentiles.cpu().numpy(), 0, vit_entropy,
                        alpha=0.2, color='tab:blue')
        
        # ===== Panel 6: Collapse Detection =====
        ax6 = fig.add_subplot(gs[2, 1])
        
        # Calculate Gini coefficient (0 = perfect balance, 1 = total collapse)
        def gini_coefficient(x):
            x = np.sort(x)
            n = len(x)
            cumsum = np.cumsum(x)
            return (2 * np.sum((np.arange(1, n+1) * x))) / (n * cumsum[-1]) - (n+1)/n
        
        unet_gini = gini_coefficient(unet_usage)
        vit_gini = gini_coefficient(vit_usage)
        
        metrics = ['UNet', 'ViT']
        gini_values = [unet_gini, vit_gini]
        colors_gini = ['red' if g > 0.4 else 'orange' if g > 0.2 else 'green' 
                      for g in gini_values]
        
        bars = ax6.bar(metrics, gini_values, color=colors_gini, alpha=0.7)
        ax6.axhline(y=0.3, color='orange', linestyle='--', 
                   label='Warning (0.3)', linewidth=2)
        ax6.axhline(y=0.5, color='red', linestyle='--',
                   label='Critical (0.5)', linewidth=2)
        ax6.set_ylabel('Gini Coefficient', fontweight='bold')
        ax6.set_title('Collapse Detection\n(Lower = Better)', fontweight='bold')
        ax6.set_ylim(0, 1)
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, gini_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # ===== Panel 7: Statistics Table =====
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        stats_data = [
            ['Metric', 'UNet', 'ViT', 'Status'],
            ['Avg Entropy', f'{unet_entropy.mean():.2f}', f'{vit_entropy.mean():.2f}',
             '✓' if min(unet_entropy.mean(), vit_entropy.mean()) > np.log(num_experts)*0.7 else '⚠'],
            ['Gini Coef.', f'{unet_gini:.3f}', f'{vit_gini:.3f}',
             '✓' if max(unet_gini, vit_gini) < 0.3 else '⚠'],
            ['Dead Experts', f'{np.sum(unet_usage < 0.01)}', f'{np.sum(vit_usage < 0.01)}',
             '✓' if (np.sum(unet_usage < 0.01) + np.sum(vit_usage < 0.01)) == 0 else '⚠'],
            ['Max Usage', f'{unet_usage.max():.2f}', f'{vit_usage.max():.2f}',
             '✓' if max(unet_usage.max(), vit_usage.max()) < 0.5 else '⚠']
        ]
        
        table = ax7.table(cellText=stats_data, cellLoc='center',
                         loc='center', bbox=(0, 0, 1, 1))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        for i in range(4):
            table[(0, i)].set_facecolor('#2E75B6')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        fig.suptitle(f'Expert Specialization Analysis: {self.run_name}',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig(f"{self.output_dir}/02_expert_specialization_advanced.png")
        print(f"✓ Saved: 02_expert_specialization_advanced.png")
        
        # Save raw data if requested
        if save_data:
            np.savez(f"{self.output_dir}/expert_activation_data.npz",
                    unet_activations=U_act,
                    vit_activations=V_act,
                    sigmas=sigmas.cpu().numpy(),
                    sigma_percentiles=sigma_percentiles.cpu().numpy())
            print(f"✓ Saved: expert_activation_data.npz")
        
        plt.close()
        
        # Print warnings if needed
        if unet_gini > 0.4 or vit_gini > 0.4:
            print("⚠️  WARNING: High Gini coefficient detected - potential router collapse!")
        if np.sum(unet_usage < 0.01) > 0 or np.sum(vit_usage < 0.01) > 0:
            print(f"⚠️  WARNING: Dead experts detected - UNet: {np.sum(unet_usage < 0.01)}, ViT: {np.sum(vit_usage < 0.01)}")

    # =========================================================
    # 3. SCALING FACTORS & GATING ANALYSIS
    # =========================================================
    def plot_scaling_and_gating_analysis(self, 
                                         scaling_history: Dict[str, List],
                                         gating_history: Dict[str, List]):
        """
        Analyzes the behavior of:
        - Scaling network (VIT vs UNet contribution)
        - Final gating network (Wx vs Wa)
        Across noise levels and training steps.
        """
        print("Creating scaling and gating analysis...")
        
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Convert to arrays
        noise_levels = np.array(scaling_history['noise_levels'])
        scaling_vit = np.array(scaling_history['scaling_vit'])
        scaling_unet = np.array(scaling_history['scaling_unet'])
        gate_wx = np.array(gating_history['gate_wx'])
        gate_wa = np.array(gating_history['gate_wa'])
        steps = np.array(scaling_history['steps'])
        
        # ===== Panel 1: Scaling Factors vs Noise (Scatter) =====
        ax1 = fig.add_subplot(gs[0, 0])
        
        scatter1 = ax1.scatter(noise_levels, scaling_vit, c=steps, 
                              cmap='viridis', alpha=0.6, s=10, label='ViT')
        scatter2 = ax1.scatter(noise_levels, scaling_unet, c=steps,
                              cmap='plasma', alpha=0.6, s=10, label='UNet', marker='x')
        
        ax1.set_xlabel('Noise Percentile (0=low, 1=high)', fontweight='bold')
        ax1.set_ylabel('Scaling Factor', fontweight='bold')
        ax1.set_title('Path Scaling vs Noise Level', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter1, ax=ax1)
        cbar.set_label('Training Step', rotation=270, labelpad=15)
        
        # ===== Panel 2: Scaling Factors Evolution (Binned) =====
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Bin noise levels
        num_bins = 20
        noise_bins = np.linspace(0, 1, num_bins + 1)
        bin_centers = (noise_bins[:-1] + noise_bins[1:]) / 2
        
        vit_binned = []
        unet_binned = []
        
        for i in range(num_bins):
            mask = (noise_levels >= noise_bins[i]) & (noise_levels < noise_bins[i+1])
            if mask.any():
                vit_binned.append(scaling_vit[mask].mean())
                unet_binned.append(scaling_unet[mask].mean())
            else:
                vit_binned.append(0)
                unet_binned.append(0)
        
        ax2.plot(bin_centers, vit_binned, 'o-', linewidth=2, 
                label='ViT', color='tab:blue', markersize=6)
        ax2.plot(bin_centers, unet_binned, 's-', linewidth=2,
                label='UNet', color='tab:orange', markersize=6)
        ax2.fill_between(bin_centers, vit_binned, alpha=0.3, color='tab:blue')
        ax2.fill_between(bin_centers, unet_binned, alpha=0.3, color='tab:orange')
        
        ax2.set_xlabel('Noise Percentile', fontweight='bold')
        ax2.set_ylabel('Average Scaling Factor', fontweight='bold')
        ax2.set_title('Expected Scaling Behavior', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        # Add ideal behavior reference
        ax2.axvline(x=0.5, color='gray', linestyle='--', 
                   alpha=0.5, label='Mid-point')
        
        # ===== Panel 3: Scaling Dominance Heatmap =====
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Create 2D histogram
        dominance = scaling_vit - scaling_unet  # >0 means ViT dominant
        
        hist, xedges, yedges = np.histogram2d(noise_levels, dominance,
                                              bins=[20, 20])
        
        im = ax3.imshow(hist.T, origin='lower', aspect='auto',
                       cmap='RdBu_r', extent=(0, 1, -1, 1),
                       interpolation='bilinear')
        
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax3.set_xlabel('Noise Percentile', fontweight='bold')
        ax3.set_ylabel('ViT - UNet Dominance', fontweight='bold')
        ax3.set_title('Path Dominance Heatmap', fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Sample Density', rotation=270, labelpad=15)
        
        # Add annotations
        ax3.text(0.7, 0.5, 'ViT Dominant', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
        ax3.text(0.3, -0.5, 'UNet Dominant', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        
        # ===== Panel 4: Gating Weights vs Noise =====
        ax4 = fig.add_subplot(gs[1, 0])
        
        ax4.scatter(noise_levels, gate_wx, c=steps, cmap='viridis',
                   alpha=0.6, s=10, label='Wx (Raw UNet)')
        ax4.scatter(noise_levels, gate_wa, c=steps, cmap='plasma',
                   alpha=0.6, s=10, label='Wa (Attended)', marker='x')
        
        ax4.set_xlabel('Noise Percentile', fontweight='bold')
        ax4.set_ylabel('Gate Weight', fontweight='bold')
        ax4.set_title('Final Gating Behavior', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        # ===== Panel 5: Gating vs Scaling Consistency =====
        ax5 = fig.add_subplot(gs[1, 1])
        
        # Check if gating follows scaling logic
        # When ViT is strong (high noise), Wa should be high
        # When UNet is strong (low noise), Wx should be high
        
        consistency_vit = []
        consistency_unet = []
        
        for i in range(len(noise_levels)):
            # High noise: expect ViT strong and Wa high
            if noise_levels[i] > 0.5:
                consistency_vit.append(scaling_vit[i] * gate_wa[i])
            # Low noise: expect UNet strong and Wx high  
            else:
                consistency_unet.append(scaling_unet[i] * gate_wx[i])
        
        ax5.hist(consistency_vit, bins=30, alpha=0.7, label='High Noise Consistency',
                color='tab:blue', edgecolor='black')
        ax5.hist(consistency_unet, bins=30, alpha=0.7, label='Low Noise Consistency',
                color='tab:orange', edgecolor='black')
        
        ax5.set_xlabel('Consistency Score', fontweight='bold')
        ax5.set_ylabel('Frequency', fontweight='bold')
        ax5.set_title('Scaling-Gating Consistency', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # ===== Panel 6: Summary Statistics =====
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        # Calculate key metrics
        vit_high_noise = scaling_vit[noise_levels > 0.7].mean()
        vit_low_noise = scaling_vit[noise_levels < 0.3].mean()
        unet_high_noise = scaling_unet[noise_levels > 0.7].mean()
        unet_low_noise = scaling_unet[noise_levels < 0.3].mean()
        
        wx_low_noise = gate_wx[noise_levels < 0.3].mean()
        wa_high_noise = gate_wa[noise_levels > 0.7].mean()
        
        stats_data = [
            ['Metric', 'Value', 'Expected', 'Status'],
            ['ViT @ High Noise', f'{vit_high_noise:.3f}', '> 0.6', 
             '✓' if vit_high_noise > 0.6 else '⚠'],
            ['UNet @ Low Noise', f'{unet_low_noise:.3f}', '> 0.6',
             '✓' if unet_low_noise > 0.6 else '⚠'],
            ['Wa @ High Noise', f'{wa_high_noise:.3f}', '> 0.5',
             '✓' if wa_high_noise > 0.5 else '⚠'],
            ['Wx @ Low Noise', f'{wx_low_noise:.3f}', '> 0.5',
             '✓' if wx_low_noise > 0.5 else '⚠'],
            ['Scaling Separation', 
             f'{abs(vit_high_noise - unet_high_noise):.3f}', '> 0.3',
             '✓' if abs(vit_high_noise - unet_high_noise) > 0.3 else '⚠']
        ]
        
        table = ax6.table(cellText=stats_data, cellLoc='center',
                         loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        for i in range(4):
            table[(0, i)].set_facecolor('#2E75B6')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        fig.suptitle(f'Scaling & Gating Analysis: {self.run_name}',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig(f"{self.output_dir}/03_scaling_gating_analysis.png")
        print(f"✓ Saved: 03_scaling_gating_analysis.png")
        plt.close()

    # =========================================================
    # 4. GRADIENT FLOW & TRAINING HEALTH
    # =========================================================
    def plot_gradient_flow(self, gradient_history: Dict[str, List]):
        """
        Monitors gradient health across different components.
        Critical for detecting vanishing/exploding gradients and dead paths.
        """
        print("Creating gradient flow analysis...")
        
        steps = np.array(gradient_history['steps'])
        unet_grads = np.array(gradient_history['unet_grad_norms'])
        vit_grads = np.array(gradient_history['vit_grad_norms'])
        router_unet_grads = np.array(gradient_history.get('router_unet_grads', []))
        router_vit_grads = np.array(gradient_history.get('router_vit_grads', []))
        scaling_grads = np.array(gradient_history.get('scaling_grads', []))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Gradient Flow & Training Health: {self.run_name}',
                    fontsize=16, fontweight='bold')
        
        # Panel 1: Expert Gradient Norms (Log Scale)
        axes[0, 0].semilogy(steps, unet_grads, label='UNet Experts',
                           linewidth=2, color='tab:orange', alpha=0.8)
        axes[0, 0].semilogy(steps, vit_grads, label='ViT Experts',
                           linewidth=2, color='tab:blue', alpha=0.8)
        
        # Add warning zones
        axes[0, 0].axhspan(1e-5, 1e-3, alpha=0.2, color='red',
                          label='Vanishing Risk')
        axes[0, 0].axhspan(10, 1000, alpha=0.2, color='orange',
                          label='Exploding Risk')
        
        axes[0, 0].set_xlabel('Training Step', fontweight='bold')
        axes[0, 0].set_ylabel('Gradient Norm (log scale)', fontweight='bold')
        axes[0, 0].set_title('Expert Path Gradients', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, which='both')
        
        # Panel 2: Router Gradients
        if len(router_unet_grads) > 0:
            axes[0, 1].semilogy(steps, router_unet_grads, 
                               label='UNet Router', linewidth=2,
                               color='tab:orange', alpha=0.8)
            axes[0, 1].semilogy(steps, router_vit_grads,
                               label='ViT Router', linewidth=2,
                               color='tab:blue', alpha=0.8)
            
            axes[0, 1].set_xlabel('Training Step', fontweight='bold')
            axes[0, 1].set_ylabel('Gradient Norm (log scale)', fontweight='bold')
            axes[0, 1].set_title('Router Gradients', fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3, which='both')
        
        # Panel 3: Gradient Ratio (Balance Check)
        axes[1, 0].plot(steps, unet_grads / (vit_grads + 1e-10),
                       linewidth=2, color='tab:purple')
        axes[1, 0].axhline(y=1.0, color='green', linestyle='--',
                          linewidth=2, label='Perfect Balance')
        axes[1, 0].axhspan(0.5, 2.0, alpha=0.2, color='green',
                          label='Healthy Range')
        
        axes[1, 0].set_xlabel('Training Step', fontweight='bold')
        axes[1, 0].set_ylabel('UNet/ViT Gradient Ratio', fontweight='bold')
        axes[1, 0].set_title('Path Balance (Closer to 1.0 = Better)', 
                            fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Panel 4: Health Score Summary
        axes[1, 1].axis('off')
        
        # Calculate health metrics
        vanishing_unet = np.sum(unet_grads < 1e-4) / len(unet_grads) * 100
        vanishing_vit = np.sum(vit_grads < 1e-4) / len(vit_grads) * 100
        exploding_unet = np.sum(unet_grads > 10) / len(unet_grads) * 100
        exploding_vit = np.sum(vit_grads > 10) / len(vit_grads) * 100
        
        ratio = unet_grads / (vit_grads + 1e-10)
        balance_score = np.sum((ratio > 0.5) & (ratio < 2.0)) / len(ratio) * 100
        
        health_data = [
            ['Health Metric', 'UNet', 'ViT', 'Status'],
            ['Vanishing (%)', f'{vanishing_unet:.1f}', f'{vanishing_vit:.1f}',
             '✓' if max(vanishing_unet, vanishing_vit) < 5 else '⚠'],
            ['Exploding (%)', f'{exploding_unet:.1f}', f'{exploding_vit:.1f}',
             '✓' if max(exploding_unet, exploding_vit) < 5 else '⚠'],
            ['Balance Score', '-', '-', f'{balance_score:.1f}%'],
            ['Overall Health', '-', '-',
             '✓ Healthy' if balance_score > 80 and 
             max(vanishing_unet, vanishing_vit) < 5 else '⚠ Needs Attention']
        ]
        
        table = axes[1, 1].table(cellText=health_data, cellLoc='center',
                                loc='center', bbox=[0, 0.2, 1, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(4):
            table[(0, i)].set_facecolor('#2E75B6')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/04_gradient_flow_health.png")
        print(f"✓ Saved: 04_gradient_flow_health.png")
        plt.close()

    # =========================================================
    # HELPER: Parse Training Logs
    # =========================================================
    @staticmethod
    def _parse_training_logs(log_file_path: str) -> Optional[pd.DataFrame]:
        """Parse training logs from JSON file (one JSON object per line)."""
        import json
        
        data = []
        try:
            with open(log_file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        # Normalize column names for compatibility
                        normalized = {
                            "Step": record.get("step"),
                            "Loss": record.get("total_loss"),
                            "MSE": record.get("raw_mse"),
                            "Zeta": record.get("zeta"),
                            "LogVar": record.get("log_var"),
                            "BalanceLoss": record.get("balance_loss", 0.0),
                            "ZLoss": record.get("z_loss", 0.0),
                            "EntropyLoss": record.get("entropy_loss", 0.0),
                            "EDMLoss": record.get("edm_loss", 0.0),
                            "LR": record.get("lr", 0.0)
                        }
                        data.append(normalized)
                    
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Skipping malformed JSON at line {line_num}: {e}")
                        continue
            
            if len(data) == 0:
                print("❌ No valid training data found in log file")
                return None
            
            df = pd.DataFrame(data)
            print(f"✓ Loaded {len(df)} training steps from {log_file_path}")
            return df
        
        except FileNotFoundError:
            print(f"❌ Log file not found: {log_file_path}")
            return None
        except Exception as e:
            print(f"❌ Error parsing logs: {e}")
            return None


# =========================================================
# USAGE EXAMPLE
# =========================================================
if __name__ == "__main__":
    plotter = Plotter(
        run_name="hdmoem_experiment_v1",
        output_dir="./analysis_results"
    )
    
    print("\n" + "="*60)
    print("Professional Diffusion Model Plotter - Ready!")
    print("="*60)
    print("\nAvailable visualization methods:")
    print("1. plot_comprehensive_training_dynamics(log_file_path)")
    print("2. plot_expert_specialization_advanced(model, device)")
    print("3. plot_scaling_and_gating_analysis(scaling_history, gating_history)")
    print("4. plot_gradient_flow(gradient_history)")
    print("\nSee documentation for data format requirements.")
