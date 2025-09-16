"""
Visualize Session 2 Step 1 Loss Curve (Sparsity Adaptation)
============================================================

Extract and plot the loss curve from the latest Session 2 Step 1 training run.
This session focuses on sparsity adaptation with curriculum learning.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def extract_loss_from_checkpoints():

    # Find Session 2 Step 1 checkpoint directory
    step1_dir = Path('../checkpoints/training_sessions/session_2/step_1')
    
    # Look for run directories first
    run_dirs = list(step1_dir.glob('run_*'))
    if run_dirs:
        latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
        checkpoint_dir = latest_run
        print(f"[DIR] Using latest run directory: {latest_run.name}")
    else:
        checkpoint_dir = step1_dir
        print(f"[DIR] Using main checkpoint directory")
    
    # Find all checkpoint files (Session 2 Step 1 pattern)
    checkpoint_files = list(checkpoint_dir.glob('s2s1_sparsity_adaptation_epoch_*.pt'))
    
    if not checkpoint_files:
        # Try alternative patterns
        checkpoint_files = list(checkpoint_dir.glob('session2_step1_epoch_*.pt'))
        if not checkpoint_files:
            checkpoint_files = list(checkpoint_dir.glob('*epoch_*.pt'))
    
    if not checkpoint_files:
        print("[ERROR] No checkpoint files found")
        print(f"[DIR] Searched in: {checkpoint_dir}")
        return [], [], []
    
    print(f"[DATA] Found {len(checkpoint_files)} checkpoint files")
    
    # Extract epoch and loss data
    epochs = []
    temp_maes = []
    wind_maes = []
    total_losses = []
    
    for checkpoint_file in sorted(checkpoint_files):
        try:
            # Extract epoch number from filename
            filename_parts = checkpoint_file.stem.split('_')
            epoch_num = None
            for i, part in enumerate(filename_parts):
                if part == 'epoch' and i + 1 < len(filename_parts):
                    epoch_num = int(filename_parts[i + 1])
                    break
            
            if epoch_num is None:
                continue
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            
            # Get loss values (Session 2 Step 1 specific metrics)
            temp_mae = checkpoint.get('temp_mae', checkpoint.get('temperature_mae', 0.0))
            wind_mae = checkpoint.get('wind_mae', checkpoint.get('wind_velocity_mae', 0.0))
            total_loss = checkpoint.get('total_loss', checkpoint.get('loss', 0.0))
            
            epochs.append(epoch_num)
            temp_maes.append(temp_mae)
            wind_maes.append(wind_mae)
            total_losses.append(total_loss)
            
        except Exception as e:
            print(f"[WARNING]  Error loading {checkpoint_file.name}: {e}")
            continue
    
    return epochs, temp_maes, wind_maes, total_losses

def plot_loss_curves(epochs, temp_maes, wind_maes, total_losses):

    if not epochs:
        print("[ERROR] No data to plot")
        return
    
    # Create figure with only top two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define Session 2 curriculum stage boundaries
    # Dense: 150, Medium: 200, Sparse: 250, Minimal: 300 epochs
    stage_boundaries = [0,150, 350, 600]  # Cumulative epochs
    stage_names = ['Dense', 'Medium', 'Sparse', 'Minimal']
    
    # Temperature MAE plot
    ax1.scatter(epochs, temp_maes, c='red', s=2, alpha=0.7, label='Temperature MAE')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Temperature MAE (K)')
    ax1.set_title('Session 2 Step 1: Temperature MAE Progress\n(Sparsity Adaptation)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add temperature statistics
    if temp_maes:
        min_temp_mae = min(temp_maes)
        final_temp_mae = temp_maes[-1]
        ax1.axhline(y=min_temp_mae, color='r', linestyle='--', alpha=0.7)
        ax1.text(0.02, 0.98, f'Best: {min_temp_mae:.2f}K    Final: {final_temp_mae:.2f}K', 
                 transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add stage boundary markers
        for i, (boundary, stage_name) in enumerate(zip(stage_boundaries, stage_names)):
            ax1.axvline(x=boundary, color='red', linestyle='--', alpha=0.6, linewidth=1)
            ax1.text(boundary, max(temp_maes) * (0.9 - i*0.1), stage_name, 
                    rotation=90, ha='right', va='top', fontsize=9, color='darkred')
    
    # Wind MAE plot
    ax2.scatter(epochs, wind_maes, c='blue', s=2, alpha=0.7, label='Wind MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Wind MAE (m/s)')
    ax2.set_title('Session 2 Step 1: Wind MAE Progress\n(Sparsity Adaptation)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add wind statistics
    if wind_maes:
        min_wind_mae = min(wind_maes)
        final_wind_mae = wind_maes[-1]
        ax2.axhline(y=min_wind_mae, color='b', linestyle='--', alpha=0.7)
        ax2.text(0.02, 0.98, f'Best: {min_wind_mae:.4f}m/s    Final: {final_wind_mae:.4f}m/s', 
                 transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add stage boundary markers
        for i, (boundary, stage_name) in enumerate(zip(stage_boundaries, stage_names)):
            ax2.axvline(x=boundary, color='blue', linestyle='--', alpha=0.6, linewidth=1)
            ax2.text(boundary, max(wind_maes) * (0.9 - i*0.1), stage_name, 
                    rotation=90, ha='right', va='top', fontsize=9, color='darkblue')
    # Bottom plots removed - showing only temperature and wind MAE progression
    
    plt.tight_layout()
    
    # Save plot
    save_dir = Path('../checkpoints/training_sessions/session_2/step_1')
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 's2s1_loss_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Save data as JSON
    data = {
        'epochs': epochs,
        'temperature_mae': temp_maes,
        'wind_mae': wind_maes,
        'total_loss': total_losses,
        'session': 'Session 2 Step 1 - Sparsity Adaptation',
        'best_temp_mae': min(temp_maes) if temp_maes else None,
        'best_wind_mae': min(wind_maes) if wind_maes else None,
        'best_total_loss': min(total_losses) if total_losses else None
    }
    
    data_path = save_dir / 's2s1_loss_data.json'
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"[DATA] Plot saved to: {save_path}")
    print(f"[DATA] Loss data saved to: {data_path}")
    
    plt.show()

def main():

    print("[PLOT] Visualizing Session 2 Step 1 Loss Curves (Sparsity Adaptation)")
    print("=" * 70)
    
    # Extract loss data from checkpoints
    epochs, temp_maes, wind_maes, total_losses = extract_loss_from_checkpoints()
    
    if epochs:
        print(f"[DATA] Extracted data from {len(epochs)} epochs")
        print(f"[DATA] Epoch range: {min(epochs)} - {max(epochs)}")
        if temp_maes:
            print(f"[DATA] Best Temperature MAE: {min(temp_maes):.2f}K")
        if wind_maes:
            print(f"[DATA] Best Wind MAE: {min(wind_maes):.4f}m/s")
        if total_losses:
            print(f"[DATA] Best Total Loss: {min(total_losses):.4f}")
        
        # Plot the curves
        plot_loss_curves(epochs, temp_maes, wind_maes, total_losses)
    else:
        print("[ERROR] No loss data found to visualize")
        print("[EMOJI] Make sure Session 2 Step 1 training has been completed")

if __name__ == "__main__":
    main()

