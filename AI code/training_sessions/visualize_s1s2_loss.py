"""
Visualize Session 1 Step 2 Loss Curve
=====================================

Extract and plot the loss curve from the latest Session 1 Step 2 training run.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import glob

def extract_loss_from_checkpoints():

    # Find Session 1 Step 2 checkpoint directory
    step2_dir = Path('../checkpoints/training_sessions/session_1/step_2')
    
    # Look for run directories first
    run_dirs = list(step2_dir.glob('run_*'))
    if run_dirs:
        latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
        checkpoint_dir = latest_run
        print(f"[DIR] Using latest run directory: {latest_run.name}")
    else:
        checkpoint_dir = step2_dir
        print(f"[DIR] Using main checkpoint directory")
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob('step2_training_BEST_epoch_*.pt'))
    
    if not checkpoint_files:
        print("[ERROR] No checkpoint files found")
        return None, None
    
    print(f"[DATA] Found {len(checkpoint_files)} checkpoint files")
    
    # Extract epoch and MAE data
    epochs = []
    temp_maes = []
    wind_maes = []
    
    for checkpoint_file in sorted(checkpoint_files):
        try:
            # Extract epoch number from filename
            epoch_num = int(checkpoint_file.stem.split('_')[-1])
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            
            # Get MAE values
            temp_mae = checkpoint.get('temp_mae', 0.0)
            wind_mae = checkpoint.get('wind_mae', 0.0)
            
            epochs.append(epoch_num)
            temp_maes.append(temp_mae)
            wind_maes.append(wind_mae)
            
        except Exception as e:
            print(f"[WARNING]  Error loading {checkpoint_file.name}: {e}")
            continue
    
    return epochs, temp_maes, wind_maes

def plot_loss_curves(epochs, temp_maes, wind_maes):

    if not epochs:
        print("[ERROR] No data to plot")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Temperature MAE plot (scatter points only, no connecting lines)
    ax1.scatter(epochs, temp_maes, c='red', s=1, alpha=0.7, label='Temperature MAE')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Temperature MAE (K)')
    ax1.set_title('Session 1 Step 2: Temperature MAE Progress')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add statistics
    min_temp_mae = min(temp_maes)
    final_temp_mae = temp_maes[-1]
    ax1.axhline(y=min_temp_mae, color='r', linestyle='--', alpha=0.7, label=f'Best: {min_temp_mae:.2f}K')
    ax1.text(0.02, 0.98, f'Best: {min_temp_mae:.2f}K  Final: {final_temp_mae:.2f}K', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Wind MAE plot (scatter points only, no connecting lines)
    ax2.scatter(epochs, wind_maes, c='blue', s=1, alpha=0.7, label='Wind MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Wind MAE (m/s)')
    ax2.set_title('Session 1 Step 2: Wind MAE Progress')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add statistics
    min_wind_mae = min(wind_maes)
    final_wind_mae = wind_maes[-1]
    ax2.axhline(y=min_wind_mae, color='b', linestyle='--', alpha=0.7, label=f'Best: {min_wind_mae:.3f}m/s')
    ax2.text(0.02, 0.98, f'Best: {min_wind_mae:.3f}m/s  Final: {final_wind_mae:.3f}m/s', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path('../checkpoints/training_sessions/session_1/step_2/s1s2_loss_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVE] Loss curves saved to: {save_path}")
    
    # Also save data as JSON
    data_path = Path('../checkpoints/training_sessions/session_1/step_2/s1s2_loss_data.json')
    loss_data = {
        'epochs': epochs,
        'temperature_mae_kelvin': temp_maes,
        'wind_mae_ms': wind_maes,
        'best_temp_mae': min_temp_mae,
        'best_wind_mae': min_wind_mae,
        'final_temp_mae': final_temp_mae,
        'final_wind_mae': final_wind_mae,
        'total_epochs': len(epochs)
    }
    
    with open(data_path, 'w') as f:
        json.dump(loss_data, f, indent=2)
    print(f"[DATA] Loss data saved to: {data_path}")
    
    plt.show()

def main():

    print("[PLOT] Visualizing Session 1 Step 2 Loss Curves")
    print("=" * 50)
    
    # Extract loss data from checkpoints
    epochs, temp_maes, wind_maes = extract_loss_from_checkpoints()
    
    if epochs:
        print(f"[DATA] Extracted data from {len(epochs)} epochs")
        print(f"[DATA] Epoch range: {min(epochs)} - {max(epochs)}")
        print(f"[DATA] Best Temperature MAE: {min(temp_maes):.2f}K")
        print(f"[DATA] Best Wind MAE: {min(wind_maes):.3f}m/s")
        
        # Plot the curves
        plot_loss_curves(epochs, temp_maes, wind_maes)
    else:
        print("[ERROR] No loss data found to visualize")

if __name__ == "__main__":
    main()

