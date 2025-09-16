"""
Visualize Session 3 Step 1 Loss Curves
======================================

Extract and plot Session 3 temporal dynamics loss curves, excluding steady_state phase.
Shows only temperature MAE and wind MAE with temporal stage boundaries.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def extract_loss_from_checkpoints():

    # Find Session 3 Step 1 checkpoint directory
    step1_dir = Path('../checkpoints/training_sessions/session_3/step_1')
    
    # Look for run directories first
    run_dirs = list(step1_dir.glob('run_*'))
    if run_dirs:
        latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
        checkpoint_dir = latest_run
        print(f"[DIR] Using latest run directory: {latest_run.name}")
    else:
        checkpoint_dir = step1_dir
        print(f"[DIR] Using main checkpoint directory")
    
    # Find all checkpoint files (Session 3 pattern, excluding steady_state)
    checkpoint_files = []
    
    # Look for temporal phase checkpoints (exclude steady_state)
    for phase in ['mixed_temporal', 'late_transient', 'early_transient', 'cold_start']:
        phase_files = list(checkpoint_dir.glob(f'session3_{phase}_epoch_*.pt'))
        checkpoint_files.extend(phase_files)
    
    if not checkpoint_files:
        print("[ERROR] No temporal checkpoint files found (steady_state excluded)")
        return [], [], []
    
    print(f"[DATA] Found {len(checkpoint_files)} temporal checkpoint files (steady_state excluded)")
    
    # Extract epoch and MAE data
    epochs = []
    temp_maes = []
    wind_maes = []
    phases = []
    
    for checkpoint_file in sorted(checkpoint_files):
        try:
            # Extract phase and epoch from filename
            filename_parts = checkpoint_file.stem.split('_')
            phase_name = '_'.join(filename_parts[1:-2])  # Extract phase name
            epoch_num = int(filename_parts[-1])-300
            #epoch num = epoch num-300
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            
            # Get MAE values
            temp_mae = checkpoint.get('temp_mae', 0.0)
            wind_mae = checkpoint.get('wind_mae', 0.0)
            
            epochs.append(epoch_num)
            temp_maes.append(temp_mae)
            wind_maes.append(wind_mae)
            phases.append(phase_name)
            
        except Exception as e:
            print(f"[WARNING]  Error loading {checkpoint_file.name}: {e}")
            continue
    
    return epochs, temp_maes, wind_maes, phases

def plot_loss_curves(epochs, temp_maes, wind_maes, phases):

    if not epochs:
        print("[ERROR] No data to plot")
        return
    
    # Create figure with only two subplots (temp MAE and wind MAE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define Session 3 temporal phase boundaries (excluding steady_state)
    # Mixed_temporal: 500, Late_transient: 500, Early_transient: 500, Cold_start: 500
    stage_boundaries = [0, 300, 600, 900]  # Cumulative epochs
    stage_names = ['Late Transient', 'Early Transient', 'Cold Start','Mixed Temporal']
    
    # Temperature MAE plot (scatter points only, no connecting lines)
    ax1.scatter(epochs, temp_maes, c='red', s=1, alpha=0.7, label='Temperature MAE')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Temperature MAE (K)')
    ax1.set_title('Session 3 Step 1: Temperature MAE Progress (Temporal Dynamics, Steady State Excluded)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add statistics
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
    
    # Wind MAE plot (scatter points only, no connecting lines)
    ax2.scatter(epochs, wind_maes, c='blue', s=1, alpha=0.7, label='Wind MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Wind MAE (m/s)')
    ax2.set_title('Session 3 Step 1: Wind MAE Progress (Temporal Dynamics, Steady State Excluded)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add statistics
    if wind_maes:
        min_wind_mae = min(wind_maes)
        final_wind_mae = wind_maes[-1]
        ax2.axhline(y=min_wind_mae, color='b', linestyle='--', alpha=0.7)
        ax2.text(0.02, 0.98, f'Best: {min_wind_mae:.3f}m/s    Final: {final_wind_mae:.3f}m/s', 
                 transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add stage boundary markers
        for i, (boundary, stage_name) in enumerate(zip(stage_boundaries, stage_names)):
            ax2.axvline(x=boundary, color='blue', linestyle='--', alpha=0.6, linewidth=1)
            ax2.text(boundary, max(wind_maes) * (0.9 - i*0.1), stage_name, 
                    rotation=90, ha='right', va='top', fontsize=9, color='darkblue')
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path('../checkpoints/training_sessions/session_3/step_1/s3s1_loss_curves.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVE] Loss curves saved to: {save_path}")
    
    # Save data as JSON
    data_path = Path('../checkpoints/training_sessions/session_3/step_1/s3s1_loss_data.json')
    loss_data = {
        'epochs': epochs,
        'temperature_mae_kelvin': temp_maes,
        'wind_mae_ms': wind_maes,
        'phases': phases,
        'best_temp_mae': min(temp_maes) if temp_maes else None,
        'best_wind_mae': min(wind_maes) if wind_maes else None,
        'final_temp_mae': final_temp_mae if temp_maes else None,
        'final_wind_mae': final_wind_mae if wind_maes else None,
        'total_epochs': len(epochs),
        'note': 'steady_state phase excluded from analysis'
    }
    
    with open(data_path, 'w') as f:
        json.dump(loss_data, f, indent=2)
    print(f"[DATA] Loss data saved to: {data_path}")
    
    plt.show()

def main():

    print("[PLOT] Visualizing Session 3 Step 1 Loss Curves (Temporal Dynamics)")
    print("[WARNING]  Note: Steady state phase excluded from analysis")
    print("=" * 70)
    
    # Extract loss data from checkpoints
    epochs, temp_maes, wind_maes, phases = extract_loss_from_checkpoints()
    
    if epochs:
        print(f"[DATA] Extracted data from {len(epochs)} epochs")
        print(f"[DATA] Epoch range: {min(epochs)} - {max(epochs)}")
        print(f"[DATA] Phases included: {list(set(phases))}")
        if temp_maes:
            print(f"[DATA] Best Temperature MAE: {min(temp_maes):.2f}K")
        if wind_maes:
            print(f"[DATA] Best Wind MAE: {min(wind_maes):.3f}m/s")
        
        # Plot the curves
        plot_loss_curves(epochs, temp_maes, wind_maes, phases)
    else:
        print("[ERROR] No temporal loss data found to visualize")
        print("[EMOJI] Make sure Session 3 Step 1 temporal training has been completed")

if __name__ == "__main__":
    main()
