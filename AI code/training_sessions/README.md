# Training Sessions Directory

## Contents

- **Session_1_Step_1_Data_Generation.py** - Foundation training data generation
- **Session_1_Step_2_Training.py** - Foundation model training
- **Session_1_Step_3_Evaluation.py** - Foundation model evaluation
- **Session_2_Step_1_Sparsity_Adaptation.py** - Sparsity adaptation training
- **Session_2_Step_2_Evaluation.py** - Sparsity performance evaluation
- **Session_3_Step_1_Temporal_Dynamics.py** - Temporal dynamics training
- **Session_3_Step_2_Evaluation.py** - Temporal performance evaluation
- **Session_4_Quick_Analysis.py** - Comprehensive analysis and visualization
- **Session_5_PathPlanning_Testing.py** - Path planning performance testing
- **visualize_s1s2_loss.py** - Session 1-2 loss visualization
- **visualize_s2s1_loss.py** - Session 2-1 loss visualization
- **visualize_s3s1_loss.py** - Session 3-1 loss visualization

## Runnable Scripts

- **Session_1_Step_1_Data_Generation.py** - Generates training data (30 min runtime)
  - Output: Training samples created, progress shown in terminal
- **Session_1_Step_2_Training.py** - Trains foundation model (4-6 hours)
  - Output: Model checkpoints saved to `../checkpoints/`, training logs in terminal
- **Session_1_Step_3_Evaluation.py** - Evaluates foundation model (1 hour)
  - Output: Evaluation metrics displayed in terminal, results in checkpoint directory
- **Session_2_Step_1_Sparsity_Adaptation.py** - Adapts model to sparse data (6-8 hours)
  - Output: Adapted model checkpoints, training progress logged to terminal
- **Session_2_Step_2_Evaluation.py** - Evaluates sparsity performance (2 hours)
  - Output: Performance metrics and visualization plots
- **Session_3_Step_1_Temporal_Dynamics.py** - Learns temporal behavior (8-10 hours)
  - Output: Temporal model checkpoints, detailed training logs in terminal
- **Session_3_Step_2_Evaluation.py** - Evaluates temporal performance (2 hours)
  - Output: Temporal evaluation results and metrics
- **Session_4_Quick_Analysis.py** - Quick analysis from saved results (10 min)
  - Output: Comprehensive analysis results and visualizations
- **Session_5_PathPlanning_Testing.py** - Tests path planning performance (30 min)
  - Output: Results saved to `../evaluation/session_5_results/` with academic metrics, plots, and summary
- **visualize_s1s2_loss.py** - Visualizes Session 1-2 loss curves
  - Output: Loss curve plots and training progress visualizations
- **visualize_s2s1_loss.py** - Visualizes Session 2-1 loss curves  
  - Output: Comparative loss visualizations
- **visualize_s3s1_loss.py** - Visualizes Session 3-1 loss curves
  - Output: Temporal training loss plots