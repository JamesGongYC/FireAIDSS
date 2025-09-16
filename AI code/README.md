# AI Code Directory

## Contents

- **checkpoints/** - Model checkpoints and training artifacts
- **data/** - Data processing, quality analysis, and standardized datasets
- **docs/** - Documentation files
- **evaluation/** - Model evaluation results and performance metrics
- **fireaidss/** - Core AI package with model, loss, data, and utility modules
- **training_sessions/** - Progressive training pipeline scripts
- **utilities/** - Environment checking and utility scripts

## Runnable Scripts

### data/ Directory
- **data_quality_analyzer.py** - Analyzes training data quality and outputs metrics
  - Output: `data_quality_logs/` directory with comprehensive quality analysis JSON files
- **preprocessors/create_stable_standardized_data.py** - Converts raw ANSYS data to standardized format
  - Output: `standardized/data_stable_standardized.pkl` (training-ready dataset)
- **preprocessors/data_preprocessor_for_session_3.py** - Creates temporal standardized data
  - Output: `standardized/data_temporal_standardized.pkl` and metadata JSON

### training_sessions/ Directory
- **Session_1_Step_1_Data_Generation.py** - Generates training data (30 min runtime)
  - Output: Check terminal for progress, creates training samples
- **Session_1_Step_2_Training.py** - Trains foundation model (4-6 hours)
  - Output: Model checkpoints in `../checkpoints/`, training logs in terminal
- **Session_1_Step_3_Evaluation.py** - Evaluates foundation model (1 hour)
  - Output: Evaluation metrics in terminal and checkpoint directory
- **Session_2_Step_1_Sparsity_Adaptation.py** - Adapts model to sparse data (6-8 hours)
  - Output: Adapted model checkpoints, training progress in terminal
- **Session_2_Step_2_Evaluation.py** - Evaluates sparsity performance (2 hours)
  - Output: Performance metrics and visualizations
- **Session_3_Step_1_Temporal_Dynamics.py** - Learns temporal behavior (8-10 hours)
  - Output: Temporal model checkpoints, training logs
- **Session_3_Step_2_Evaluation.py** - Evaluates temporal performance (2 hours)
  - Output: Temporal evaluation results
- **Session_4_Quick_Analysis.py** - Quick analysis from saved results (10 min)
  - Output: Analysis results and visualizations
- **Session_5_PathPlanning_Testing.py** - Tests path planning performance (30 min)
  - Output: Results in `../evaluation/session_5_results/` with academic metrics and plots
- **visualize_*.py** - Loss curve visualization scripts
  - Output: Loss curve plots and training progress visualizations
