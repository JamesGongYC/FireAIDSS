# Training Sessions - 3-Stage Progressive Training Pipeline

## Directory Contents

**3-Stage Progressive Training:**
1. **Session 1**: Foundation training on steady-state fields
2. **Session 2**: Sparsity adaptation with curriculum learning  
3. **Session 3**: Temporal dynamics across all fire phases

## Training Scripts

- **Session_1_Step_1_Data_Generation.py**: Generate training data (30 min)
- **Session_1_Step_2_Training.py**: Train foundation model (4-6 hours)
- **Session_1_Step_3_Evaluation.py**: Evaluate foundation model (1 hour)
- **Session_2_Step_1_Sparsity_Adaptation.py**: Adapt to sparse data (6-8 hours)
- **Session_2_Step_2_Evaluation.py**: Evaluate sparsity performance (2 hours)
- **Session_3_Step_1_Temporal_Dynamics.py**: Learn temporal behavior (8-10 hours)
- **Session_3_Step_2_Evaluation.py**: Evaluate temporal performance (2 hours)
- **Session_4_Quick_Analysis.py**: Quick analysis from saved results (10 min)
- **Session_5_PathPlanning_Testing.py**: Test path planning (30 min)
- **visualize_*.py**: Loss curve visualization scripts

## What to Run

```bash
# Complete pipeline (24-30 hours total)
python Session_1_Step_1_Data_Generation.py      # 30 min
python Session_1_Step_2_Training.py             # 4-6 hours  
python Session_1_Step_3_Evaluation.py           # 1 hour
python Session_2_Step_1_Sparsity_Adaptation.py  # 6-8 hours
python Session_2_Step_2_Evaluation.py           # 2 hours
python Session_3_Step_1_Temporal_Dynamics.py    # 8-10 hours
python Session_3_Step_2_Evaluation.py           # 2 hours
```

## What to Expect

- **Session 1**: 2.0K MAE on temperature reconstruction
- **Session 2**: Models for 3-20 measurements (extreme sparsity)
- **Session 3**: Temporal models for all fire phases (0-10s)
