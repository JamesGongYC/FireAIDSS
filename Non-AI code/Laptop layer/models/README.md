# FireAIDSS Models Directory

This directory contains the trained AI models used by the FireAIDSS hardware control system.

## Models

### BEST_Stage1_Model.pt
- **Type**: Foundation FireAIDSS Spatial Reconstruction Model
- **Architecture**: 5-layer attention-enhanced neural network
- **Input**: Sparse drone measurements (temperature + wind velocity)
- **Output**: Complete 3D fire field reconstruction (40×40×10 grid)
- **Training**: Session 1 foundation training on steady-state fire scenarios
- **Performance**: ~2.0K MAE on temperature reconstruction
- **Usage**: Real-time fire field prediction for drone path planning

## Model Loading

The model is automatically loaded by `FireAIDSS_Controller.py`:

```python
controller = FireAIDSSController(model_path="models/BEST_Stage1_Model.pt")
```

## Requirements

- PyTorch
- FireAIDSS package (AI directory)
- CUDA-compatible GPU (recommended)

## Model Updates

To update the model:
1. Train new model in AI directory
2. Copy best checkpoint to this directory
3. Update model path in FireAIDSS_Controller.py if needed
