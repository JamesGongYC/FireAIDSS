# Data Directory

## Contents

- **data_quality_logs/** - Data quality analysis results and comprehensive metrics
- **preprocessors/** - Data preprocessing and standardization scripts
- **standardized/** - Training-ready datasets in standardized format
- **truly_raw/** - Original ANSYS simulation data (raw format)
- **data_quality_analyzer.py** - Universal data quality assessment tool

## Runnable Scripts

- **data_quality_analyzer.py** - Analyzes Session 1, 2, 3 training data quality
  - Output: Creates JSON files in `data_quality_logs/` with standardized quality metrics
- **preprocessors/create_stable_standardized_data.py** - Converts 218k irregular ANSYS data to 16k regular grid
  - Output: `standardized/data_stable_standardized.pkl` (physics-preserving downsampled data)
- **preprocessors/data_preprocessor_for_session_3.py** - Creates temporal standardized data for Session 3
  - Output: `standardized/data_temporal_standardized.pkl` and `data_temporal_standardized_metadata.json`
