# NeuroFormer Project Structure

This document describes the organization of the NeuroFormer codebase.

## Directory Structure

```
neuroformer/
├── README.md                           # Main documentation
├── LICENSE                             # MIT License
├── setup.py                            # Package setup
├── requirements.txt                    # Python dependencies
├── requirements-dev.txt                # Development dependencies
├── .gitignore                         # Git ignore rules
├── CONTRIBUTING.md                     # Contribution guidelines
├── PROJECT_STRUCTURE.md               # This file
│
├── configs/                           # Configuration files
│   ├── neuroformer_config.yaml       # Base model config
│   ├── neuroformer++_config.yaml     # Model with auxiliary losses
│   ├── preprocessing_config.yaml     # Preprocessing parameters
│   └── ablation_config.yaml          # Ablation study configs
│
├── neuroformer/                       # Main package
│   ├── __init__.py
│   │
│   ├── models/                        # Model implementations
│   │   ├── __init__.py
│   │   ├── neuroformer.py            # Main NeuroFormer model
│   │   ├── encoders.py               # Modality encoders
│   │   ├── fusion.py                 # Fusion mechanisms
│   │   └── baselines.py              # Baseline models
│   │
│   ├── data/                          # Data processing
│   │   ├── __init__.py
│   │   ├── dataset.py                # PyTorch Dataset classes
│   │   ├── augmentation.py           # Data augmentation
│   │   └── preprocessing/            # Preprocessing modules
│   │       ├── __init__.py
│   │       ├── eeg_preprocessing.py  # EEG pipeline
│   │       ├── eyetracking_preprocessing.py  # Eye-tracking pipeline
│   │       └── behavioral_preprocessing.py   # Behavioral pipeline
│   │
│   ├── training/                      # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py                # Training loop
│   │   ├── losses.py                 # Loss functions
│   │   └── metrics.py                # Evaluation metrics
│   │
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       ├── visualization.py          # Plotting and visualization
│       ├── logging.py                # Logging utilities
│       └── config.py                 # Config management
│
├── scripts/                           # Executable scripts
│   ├── preprocess_data.py            # Data preprocessing
│   ├── train.py                      # Model training
│   ├── evaluate.py                   # Model evaluation
│   ├── cross_dataset_eval.py         # Cross-dataset evaluation
│   ├── ablation_modalities.py        # Modality ablation
│   ├── ablation_architecture.py      # Architecture ablation
│   ├── visualize_attention.py        # Attention visualization
│   ├── plot_training.py              # Training curve plots
│   ├── run_all_baselines.sh          # Run all baselines
│   ├── run_neuroformer_variants.sh   # Run model variants
│   └── quickstart.sh                 # Quick start demo
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # Explore raw data
│   ├── 02_preprocessing_demo.ipynb   # Preprocessing examples
│   ├── 03_model_analysis.ipynb       # Model interpretation
│   └── 04_visualization.ipynb        # Result visualization
│
├── tests/                             # Unit tests
│   ├── test_models.py                # Model tests
│   ├── test_preprocessing.py         # Preprocessing tests
│   └── test_dataset.py               # Dataset tests
│
└── docs/                              # Documentation
    ├── installation.md               # Installation guide
    ├── quickstart.md                 # Quick start guide
    ├── api_reference.md              # API documentation
    └── training_guide.md             # Training guide
```

## Key Components

### Models (neuroformer/models/)
- **neuroformer.py**: Main NeuroFormer architecture
  - ModalityEncoder: Transformer encoders for each modality
  - CrossModalFusionTransformer: Cross-modal attention fusion
  - NeuroFormer: Complete model with classification head
  
- **baselines.py**: Baseline model implementations
  - MLPBaseline: Simple MLP with early fusion
  - CNN1DBaseline: 1D CNN for sequences
  - LSTMBaseline: LSTM recurrent model
  - TCNBaseline: Temporal Convolutional Network
  - SimpleTransformer: Basic transformer
  - LateFusionEnsemble: Late fusion approach
  - GatedMultimodalUnit: Gated fusion

### Data Processing (neuroformer/data/)
- **preprocessing/**: Modality-specific preprocessing
  - EEG: Filtering, ICA, epoching, feature extraction
  - Eye-tracking: Fixation/saccade detection, pupil analysis
  - Behavioral: RT analysis, accuracy metrics, trends
  
- **dataset.py**: PyTorch Dataset and DataLoader creation
- **augmentation.py**: Data augmentation strategies

### Training (neuroformer/training/)
- **trainer.py**: Complete training loop with:
  - Learning rate scheduling
  - Early stopping
  - Checkpoint saving
  - Validation tracking
  
- **losses.py**: Loss functions including auxiliary losses
- **metrics.py**: Evaluation metrics (accuracy, F1, AUROC, etc.)

### Scripts (scripts/)
All scripts are command-line ready:
- Data preprocessing pipelines
- Model training with config files
- Comprehensive evaluation
- Ablation studies
- Visualization tools
- Convenience bash scripts

## Usage Flow

1. **Data Preparation**:
   ```bash
   python scripts/preprocess_data.py --data_dir data/raw --output_dir data/processed
   ```

2. **Training**:
   ```bash
   python scripts/train.py --config configs/neuroformer_config.yaml \
       --data_dir data/processed --output_dir experiments/run1
   ```

3. **Evaluation**:
   ```bash
   python scripts/evaluate.py --checkpoint experiments/run1/best_model.pth \
       --data_dir data/processed --output_dir results/
   ```

4. **Analysis**:
   ```bash
   python scripts/visualize_attention.py --checkpoint experiments/run1/best_model.pth
   ```

## Configuration System

All experiments use YAML configuration files in `configs/`:
- Model hyperparameters
- Data processing settings
- Training parameters
- Experiment tracking

Example:
```yaml
model:
  d_model: 256
  n_heads: 8
  dropout: 0.3

training:
  learning_rate: 0.0001
  batch_size: 32
  num_epochs: 60
```

## Extension Points

To add new functionality:
1. **New models**: Add to `neuroformer/models/`
2. **New preprocessing**: Add to `neuroformer/data/preprocessing/`
3. **New metrics**: Add to `neuroformer/training/metrics.py`
4. **New experiments**: Create config in `configs/`

## Testing

Run tests:
```bash
pytest tests/
pytest tests/ --cov=neuroformer  # With coverage
```

## Documentation

API documentation:
```bash
cd docs/
sphinx-build -b html . _build/
```
