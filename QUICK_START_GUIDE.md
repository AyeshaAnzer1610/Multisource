# NeuroFormer Complete Implementation - Quick Start Guide

##  Package Contents

This archive contains the **complete, production-ready implementation** of NeuroFormer from the paper "Transformer-Based Multimodal Integration for Diagnostic Modelling of Mental Health" (IEEE Access 2025).

### What's Included:

 **Full Model Implementation**
- Complete NeuroFormer architecture with modality-specific encoders
- Cross-modal fusion transformer
- All 9 baseline models (MLP, CNN, LSTM, TCN, etc.)
- NeuroFormer++ variant with auxiliary reconstruction losses

 **Data Preprocessing Pipelines**
- EEG preprocessing (filtering, ICA, feature extraction)
- Eye-tracking preprocessing (fixation/saccade detection, pupil analysis)
- Behavioral data preprocessing (RT analysis, accuracy metrics)

 **Training & Evaluation Infrastructure**
- Complete training loop with early stopping
- Comprehensive evaluation metrics
- Cross-dataset evaluation framework
- Attention visualization tools

 **Experiment Reproduction**
- Configuration files for all experiments in the paper
- Ablation study scripts (modality and architecture)
- Baseline comparison scripts
- Cross-dataset generalization scripts

 **Documentation**
- Comprehensive README with installation instructions
- API reference documentation
- Example Jupyter notebooks
- Contributing guidelines

## 🚀 Installation

### Step 1: Extract the Archive

```bash
tar -xzf neuroformer_complete_code.tar.gz
cd neuroformer_project
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Step 4: Verify Installation

```bash
python -c "import neuroformer; print('Installation successful!')"
```

## 📊 Data Preparation

### Download CMI Dataset

1. Visit the [CMI Healthy Brain Network portal](http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/)
2. Accept the data use agreement
3. Download the following data:
   - EEG recordings (.edf format)
   - Eye-tracking data (.asc format)
   - Behavioral task data (.csv format)
   - Clinical assessments (phenotypic data)

### Organize Data Structure

```
data/
├── raw/
│   ├── eeg/
│   │   ├── sub-001_task-gonogo_eeg.edf
│   │   └── ...
│   ├── eyetracking/
│   │   ├── sub-001_task-gonogo_eyetrack.asc
│   │   └── ...
│   ├── behavioral/
│   │   ├── sub-001_task-gonogo_beh.csv
│   │   └── ...
│   └── phenotypic/
│       └── participants.csv
└── processed/
    └── (will be generated)
```

### Preprocess Data

```bash
python scripts/preprocess_data.py \
    --data_dir data/raw \
    --output_dir data/processed \
    --config configs/preprocessing_config.yaml
```

**Expected output:**
- `train_samples.npz`: Training data
- `val_samples.npz`: Validation data
- `test_samples.npz`: Test data

## 🎯 Training Models

### Train NeuroFormer (Base Model)

```bash
python scripts/train.py \
    --config configs/neuroformer_config.yaml \
    --data_dir data/processed \
    --output_dir experiments/neuroformer_base
```

**Expected results:**
- Accuracy: ~80.3% ± 2.3%
- Training time: ~2.2 hours/fold (NVIDIA RTX 3090)

### Train NeuroFormer++ (With Auxiliary Losses)

```bash
python scripts/train.py \
    --config configs/neuroformer++_config.yaml \
    --data_dir data/processed \
    --output_dir experiments/neuroformer_plus
```

**Expected results:**
- Accuracy: ~85.1% ± 1.9%
- Training time: ~2.7 hours/fold

### Train All Baselines

```bash
bash scripts/run_all_baselines.sh
```

This will train all 9 baseline models and save results to `experiments/baselines/`.

## 📈 Evaluation

### Evaluate on Test Set

```bash
python scripts/evaluate.py \
    --checkpoint experiments/neuroformer_base/best_model.pth \
    --data_dir data/processed \
    --output_dir results/neuroformer_base
```

**Outputs:**
- `metrics.npy`: All evaluation metrics
- `confusion_matrix.npy`: Confusion matrix
- `confusion_matrix.png`: Visualization

### Cross-Dataset Evaluation

```bash
python scripts/cross_dataset_eval.py \
    --checkpoint experiments/neuroformer_base/best_model.pth \
    --external_datasets data/external/ \
    --mode zero_shot \
    --output results/cross_dataset/
```

**Supported external datasets:**
- MODMA (Depression)
- ADHD-200 (ADHD)
- COBRE (Schizophrenia)
- DEAP (Emotion)
- EmotionNet (Emotion)

## 🔬 Ablation Studies

### Modality Ablation

Test contribution of each modality:

```bash
python scripts/ablation_modalities.py \
    --config configs/ablation_config.yaml \
    --data_dir data/processed \
    --output results/ablation_modalities.csv
```

**Tests all combinations:**
- EEG only
- Eye-tracking only
- Behavioral only
- EEG + Eye
- EEG + Behavioral
- Eye + Behavioral
- All three (baseline)

### Architecture Ablation

Test different architectural configurations:

```bash
python scripts/ablation_architecture.py \
    --config configs/ablation_config.yaml \
    --data_dir data/processed \
    --output results/ablation_architecture.csv
```

**Tests:**
- Number of encoder layers (2, 4, 6, 8)
- Number of fusion layers (4, 6, 8)
- Number of attention heads (4, 8, 12, 16)
- Temporal segment length (1s, 2s, 3s, 4s)

## 📊 Visualization

### Visualize Attention Weights

```bash
python scripts/visualize_attention.py \
    --checkpoint experiments/neuroformer_base/best_model.pth \
    --data_dir data/processed \
    --output_dir visualizations/attention/
```

**Generates:**
- Cross-modal attention heatmaps
- Temporal attention patterns
- Modality-level attention scores

### Plot Training Curves

```bash
python scripts/plot_training.py \
    --log_dir experiments/neuroformer_base/logs \
    --output figures/training_curves.png
```

## 🔧 Configuration

All experiments use YAML configuration files. Modify `configs/*.yaml` to change:

### Model Hyperparameters

```yaml
model:
  d_model: 256              # Embedding dimension
  n_encoder_layers: 4       # Modality encoder depth
  n_fusion_layers: 6        # Fusion transformer depth
  n_heads: 8                # Number of attention heads
  dropout: 0.3              # Dropout rate
```

### Training Parameters

```yaml
training:
  num_epochs: 60
  learning_rate: 0.0001
  batch_size: 32
  early_stopping_patience: 10
```

### Data Processing

```yaml
preprocessing:
  eeg:
    sampling_rate: 500
    n_channels: 5
    lowcut: 0.5
    highcut: 50.0
  eyetracking:
    sampling_rate: 60
    window_length: 1.0
  behavioral:
    rt_min: 0.15
    rt_max: 2.0
```

## 📝 Example Usage in Python

### Load and Use Trained Model

```python
import torch
from neuroformer.models.neuroformer import create_neuroformer

# Load config
import yaml
with open('configs/neuroformer_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model
model = create_neuroformer(config['model'])

# Load checkpoint
checkpoint = torch.load('experiments/neuroformer_base/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    outputs = model(eeg, eyetracking, behavioral)
    predictions = torch.argmax(outputs['probs'], dim=1)
```

### Custom Training Loop

```python
from neuroformer.training.trainer import NeuroFormerTrainer
from neuroformer.data.dataset import create_dataloaders

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    data_dir='data/processed',
    batch_size=32,
    modalities=['eeg', 'eyetracking', 'behavioral']
)

# Create trainer
trainer = NeuroFormerTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config['training']
)

# Train
trainer.train(
    num_epochs=60,
    save_dir='experiments/custom_run'
)
```

## 📚 Jupyter Notebooks

Explore interactive examples in the `notebooks/` directory:

1. **01_data_exploration.ipynb**
   - Visualize raw EEG, eye-tracking, and behavioral data
   - Understand data distributions and quality

2. **02_preprocessing_demo.ipynb**
   - Step-by-step preprocessing demonstration
   - Feature extraction visualization

3. **03_model_analysis.ipynb**
   - Model architecture analysis
   - Attention mechanism visualization
   - Feature importance analysis

4. **04_visualization.ipynb**
   - Results visualization
   - Cross-dataset comparison plots
   - Publication-quality figures

## 🎓 Reproducing Paper Results

### Main Results (Table 2)

```bash
# Train all models
bash scripts/run_all_baselines.sh
bash scripts/run_neuroformer_variants.sh

# Generate results table
python scripts/generate_results_table.py \
    --experiments_dir experiments/ \
    --output results/main_results.csv
```

### Cross-Dataset Generalization (Table 5)

```bash
# Download external datasets (MODMA, ADHD-200, COBRE, DEAP, EmotionNet)
# Place in data/external/

# Run zero-shot evaluation
python scripts/cross_dataset_eval.py \
    --checkpoint experiments/neuroformer_base/best_model.pth \
    --external_datasets data/external/ \
    --mode zero_shot \
    --output results/cross_dataset_zero_shot.csv
```

## 💻 Hardware Requirements

### Minimum Requirements

- CPU: 4+ cores
- RAM: 16GB
- Storage: 50GB
- GPU: Optional (CPU training supported)

### Recommended for Paper Reproduction

- GPU: NVIDIA RTX 3090 (24GB VRAM) or equivalent
- RAM: 64GB
- Storage: 100GB SSD

## 🐛 Troubleshooting

### Common Issues

**1. CUDA out of memory**
```bash
# Reduce batch size in config
batch_size: 16  # instead of 32
```

**2. MNE import error**
```bash
pip install --upgrade mne
```

**3. Preprocessing takes too long**
```bash
# Use parallel processing
python scripts/preprocess_data.py --n_jobs 8
```

**4. Missing data files**
```bash
# Check data directory structure
python scripts/check_data_integrity.py --data_dir data/raw
```

##  Documentation

Full documentation available at:
- **README.md**: General overview
- **PROJECT_STRUCTURE.md**: Codebase organization
- **CONTRIBUTING.md**: Development guidelines
- **docs/**: Detailed API reference

##  Support

For questions or issues:
1. Check the documentation first
2. Open an issue on GitHub
3. Contact the authors:
   - Ayesha Anzer: aag833@uregina.ca
   - Abdul Bais: abdul.bais@uregina.ca

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{anzer2025neuroformer,
  title={Transformer-Based Multimodal Integration for Diagnostic Modelling of Mental Health},
  author={Anzer, Ayesha and Bais, Abdul},
  journal={IEEE Access},
  year={2025},
  volume={},
  pages={},
  doi={10.1109/ACCESS.2025.0429000}
}
```

##  Quick Checklist

- [ ] Extract archive
- [ ] Install dependencies
- [ ] Download CMI dataset
- [ ] Preprocess data
- [ ] Train NeuroFormer
- [ ] Evaluate on test set
- [ ] Run ablation studies
- [ ] Visualize results

## 🚀 One-Command Quick Start

For a quick demo with synthetic data:

```bash
# Extract, install, and run demo
tar -xzf neuroformer_complete_code.tar.gz && \
cd neuroformer_project && \
pip install -e . && \
bash scripts/quickstart.sh
```

---

**Note:** This implementation is for research purposes. Clinical deployment requires additional validation, regulatory approval, and ethical considerations.

**License:** MIT License - see LICENSE file for details.

**Version:** 1.0.0 (February 2025)
