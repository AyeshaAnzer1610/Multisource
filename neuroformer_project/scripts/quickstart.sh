#!/bin/bash
# Quick start script for NeuroFormer

echo "NeuroFormer Quick Start"
echo "======================="

# Check if data directory exists
if [ ! -d "data/processed" ]; then
    echo "Error: data/processed directory not found"
    echo "Please run preprocessing first: python scripts/preprocess_data.py"
    exit 1
fi

# Create experiment directory
mkdir -p experiments/quickstart

# Train NeuroFormer base model
echo ""
echo "Training NeuroFormer (base model)..."
python scripts/train.py \
    --config configs/neuroformer_config.yaml \
    --data_dir data/processed \
    --output_dir experiments/quickstart

# Evaluate model
echo ""
echo "Evaluating model..."
python scripts/evaluate.py \
    --checkpoint experiments/quickstart/best_model.pth \
    --data_dir data/processed \
    --output_dir experiments/quickstart/results

echo ""
echo "Training and evaluation complete!"
echo "Results saved to: experiments/quickstart/results"
