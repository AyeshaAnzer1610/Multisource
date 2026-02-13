"""
Baseline models for comparison with NeuroFormer.

Includes:
- MLP (early fusion)
- 1D CNN
- LSTM
- TCN (Temporal Convolutional Network)
- Simple Transformer
- Late Fusion Ensemble
- GMU (Gated Multimodal Unit)
- GAT (Graph Attention Network)
- Cross-Modal Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


class MLPBaseline(nn.Module):
    """Simple MLP with early fusion."""
    
    def __init__(
        self,
        eeg_dim: int = 128,
        eye_dim: int = 96,
        beh_dim: int = 64,
        hidden_dims: List[int] = [512, 256, 128],
        n_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        input_dim = eeg_dim + eye_dim + beh_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, eeg, eyetracking, behavioral, **kwargs):
        # Flatten sequences and concatenate
        eeg_flat = eeg.mean(dim=1)
        eye_flat = eyetracking.mean(dim=1)
        beh_flat = behavioral.mean(dim=1)
        
        x = torch.cat([eeg_flat, eye_flat, beh_flat], dim=-1)
        logits = self.model(x)
        
        return {'logits': logits, 'probs': F.softmax(logits, dim=-1)}


class CNN1DBaseline(nn.Module):
    """1D CNN for temporal sequence processing."""
    
    def __init__(
        self,
        eeg_dim: int = 128,
        eye_dim: int = 96,
        beh_dim: int = 64,
        n_filters: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [7, 5, 3],
        n_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        input_dim = eeg_dim + eye_dim + beh_dim
        
        conv_layers = []
        in_channels = input_dim
        for n_filter, kernel_size in zip(n_filters, kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(in_channels, n_filter, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(n_filter),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = n_filter
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(n_filters[-1], n_classes)
    
    def forward(self, eeg, eyetracking, behavioral, **kwargs):
        # Concatenate modalities
        x = torch.cat([eeg, eyetracking, behavioral], dim=-1)
        
        # Transpose for Conv1d (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classify
        logits = self.classifier(x)
        
        return {'logits': logits, 'probs': F.softmax(logits, dim=-1)}


class LSTMBaseline(nn.Module):
    """LSTM for temporal sequence modelling."""
    
    def __init__(
        self,
        eeg_dim: int = 128,
        eye_dim: int = 96,
        beh_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 2,
        n_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        input_dim = eeg_dim + eye_dim + beh_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, eeg, eyetracking, behavioral, **kwargs):
        # Concatenate modalities
        x = torch.cat([eeg, eyetracking, behavioral], dim=-1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final states
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            hidden = h_n[-1]
        
        # Classify
        logits = self.classifier(hidden)
        
        return {'logits': logits, 'probs': F.softmax(logits, dim=-1)}


class TemporalBlock(nn.Module):
    """Temporal convolutional block for TCN."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        out = self.relu2(out + res)
        out = self.dropout2(out)
        
        # Remove extra padding
        return out[:, :, :x.size(2)]


class TCNBaseline(nn.Module):
    """Temporal Convolutional Network."""
    
    def __init__(
        self,
        eeg_dim: int = 128,
        eye_dim: int = 96,
        beh_dim: int = 64,
        n_channels: List[int] = [64, 128, 256],
        kernel_size: int = 7,
        n_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        input_dim = eeg_dim + eye_dim + beh_dim
        
        layers = []
        num_levels = len(n_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else n_channels[i-1]
            out_ch = n_channels[i]
            
            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size, dilation, dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(n_channels[-1], n_classes)
    
    def forward(self, eeg, eyetracking, behavioral, **kwargs):
        # Concatenate modalities
        x = torch.cat([eeg, eyetracking, behavioral], dim=-1)
        
        # Transpose for Conv1d
        x = x.transpose(1, 2)
        
        # TCN
        x = self.network(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classify
        logits = self.classifier(x)
        
        return {'logits': logits, 'probs': F.softmax(logits, dim=-1)}


class SimpleTransformer(nn.Module):
    """Simple transformer baseline with concatenated modalities."""
    
    def __init__(
        self,
        eeg_dim: int = 128,
        eye_dim: int = 96,
        beh_dim: int = 64,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        n_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        input_dim = eeg_dim + eye_dim + beh_dim
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 500, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, eeg, eyetracking, behavioral, **kwargs):
        # Concatenate modalities
        x = torch.cat([eeg, eyetracking, behavioral], dim=-1)
        
        # Project and add positional encoding
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer
        x = self.transformer(x)
        
        # Use mean pooling
        x = x.mean(dim=1)
        
        # Classify
        logits = self.classifier(x)
        
        return {'logits': logits, 'probs': F.softmax(logits, dim=-1)}


class LateFusionEnsemble(nn.Module):
    """Late fusion ensemble of modality-specific classifiers."""
    
    def __init__(
        self,
        eeg_dim: int = 128,
        eye_dim: int = 96,
        beh_dim: int = 64,
        hidden_dim: int = 256,
        n_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Modality-specific classifiers
        self.eeg_classifier = self._make_classifier(eeg_dim, hidden_dim, n_classes, dropout)
        self.eye_classifier = self._make_classifier(eye_dim, hidden_dim, n_classes, dropout)
        self.beh_classifier = self._make_classifier(beh_dim, hidden_dim, n_classes, dropout)
    
    def _make_classifier(self, input_dim, hidden_dim, n_classes, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self, eeg, eyetracking, behavioral, **kwargs):
        # Average pooling over sequence
        eeg_pooled = eeg.mean(dim=1)
        eye_pooled = eyetracking.mean(dim=1)
        beh_pooled = behavioral.mean(dim=1)
        
        # Get predictions from each modality
        eeg_logits = self.eeg_classifier(eeg_pooled)
        eye_logits = self.eye_classifier(eye_pooled)
        beh_logits = self.beh_classifier(beh_pooled)
        
        # Average logits
        logits = (eeg_logits + eye_logits + beh_logits) / 3.0
        
        return {'logits': logits, 'probs': F.softmax(logits, dim=-1)}


class GatedMultimodalUnit(nn.Module):
    """Gated Multimodal Unit (GMU) for adaptive fusion."""
    
    def __init__(
        self,
        eeg_dim: int = 128,
        eye_dim: int = 96,
        beh_dim: int = 64,
        hidden_dim: int = 256,
        n_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Project each modality
        self.eeg_proj = nn.Linear(eeg_dim, hidden_dim)
        self.eye_proj = nn.Linear(eye_dim, hidden_dim)
        self.beh_proj = nn.Linear(beh_dim, hidden_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid()
        )
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, eeg, eyetracking, behavioral, **kwargs):
        # Pool sequences
        eeg_pooled = eeg.mean(dim=1)
        eye_pooled = eyetracking.mean(dim=1)
        beh_pooled = behavioral.mean(dim=1)
        
        # Project modalities
        h_eeg = self.eeg_proj(eeg_pooled)
        h_eye = self.eye_proj(eye_pooled)
        h_beh = self.beh_proj(beh_pooled)
        
        # Concatenate for gating
        h_concat = torch.cat([h_eeg, h_eye, h_beh], dim=-1)
        
        # Compute gates
        gate_weights = self.gate(h_concat)
        
        # Fuse with gating
        h_fused = gate_weights * (h_eeg + h_eye + h_beh)
        
        # Classify
        logits = self.classifier(h_fused)
        
        return {'logits': logits, 'probs': F.softmax(logits, dim=-1)}


def create_baseline(model_name: str, config: dict) -> nn.Module:
    """Factory function to create baseline models."""
    
    models = {
        'mlp': MLPBaseline,
        'cnn': CNN1DBaseline,
        'lstm': LSTMBaseline,
        'tcn': TCNBaseline,
        'simple_transformer': SimpleTransformer,
        'late_fusion': LateFusionEnsemble,
        'gmu': GatedMultimodalUnit,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name](**config)


if __name__ == "__main__":
    # Test all baselines
    batch_size = 4
    seq_len = 300
    
    eeg = torch.randn(batch_size, seq_len, 128)
    eye = torch.randn(batch_size, seq_len, 96)
    beh = torch.randn(batch_size, seq_len, 64)
    
    for name in ['mlp', 'cnn', 'lstm', 'tcn', 'simple_transformer', 'late_fusion', 'gmu']:
        model = create_baseline(name, {})
        output = model(eeg, eye, beh)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {output['logits'].shape}, {params/1e6:.2f}M params")
