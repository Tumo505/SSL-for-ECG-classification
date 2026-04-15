"""Improved CNN architectures for ECG classification with better capacity."""

import torch
import torch.nn as nn


class ImprovedConvBlock(nn.Module):
    """Improved conv block with residual connection and dropout."""
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.residual = (in_ch == out_ch and stride == 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.residual else None
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        if identity is not None:
            out = out + identity
        return out


class ImprovedECGEncoder(nn.Module):
    """Improved ECG encoder with deeper architecture and better capacity."""
    
    def __init__(self, in_ch: int = 12, width: int = 64, dropout: float = 0.2):
        super().__init__()
        
        # Initial conv to increase channels
        self.initial = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=15, stride=1, padding=7, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        
        # Deeper feature extraction with residual blocks
        self.features = nn.Sequential(
            ImprovedConvBlock(width, width, kernel_size=11, stride=2, dropout=dropout),
            ImprovedConvBlock(width, width, kernel_size=7, stride=1, dropout=dropout),
            ImprovedConvBlock(width, width * 2, kernel_size=7, stride=2, dropout=dropout),
            ImprovedConvBlock(width * 2, width * 2, kernel_size=5, stride=1, dropout=dropout),
            ImprovedConvBlock(width * 2, width * 2, kernel_size=5, stride=1, dropout=dropout),
            ImprovedConvBlock(width * 2, width * 4, kernel_size=5, stride=2, dropout=dropout),
            ImprovedConvBlock(width * 4, width * 4, kernel_size=3, stride=1, dropout=dropout),
            ImprovedConvBlock(width * 4, width * 4, kernel_size=3, stride=1, dropout=dropout),
        )
        
        self.out_channels = width * 4
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.features(x)
        return x


class ImprovedECGClassifier(nn.Module):
    """Improved classifier with better head and regularization."""
    
    def __init__(self, encoder: ImprovedECGEncoder, n_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Better classification head
        hidden_dim = self.encoder.out_channels
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for concat avg+max pool
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        # Use both average and max pooling
        avg_pool = self.pool(z).squeeze(-1)
        max_pool = self.max_pool(z).squeeze(-1)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        return self.head(combined)


class BinaryECGClassifier(nn.Module):
    """Binary classifier for individual diagnostic class prediction."""
    
    def __init__(self, encoder: ImprovedECGEncoder, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Binary classification head
        hidden_dim = self.encoder.out_channels
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        avg_pool = self.pool(z).squeeze(-1)
        max_pool = self.max_pool(z).squeeze(-1)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        return self.head(combined).squeeze(-1)  # Return logits
