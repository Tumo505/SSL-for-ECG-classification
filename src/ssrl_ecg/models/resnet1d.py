#!/usr/bin/env python3
"""ResNet-1D baseline model for ECG classification."""

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """1D Residual Block for ResNet."""
    
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, 
                               padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """ResNet-1D for ECG classification."""
    
    def __init__(self, in_channels=12, num_classes=5, width=64, depth=18):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        
        # Initial layer
        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=15, stride=1, 
                               padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        channels = [width, width*2, width*4, width*8]
        self.layer1 = self._make_layer(width, channels[0], 2, kernel_size=7)
        self.layer2 = self._make_layer(channels[0], channels[1], 2, stride=2, kernel_size=7)
        self.layer3 = self._make_layer(channels[1], channels[2], 2, stride=2, kernel_size=7)
        self.layer4 = self._make_layer(channels[2], channels[3], 2, stride=2, kernel_size=7)
        
        # Adaptive pooling and classification head
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[3], num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, kernel_size=7):
        """Create a layer with multiple residual blocks."""
        layers = []
        
        # First block with stride
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            downsample = None
        
        layers.append(ResidualBlock1D(in_channels, out_channels, kernel_size, 
                                     stride=stride, downsample=downsample))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, kernel_size))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        # Classification head
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # Test model
    model = ResNet1D(in_channels=12, num_classes=5, width=64)
    x = torch.randn(4, 12, 1000)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
