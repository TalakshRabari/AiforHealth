"""
models/cnn_model.py - 1D Convolutional Neural Network for Sleep Apnea Detection
=================================================================================
This module defines a 1D CNN model that takes raw windowed physiological signals
as input and classifies each window as Normal, Hypopnea, Apnea, etc.

Architecture explanation:
- Input: (batch_size, n_channels=3, window_length=960)
  - 3 channels: Nasal Airflow, Thoracic Movement, SpO2
  - 960 samples = 30 seconds × 32 Hz
- Two Conv1D blocks: extract local temporal patterns (breathing cycles)
- MaxPooling: downsample and achieve translation invariance
- Fully connected layers: map features to class probabilities

Why 1D CNN for time series?
- 1D convolutions slide along the time axis, detecting patterns like
  breathing cycles regardless of their exact position in the window
- Multiple stacked layers allow detection of increasingly complex patterns
- Much faster than RNNs for inference, works well on 30s windows
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for breathing irregularity classification.

    Input shape : (batch_size, n_channels, window_length)
    Output shape: (batch_size, n_classes)
    """

    def __init__(self, n_channels=3, window_length=960, n_classes=3, dropout=0.4):
        """
        Parameters:
        -----------
        n_channels    : number of signal channels (default 3: nasal, thoracic, spo2)
        window_length : samples per window (default 960 = 30s × 32 Hz)
        n_classes     : number of output classes
        dropout       : dropout rate for regularization
        """
        super(CNN1D, self).__init__()

        # ── Block 1: First convolutional layer ────────────────────────────────
        # kernel_size=7: captures ~0.2s patterns at 32 Hz (first breathing sub-cycles)
        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=32,
                               kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)   # normalize across batch for stable training
        self.pool1 = nn.MaxPool1d(kernel_size=4)  # 960 → 240

        # ── Block 2: Second convolutional layer ───────────────────────────────
        # 64 filters, kernel_size=5: captures slightly longer patterns
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64,
                               kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=4)  # 240 → 60

        # ── Block 3: Third convolutional layer ────────────────────────────────
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128,
                               kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=4)  # 60 → 15

        # ── Global Average Pooling ────────────────────────────────────────────
        # Reduces (batch, 128, 15) → (batch, 128), independent of input length
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # ── Fully connected classifier ────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        """
        Forward pass.
        x shape: (batch_size, n_channels, window_length)
        """
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Global average pooling → flatten
        x = self.global_avg_pool(x)      # (batch, 128, 1)
        x = x.squeeze(-1)                # (batch, 128)

        # Classifier
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                  # raw logits (no softmax — handled by CrossEntropyLoss)

        return x


def get_model(n_classes, n_channels=3, window_length=960):
    """Factory function to create and return a new CNN1D model."""
    return CNN1D(n_channels=n_channels, window_length=window_length, n_classes=n_classes)
