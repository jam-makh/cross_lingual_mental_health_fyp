"""
deep_models.py

PyTorch model definitions for cached-feature mental-health classification.

This file is designed to work with main.py + config.yaml.
The important idea is simple:
    - config.yaml controls model hyperparameters.
    - main.py reads config.yaml.
    - main.py calls get_model_from_config(...).
    - this file builds the requested model using those config values.

Supported architectures:
    lstm
    bilstm
    cnn_rnn

Supported input mode used in this project:
    Cached dense/sparse feature vectors, for example:
        - TF-IDF vectors
        - DistilBERT embeddings

For cached feature vectors, the model uses:
    Linear(input_dim -> hidden_size) -> recurrent model -> classifier
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    LSTM classifier for cached vector features.

    Expected input:
        x shape = (batch_size, input_dim)

    Architecture:
        Linear(input_dim, hidden_size)
        ReLU
        Dropout
        LSTM
        Dropout
        Linear(hidden_size, num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_value = dropout

        self.projection = nn.Linear(input_dim, hidden_size)
        self.activation = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.activation(x)
        x = self.input_dropout(x)

        # Make one vector act like a sequence of length 1.
        x = x.unsqueeze(1)

        output, _ = self.lstm(x)
        output = output[:, -1, :]

        output = self.output_dropout(output)
        return self.classifier(output)


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier for cached vector features.

    Expected input:
        x shape = (batch_size, input_dim)

    Architecture:
        Linear(input_dim, hidden_size)
        ReLU
        Dropout
        BiLSTM
        Dropout
        Linear(hidden_size * 2, num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_value = dropout

        self.projection = nn.Linear(input_dim, hidden_size)
        self.activation = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.activation(x)
        x = self.input_dropout(x)

        # Make one vector act like a sequence of length 1.
        x = x.unsqueeze(1)

        output, _ = self.lstm(x)
        output = output[:, -1, :]

        output = self.output_dropout(output)
        return self.classifier(output)


class CNNRNNClassifier(nn.Module):
    """
    CNN + BiGRU classifier for cached vector features.

    Expected input:
        x shape = (batch_size, input_dim)

    Architecture:
        Linear(input_dim, hidden_size)
        ReLU
        Dropout
        Repeat vector into a short pseudo-sequence
        Conv1d
        BiGRU
        Dropout
        Linear(hidden_size * 2, num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        cnn_out_channels: int,
        cnn_kernel_size: int,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_value = dropout
        self.cnn_out_channels = cnn_out_channels
        self.cnn_kernel_size = cnn_kernel_size

        self.projection = nn.Linear(input_dim, hidden_size)
        self.activation = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout)

        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=cnn_out_channels,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2,
        )

        self.rnn = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.activation(x)
        x = self.input_dropout(x)

        # Convert one feature vector into a short pseudo-sequence so Conv1d can run.
        x = x.unsqueeze(1).repeat(1, self.cnn_kernel_size, 1)
        x = x.transpose(1, 2)

        x = self.conv(x)
        x = self.activation(x)
        x = x.transpose(1, 2)

        output, _ = self.rnn(x)
        output = output[:, -1, :]

        output = self.output_dropout(output)
        return self.classifier(output)


def get_model(
    architecture: str,
    input_dim: int,
    num_classes: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    cnn_out_channels: int = 128,
    cnn_kernel_size: int = 3,
) -> nn.Module:
    """
    Build one model using explicit parameters.

    main.py usually calls get_model_from_config(), but this function is kept
    available for direct usage and testing.
    """

    architecture = architecture.lower().strip()

    if architecture == "lstm":
        return LSTMClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    if architecture == "bilstm":
        return BiLSTMClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    if architecture in {"cnn_rnn", "cnnrnn", "cnn-rnn"}:
        return CNNRNNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cnn_out_channels=cnn_out_channels,
            cnn_kernel_size=cnn_kernel_size,
        )

    raise ValueError("architecture must be one of: lstm | bilstm | cnn_rnn")


def get_model_from_config(
    architecture: str,
    input_dim: int,
    num_classes: int,
    config: Dict[str, Any],
) -> nn.Module:
    """
    Build one model directly from config.yaml.

    Parameters read from config:
        model.hidden_size
        model.num_layers
        model.dropout
        model.cnn_out_channels
        model.cnn_kernel_size

    Runtime parameters:
        architecture: selected from config.model.architectures
        input_dim: selected from config.vectorizer_dimensions or detected from cache
        num_classes: selected from config.num_classes or detected from labels
    """

    model_cfg = config.get("model", {}) or {}

    return get_model(
        architecture=architecture,
        input_dim=int(input_dim),
        num_classes=int(num_classes),
        hidden_size=int(model_cfg["hidden_size"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        cnn_out_channels=int(model_cfg.get("cnn_out_channels", 128)),
        cnn_kernel_size=int(model_cfg.get("cnn_kernel_size", 3)),
    )
