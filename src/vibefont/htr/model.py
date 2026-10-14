"""Online Handwriting Text Recognition (HTR) model using 1D CNN + BiLSTM + CTC.

Used for:
  1. CTC Forced Alignment: Mapping stroke points -> character indices (adv/pointer conditioning).
  2. Readability Evaluation: Measuring CER/WER on generated or reconstructed ink.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..text_codec import VOCAB_SIZE


class OnlineHTR(nn.Module):
    """Input: (B, T, 4) stroke features (dx, dy, dt, pen).

    Output: (T', B, num_classes) CTC logits.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = VOCAB_SIZE + 1,  # +1 for CTC blank (index 0)
        conv_dim: int = 128,
        rnn_dim: int = 128,
        num_rnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blank_idx = 0
        self.num_classes = num_classes

        # 1D Convolutional feature extractor (temporal stride 2 to compress sequence)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(conv_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(conv_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Bidirectional GRU / LSTM
        self.rnn = nn.GRU(
            input_size=conv_dim,
            hidden_size=rnn_dim,
            num_layers=num_rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_rnn_layers > 1 else 0.0,
        )

        # Classifier head
        self.fc = nn.Linear(rnn_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 4) -> returns (T', B, num_classes) for F.ctc_loss."""
        # Conv1d expects (B, C, T)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.rnn(h)
        logits = self.fc(out)  # (B, T', num_classes)
        return logits.transpose(0, 1)  # (T', B, num_classes)
