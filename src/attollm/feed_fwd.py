import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """Position‑wise MLP with GELU and dropout."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),   # expand
            nn.GELU(),                  # nonlinearity
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),   # project back
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)