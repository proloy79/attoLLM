import torch
import torch.nn as nn
from attollm.mha import MultiHeadAttention
from attollm.residual import Residual
from attollm.feed_fwd import FeedForward

class TransformerBlock(nn.Module):
    """One pre‑norm transformer block: MHA + FFN with residuals + final norm + optional wt tying."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, dropout: float = 0.0):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.res1 = Residual(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.res2 = Residual(d_model)

        # final normalization
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.res1(x, self.mha, mask)   # attend + residual
        #print('mha', x.shape)
        x = self.res2(x, self.ffn)         # think (FFN) + residual
        #print('res1', x.shape)
        return x