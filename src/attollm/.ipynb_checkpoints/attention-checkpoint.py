import torch
from torch import Tensor
from typing import Tuple

def scaled_dot_product_attention(q: Tensor,k: Tensor,v: Tensor,mask: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor]:
    # q,k,v: [B, T, D]
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / (d ** 0.5) # [B, T, T]
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf")) # Remove masked
    
    w = torch.softmax(scores, dim=-1) # [B, T, T]
    x = w @ v # [B, T, D]

    return scores, w, x