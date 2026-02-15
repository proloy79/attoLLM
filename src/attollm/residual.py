import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model) # Pre-norm layer
    
    def forward(self, 
                x: torch.Tensor, 
                sublayer: nn.Module, 
                *args, **kwargs):
        
        out = sublayer(self.norm(x), *args, **kwargs)
        return x + out # Norm -> sublayer -> residual