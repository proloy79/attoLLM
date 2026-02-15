from __future__ import annotations

import torch
import torch.nn as nn
import math

def sinusoidal_positions(T: int, d_model: int) -> torch.Tensor:
        pos = torch.arange(T).float()[:, None] # Positions as column
        i = torch.arange(d_model).float()[None, :] # Dimension index
        angle = pos / (10000 ** (2 * (i // 2) / d_model)) # Frequency grid
        enc = torch.zeros(T, d_model)
        enc[:, 0::2] = torch.sin(angle[:, 0::2]) # Sine on even dims
        enc[:, 1::2] = torch.cos(angle[:, 1::2]) # Cosine on odd dims
        
        return enc # [T, D]
    
class EmbeddingBlock(nn.Module):
    """
    This will embed the tokens, apply positionals embedding on top and normalize the output.
    """
    def __init__(self, vocab_size, d_model, max_len=128, use_sinusoidal=False):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.use_sinusoidal = use_sinusoidal

        if use_sinusoidal:
            pe = sinusoidal_positions(max_len, d_model, device)
            self.register_buffer("pos_emb", pe)   # not trainable
        else:
            self.pos_emb = nn.Embedding(max_len, d_model)  # trainable

    def forward(self, ids):
        # ids: [B, T]
        B, T = ids.shape

        tok = self.token_emb(ids)                     # [B, T, D]
        pos_ids = torch.arange(T, device=ids.device)  # [T]
        pos = self.pos_emb(pos_ids)[None, :, :]       # [1, T, D]

        x = tok + pos                                 # add positional info
        x = self.ln(x)                                # pre norm

        return x                                     # [B, T, D]
       
    @staticmethod
    def create_combined_mask(P, batch):
        L = max(len(s) for s in batch) # Target length
        right_pad = [s + [P] * (L - len(s)) for s in batch] # Usual right pad
        #left_pad = [[P] * (L - len(s)) + s for s in batch] # Left pad for causal models
    
        x = torch.tensor(right_pad) # Batch -> tensor
        pad_mask = (x != P).float() # [B, T] mask of real tokens
        T = x.size(1) # Sequence length
        causal = torch.tril(torch.ones(T, T)) # Lower-triangular causal mask
        combined = pad_mask[:, None, :] * causal # Broadcast pad over heads∕time
        # pad_mask.shape, causal.shape, combined.shape
    
        return combined

    @staticmethod
    def pad_batch_ids(pad_id: int, batch_ids):
        max_len = max(len(seq) for seq in batch_ids)
        padded = [seq + [pad_id] * (max_len - len(seq)) for seq in batch_ids]
        return torch.tensor(padded, dtype=torch.long)   # (B, T)
        