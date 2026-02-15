from attollm.attention import scaled_dot_product_attention
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """Multi‑head self‑attention (single module, H heads).

    d_model = H * Dh, where Dh is per‑head dim. We project to Q,K,V, split into
    heads, apply scaled dot‑product attention per head, then concat and project out.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.h = num_heads                                     # number of heads
        self.d = d_model // num_heads                          # per‑head dim
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False) # shared proj
        self.out = nn.Linear(d_model, d_model, bias=False)     # output proj
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, Dm = x.shape                                     # batch, time, model dim
        qkv = self.qkv(x)                                      # [B, T, 3*Dm]
        q, k, v = qkv.chunk(3, dim=-1)                         # each [B, T, Dm]

        # Split heads: [B,T,Dm] -> [B,H,T,Dh],
        # then put heads dimension before time.
        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.h, self.d).transpose(1, 2)

        q, k, v = map(split, (q, k, v))                        # [B, H, T, Dh]

        # Build [B, H, T, T] boolean mask (True = disallowed)
        mask_bool = None
        if mask is not None:
            if mask.dim() == 2:
                mask_bool = (
                    (mask == 0)
                    .bool()[None, None, :, :]
                    .expand(B, self.h, T, T)
                )
            elif mask.dim() == 3:
                mask_bool = (
                    (mask == 0)
                    .bool()
                    .unsqueeze(1)
                    .expand(B, self.h, T, T)
                )
            elif mask.dim() == 4:
                if mask.size(1) == 1:
                    mask_bool = (
                        (mask == 0).bool().expand(B, self.h, T, T)
                    )
                else:
                    mask_bool = (mask == 0).bool()

        # Manual scaled dot‑product attention for portability (MPS-safe)
        Dh = self.d
        scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)  # [B,H,T,T]
        if mask_bool is not None:
            scores = scores.masked_fill(mask_bool, float(-1e9))
        w = torch.softmax(scores, dim=-1)
        attn = w @ v  # [B,H,T,Dh]
        attn = self.drop(attn)

        # Concatenate heads back: [B, H, T, Dh] -> [B, T, Dm]
        y = (
            attn.transpose(1, 2)
            .contiguous()
            .view(B, T, Dm)
        )
        return self.out(y)