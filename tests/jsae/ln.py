# %%
import torch
import torch.nn as nn
from jaxtyping import Float
from typing import Any
from torch import Tensor
device = "cpu"
# %%

class LayerNorm(nn.Module):
    def __init__(self,normalized_shape, eps: float = 1e-5):
        """
        LayerNorm with optional length parameter

        length (Optional[int]): If the dimension of the LayerNorm. If not provided, assumed to be d_model
        """
        super().__init__()
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"], return_std: bool = False
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        
        x = x - x.mean(-1, keepdim=True)  # [batch, pos, length]
        scale: Float[torch.Tensor, "batch pos 1"] = (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        x = x / scale  # [batch, pos, length]
        
        out = x * self.weight + self.bias
        if return_std:
            return out, x, scale
        else:
            return out

# Test code to verify behavior matches nn.LayerNorm
# %%
# Create test input
batch_size = 2
seq_len = 3
d_model = 10
x = torch.randn(batch_size, seq_len, d_model)

# Initialize both layer norms
custom_ln = LayerNorm(d_model).to(device)
torch_ln = nn.LayerNorm(d_model).to(device)

# Copy weights to ensure same initialization
with torch.no_grad():
    torch_ln.weight.copy_(custom_ln.weight)
    torch_ln.bias.copy_(custom_ln.bias)

# Forward pass
custom_out = custom_ln(x)
torch_out = torch_ln(x)

# Verify outputs match
assert torch.allclose(custom_out, torch_out, rtol=1e-5, atol=1e-5), "Outputs don't match!"

# Test with return_std=True
custom_out, custom_resid, custom_std = custom_ln(x, return_std=True)
assert torch.allclose(custom_out, torch_out, rtol=1e-5, atol=1e-5), "Outputs don't match with return_std=True!"

print("All tests passed! Custom LayerNorm matches nn.LayerNorm behavior.")
# %%
