# %%

import os
import sys

while not os.getcwd().endswith("gpt-circuits"):
    os.chdir("..")
print(os.getcwd())


MAIN = __name__ == "__main__"

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Callable, Dict, Tuple
from jaxtyping import Float, Int
from torch import Tensor
from dataclasses import dataclass
import time # For timing comparison
from typing import Union
from torch import Size

_shape_t = Union[int, list[int], Size]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}") # Already printed

# %% Model Definitions (reuse MLP, DummySAE)
@dataclass
class GPTConfig:
    n_embd: int
    bias: bool = False

class MLP(nn.Module): # (definition unchanged)
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.W_in = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.W_out = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        x = self.gelu(self.W_in(x))
        x = self.W_out(x)
        return x

class DummySAE(nn.Module): # (definition unchanged)
    def __init__(self, n_embd: int, n_features: int, k : int = 5):
        super().__init__()
        self.n_features = n_features
        self.W_dec = nn.Parameter(torch.randn((n_features, n_embd), device=device) / n_features**0.5)
        self.W_enc = nn.Parameter(torch.randn((n_embd, n_features), device=device) / n_embd**0.5)
        self.b_enc = nn.Parameter(torch.randn(n_features, device=device))
        self.b_dec = nn.Parameter(torch.randn(n_embd, device=device))
        self.k = k

    def encode(self, x: Float[Tensor, "... n_embd"]) -> Float[Tensor, "... n_features"]:
        return x @ self.W_enc + self.b_enc

    def decode(self, x: Float[Tensor, "... n_features"]) -> Float[Tensor, "... n_embd"]:
        return x @ self.W_dec + self.b_dec

    def decode_sparse(self,
                      feat_values: Float[Tensor, "batch seq k"],
                      feat_indices: Int[Tensor, "batch seq k"]
                      ) -> Float[Tensor, "batch seq n_embd"]:
        b, s, k = feat_indices.shape
        n_embd = self.W_dec.shape[1]
        W_dec_active = self.W_dec[feat_indices] # Shape: (b, s, k, n_embd)
        decoded = einops.einsum(feat_values, W_dec_active, "b s k, b s k e -> b s e")
        decoded += self.b_dec
        return decoded

# %% Modified Sandwich Function (reuse sandwich_sparse_input)
# activations_store: Dict[str, Tensor] = {} # (definition unchanged)

# def sandwich_sparse_input(
#     in_feat_values : Float[Tensor, "batch seq k_in"],
#     in_feat_indices: Int[Tensor, "batch seq k_in"],
#     sae_mlpin : DummySAE, mlp : MLP, sae_mlpout : DummySAE,
#     normalize : Callable[[Tensor], Tensor]
# ) -> Float[Tensor, "batch seq feat_out"]:
#     normalized_values = normalize(in_feat_values)
#     resid_mid = sae_mlpin.decode_sparse(normalized_values, in_feat_indices)
#     mlp_post = mlp(resid_mid)
#     resid_post = mlp_post + resid_mid
#     resid_post_featmag = sae_mlpout.encode(resid_post)
#     return resid_post_featmag

# # %% Hook Function & Derivative Helper (reuse mlp_gelu_hook_fn, egrad)
# def mlp_gelu_hook_fn(module, inputs, outputs): # (definition unchanged)
#     pre_actfn = inputs[0]
#     pre_actfn_copy = pre_actfn.detach().requires_grad_(True)
#     with torch.enable_grad():
#         recomputed_post_actfn = F.gelu(pre_actfn_copy, approximate="tanh")
#         grad_of_actfn = torch.autograd.grad(
#             outputs=recomputed_post_actfn, inputs=pre_actfn_copy,
#             grad_outputs=torch.ones_like(recomputed_post_actfn),
#             retain_graph=False, create_graph=False
#         )[0]
#     activations_store["mlp_gelu_grad"] = grad_of_actfn
#     return outputs

# def egrad(func: Callable, x: Tensor) -> Tensor: # (definition unchanged)
#     x_detached = x.detach().requires_grad_(True)
#     with torch.enable_grad():
#         y = func(x_detached)
#         grads = torch.autograd.grad(outputs=y, inputs=x_detached,
#                                     grad_outputs=torch.ones_like(y),
#                                     create_graph=False, retain_graph=False)[0]
#     return grads

# %% Main Execution Block (Identical to previous, just uses corrected function)

# --- Hyperparameters ---
# Use the dimensions from the error report for consistency
if MAIN:
    n_embd = 32      # Adjusted based on typical d_mlp = 4*n_embd and previous examples
    n_feat = 8     # From dense jacobian shape
    k_in = 5       # From sparse jacobian shape
    k_out = 5      # From sparse jacobian shape
    batch_size = 3  # From sparse jacobian shape
    seq_len = 7     # From sparse jacobian shape
    d_mlp = 4 * n_embd # Make sure this aligns if possible, or use a known d_mlp

    print(f"Using dimensions: B={batch_size}, S={seq_len}, N_feat={n_feat}, K_in={k_in}, K_out={k_out}, N_embd={n_embd}")

    # --- Instantiate Models ---
    # Ensure consistent dimensions
    if d_mlp != 4 * n_embd:
        print(f"Warning: d_mlp ({d_mlp}) not 4*n_embd ({4*n_embd}). Adjusting n_embd based on d_mlp from hook if needed.")
        # This shouldn't happen if mlp_gelu_grads shape is consistent, but good check.


    gpt_config = GPTConfig(n_embd=n_embd)
    mlp = MLP(gpt_config).to(device)
    # Ensure SAEs use n_feat
    sae_mlpin = DummySAE(n_embd, n_feat).to(device)
    sae_mlpout = DummySAE(n_embd, n_feat).to(device)



class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, residual: Float[Tensor, "batch posn d_model"], return_std: bool = False) -> Float[Tensor, "batch posn d_model"]:
        residual_mean = residual.mean(dim=-1, keepdim=True)
        residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.eps).sqrt()

        residual = (residual - residual_mean) / residual_std
        out = residual * self.weight + self.bias
        if return_std:
            return out, residual, residual_std
        else:
            return out
      
if MAIN:
    ln = LayerNorm(n_embd).to(device)






    
# def jacobian_slow(
#     j_left : Float[Tensor, "d_left d_model"],
#     j_right : Float[Tensor, "d_model d_right"],
#     ln_y : Float[Tensor, "b s d_model"],
#     ln_scale : Float[Tensor, "b s 1"],
# ) -> Float[Tensor, "b s d_left d_right"]:




# %%
# cache = {}
# z = torch.randn(batch_size, seq_len, n_feat, device=device)

# def sandwich(z, sae_mlpin=sae_mlpin, mlp=mlp, ln=ln, cache=cache):
#     x = sae_mlpin.decode(z)
#     ln_y, ln_pre_y, ln_scale = ln(x, return_std=True)
#     cache["ln_y"] = ln_y
#     cache["ln_pre_y"] = ln_pre_y
#     cache["ln_scale"] = ln_scale
#     h = mlp.W_in(ln_y)
#     return h

# jacobian_autograd = torch.autograd.functional.jacobian(
#     func=sandwich,
#     inputs=z,
#     create_graph=False,
#     vectorize=True # Important for batch processing!
# )

# jacobian_diag = einops.einsum(jacobian_autograd, "b s din b s dout -> b s dout din")

# jacobian_guess = jacobian_fold_layernorm(
#     j_left=torch.eye(n_embd, device=device),# sae_mlpin.W_dec,
#     j_right=torch.eye(n_embd, device=device),# mlp.W_in.weight.T,
#     ln_pre_y=cache["ln_pre_y"],
#     ln_scale=cache["ln_scale"]
# )

# torch.testing.assert_close(jacobian_guess, jacobian_diag)
# %%        

def jacobian_ln_naive(
    ln_pre_y : Float[Tensor, "b s d_model"],
    ln_scale : Float[Tensor, "b s 1"],
    gamma : Float[Tensor, "d_model"],
) -> Float[Tensor, "b s d_model d_model"]:
    
    d_model = ln_pre_y.shape[-1]
    
    id_component = torch.eye(d_model, device=ln_pre_y.device) * einops.rearrange(1 / ln_scale, "b s 1 -> b s 1 1")
    y_s = einops.einsum(ln_pre_y, ln_pre_y, "b s d1, b s d2 -> b s d1 d2")
    ones = torch.ones_like(y_s)
    off_comp_scale = einops.rearrange(1 / ln_scale, "b s 1 -> b s 1 1") / d_model
    off_component = off_comp_scale * (ones + y_s)
    
    jacobian = id_component - off_component
    gamma_jacobian = einops.einsum(gamma, jacobian, "dout, b s dout din -> b s dout din")
    return gamma_jacobian
    

def jacobian_exact(f, x : Float[Tensor, "b s d"]):
    jacobian =  torch.autograd.functional.jacobian(
        func=f,
        inputs=x,
        create_graph=False,
        vectorize=True
    )
    if isinstance(jacobian, tuple):
        print(f"jacobian is a tuple: {jacobian}. Taking first element.")
        jacobian = jacobian[0]
    
    
    jacobian = einops.einsum(jacobian, "b s dout b s din -> b s dout din")
    return jacobian


if MAIN:

    batch_size = 2
    seq_len = 3
    n_embd = 5

    ln = LayerNorm(n_embd).to(device)
    ln.weight.data = torch.randn_like(ln.weight.data)
    ln.bias.data = torch.randn_like(ln.bias.data)
    x = torch.randn(batch_size, seq_len, n_embd, device=device)
    ln_y, ln_pre_y,ln_scale = ln(x, return_std=True)
    
    jacobian_ln_guess = jacobian_ln_naive(ln_pre_y,ln_scale, ln.weight)
    
    jacobian_ln = jacobian_exact(ln.forward, x)

    torch.testing.assert_close(jacobian_ln_guess, jacobian_ln)
    print("Naive jacobian test passed")

        
    

# %%


def jacobian_fold_layernorm(
    j_after: Float[Tensor, "[b s] d_out d_model"], # sae_residmid.W_dec
    j_before: Float[Tensor, "[b s] d_model d_in"],
    ln_pre_y: Float[Tensor, "b s d_model"],
    ln_scale: Float[Tensor, "b s 1"],
    gamma: Float[Tensor, "d_model"],
) -> Float[Tensor, "b s d_out d_in"]:
    """
    Computes the Jacobian of  mlp.W_in(LN(sae.W_dec(x)))
    Since LayerNorm Jacobian is a rank-1 perturbation from the identity,
    we can fold the Jacobian of the LayerNorm into the Jacobian before (sae.W_dec)
    and after (mlp.W_out) for efficency.
    
    J^{LN} = scale^{-1} * I - (N * scale)^{-1} (11^T + y y^T)
    where N is the batch size, y is the output of the LayerNorm, and 1 is a vector of ones.
    """
    
    #d_left = d_after
    #d_right = d_before
    batch_size, seq_len = ln_pre_y.shape[:2]
    assert j_before.shape[-2] == j_after.shape[-1]
    
    if j_after.ndim == 2:  # shape (l, d)
        j_after = einops.repeat(j_after, "l d -> b s l d", b=batch_size, s=seq_len)
        
    if j_before.ndim == 2:  # shape (d, r)
        j_before = einops.repeat(j_before, "d r -> b s d r", b=batch_size, s=seq_len)
    
    
    d_out, d_model = j_after.shape[-2], j_after.shape[-1]
    d_model, d_in = j_before.shape[-2], j_before.shape[-1]
    # 1. Index W_dec according to topk_indices_residmid
    # Original W_dec shape: (feat_size, n_embd)
    # Indices shape: (b, seq, k1) -> selecting along the first dimension (feat_size)
    id_scale = (1 / ln_scale).squeeze(-1)
    id_component = einops.einsum(id_scale, j_after, gamma, j_before, "b s, b s l d, d, b s d r -> b s l r")

    ones = torch.ones((d_model), device=j_after.device)
    
    off_ones = einops.einsum( j_after @ gamma, j_before.sum(dim=-2), "b s l, b s r -> b s l r") # (d_out, d_in) outer product
    
    off_y_left = einops.einsum(j_after, (gamma * ln_pre_y), "b s l d, b s d -> b s l")
    off_y_right = einops.einsum(ln_pre_y, j_before, "b s d, b s d r -> b s r")
    
    off_y = einops.einsum(off_y_left, off_y_right, "b s l, b s r -> b s l r")
    
    off_scale = einops.rearrange(id_scale / d_model, "b s -> b s 1 1")
    off_component = off_scale * (off_ones + off_y) # (b, s, d_out, d_in)
    
    jacobian = id_component - off_component
    assert jacobian.shape == (batch_size, seq_len, d_out, d_in), f"Expected {(batch_size, seq_len, d_out, d_in)}, got {jacobian.shape}"
    return jacobian
# %%
if MAIN:
        
    n_embd = 5
    ln = LayerNorm(n_embd).to(device)
    ln.weight.data = torch.randn_like(ln.weight.data)
    ln.bias.data = torch.randn_like(ln.bias.data)
    z = torch.randn(batch_size, seq_len, n_embd, device=device)

# %%

if MAIN:

    cache = {}

    def f(z, cache=cache):
        #x = sae_mlpin.decode(z)
        ln_y, ln_pre_y, ln_scale = ln(z, return_std=True)
        cache["ln_y"] = ln_y
        cache["ln_pre_y"] = ln_pre_y
        cache["ln_scale"] = ln_scale
        return ln_y

    _ = f(z)


    jacobian_true = jacobian_exact(f, z)


    jacobian_guess2 = jacobian_fold_layernorm(
        j_after=torch.eye(n_embd, device=device),# mlp.W_in.weight.T,
        j_before=torch.eye(n_embd, device=device),# sae_mlpin.W_dec,
        ln_pre_y=cache["ln_pre_y"],
        ln_scale=cache["ln_scale"],
        gamma=ln.weight
    )



    torch.testing.assert_close(jacobian_guess2, jacobian_true)
    print("jacobian ln fold passed")
# %%

if MAIN:
    n_embd = 32
    ln = LayerNorm(n_embd).to(device)
    ln.weight.data = torch.randn_like(ln.weight.data)
    ln.bias.data = torch.randn_like(ln.bias.data)
    z = torch.randn(batch_size, seq_len, n_feat, device=device)

    cache = {}

    def g(z, cache=cache):
        x = sae_mlpin.decode(z)
        ln_y, ln_pre_y, ln_scale = ln(x, return_std=True)
        cache["ln_y"] = ln_y
        cache["ln_pre_y"] = ln_pre_y
        cache["ln_scale"] = ln_scale
        return ln_y

    _ = g(z)


    jacobian_true = jacobian_exact(g, z)


    jacobian_guess2 = jacobian_fold_layernorm(
        j_after=torch.eye(n_embd, device=device),
        j_before=sae_mlpin.W_dec.T,
        ln_pre_y=cache["ln_pre_y"],
        ln_scale=cache["ln_scale"],
        gamma=ln.weight
    )

    jacobian_ln = jacobian_ln_naive(
        ln_pre_y=cache["ln_pre_y"],
        ln_scale=cache["ln_scale"],
        gamma=ln.weight
    )

    jacobian_guess3 = einops.einsum(jacobian_ln, sae_mlpin.W_dec.T, "b s d_out d_embd, d_embd d_feat -> b s d_out d_feat")


    torch.testing.assert_close(jacobian_guess2, jacobian_true)
    torch.testing.assert_close(jacobian_guess3, jacobian_true)
    print("Test sae_mlpin and sae_mlpout passed")
# %%

if MAIN:
    n_embd = 32
    ln = LayerNorm(n_embd).to(device)
    ln.weight.data = torch.randn_like(ln.weight.data)
    ln.bias.data = torch.randn_like(ln.bias.data)
    z = torch.randn(batch_size, seq_len, n_feat, device=device)

    cache = {}

    def g(z, cache=cache):
        x = sae_mlpin.decode(z)
        ln_y, ln_pre_y, ln_scale = ln(x, return_std=True)
        cache["ln_y"] = ln_y
        cache["ln_pre_y"] = ln_pre_y
        cache["ln_scale"] = ln_scale
        return ln_y
        # h = mlp.W_in(ln_y)
        # return h

    _ = g(z)


    jacobian_true = jacobian_exact(g, z)


    jacobian_guess2 = jacobian_fold_layernorm(
        j_after=torch.eye(n_embd, device=device),
        j_before=sae_mlpin.W_dec.T,# mlp.W_in.weight.T,
        ln_pre_y=cache["ln_pre_y"],
        ln_scale=cache["ln_scale"],
        gamma=ln.weight
    )

    jacobian_ln = jacobian_ln_naive(
        ln_pre_y=cache["ln_pre_y"],
        ln_scale=cache["ln_scale"],
        gamma=ln.weight
    )

    jacobian_guess3 = einops.einsum(jacobian_ln, sae_mlpin.W_dec.T, "b s d_out d_embd, d_embd d_feat -> b s d_out d_feat")


    torch.testing.assert_close(jacobian_guess2, jacobian_true)
    torch.testing.assert_close(jacobian_guess3, jacobian_true)
    print("Test sae_mlpin and sae_mlpout passed")
# %%
if MAIN:
    n_embd = 32
    ln = LayerNorm(n_embd).to(device)
    ln.weight.data = torch.randn_like(ln.weight.data)
    ln.bias.data = torch.randn_like(ln.bias.data)
    z = torch.randn(batch_size, seq_len, n_feat, device=device)

    cache = {}

    def mlp_ln_sae(z, cache=cache):
        x = sae_mlpin.decode(z)
        ln_y, ln_pre_y, ln_scale = ln(x, return_std=True)
        cache["ln_y"] = ln_y
        cache["ln_pre_y"] = ln_pre_y
        cache["ln_scale"] = ln_scale
        h = mlp.W_in(ln_y)
        return h

    _ = mlp_ln_sae(z)


    jacobian_true = jacobian_exact(mlp_ln_sae, z)


    jacobian_guess2 = jacobian_fold_layernorm(
        j_after=mlp.W_in.weight, #should be mlp.W_in.weight.T but Linear stores weights backwards
        j_before=sae_mlpin.W_dec.T,# mlp.W_in.weight.T,
        ln_pre_y=cache["ln_pre_y"],
        ln_scale=cache["ln_scale"],
        gamma=ln.weight
    )

    jacobian_ln = jacobian_ln_naive(
        ln_pre_y=cache["ln_pre_y"],
        ln_scale=cache["ln_scale"],
        gamma=ln.weight
    )

    jacobian_guess3 = einops.einsum(mlp.W_in.weight.T, jacobian_ln, sae_mlpin.W_dec.T, "d_embd2 d_mlp, b s d_embd2 d_embd1, d_embd1 d_feat -> b s d_mlp d_feat")

    torch.testing.assert_close(jacobian_guess2, jacobian_true)
    torch.testing.assert_close(jacobian_guess3, jacobian_true)
    print("Test j_after and j_before passed")
# %%

cache = {}

def jacobian_full_block(sae_mlpin, 
                    sae_mlpout, 
                    mlp, 
                    mlp_grads, 
                    cache=cache):
    j_after = einops.einsum(sae_mlpout.W_enc.T, mlp.W_out.weight, mlp_grads, mlp.W_in.weight, "k2 d_e_1, d_e_1 d_mlp, b s d_mlp, d_mlp d_e_2 -> b s k2 d_e_2")
    j_before = sae_mlpin.W_dec.T
    jacobian = jacobian_fold_layernorm(j_after, j_before, cache["ln_pre_y"], cache["ln_scale"], ln.weight)
    return jacobian


def egrad(func: Callable, x : Tensor) -> Tensor:
    with torch.enable_grad():
        if not x.requires_grad:
            x.requires_grad = True
        f_x = func(x)  # [batch, pos, d_mlp]
        grad_f_x = torch.autograd.grad(
            outputs=f_x, inputs=x,
            grad_outputs=torch.ones_like(f_x), retain_graph=True
        )[0]
    return grad_f_x
    
    # with torch.enable_grad():
    #     input = input.detach().clone()
    #     input.requires_grad = True
        
    #     output = func(input)
    #     grads = torch.autograd.grad(outputs=output, inputs=input,
    #                                 grad_outputs=torch.ones_like(output),
    #                                 create_graph=False, retain_graph=True, allow_unused=True)[0]
    # return grads


if MAIN:
    cache = {}

    def full_block(z):
        x = sae_mlpin.decode(z)
        ln_y, ln_pre_y, ln_scale = ln(x, return_std=True)
        h = mlp.W_in(ln_y)
        phi_h = mlp.gelu(h)
        y = mlp.W_out(phi_h)
        z_2 = sae_mlpout.encode(y)
        
        cache["ln_y"] = ln_y
        cache["ln_pre_y"] = ln_pre_y
        cache["ln_scale"] = ln_scale
        cache["h"] = h
        cache["phi_h"] = phi_h
        cache["y"] = y
        cache["z_2"] = z_2
        
        return z_2

    _ = full_block(z)




    mlp_grads = egrad(mlp.gelu, cache["h"])


    jacobian_true = jacobian_exact(full_block, z)
    jacobian_guess = jacobian_full_block(sae_mlpin, sae_mlpout, mlp, mlp_grads, cache)

    torch.testing.assert_close(jacobian_guess, jacobian_true)
    print("Test full block passed")
# %%


def jacobian_full_block(sae_mlpin, 
                       sae_mlpout, 
                       mlp, 
                       mlp_grads, 
                       ln,
                       cache=cache):
    j_after = einops.einsum(sae_mlpout.W_enc.T, mlp.W_out.weight, mlp_grads, mlp.W_in.weight, "k2 d_e_1, d_e_1 d_mlp, b s d_mlp, d_mlp d_e_2 -> b s k2 d_e_2")
    j_before = sae_mlpin.W_dec.T
    jacobian = jacobian_fold_layernorm(j_after, j_before, cache["ln_pre_y"], cache["ln_scale"], ln.weight)
    
    jacobian_skip = sae_mlpout.W_enc.T @ sae_mlpin.W_dec.T
    
    return jacobian + jacobian_skip



if MAIN:

    cache = {}

    def full_block_with_skip(z):
        x = sae_mlpin.decode(z)
        
        ln_y, ln_pre_y, ln_scale = ln(x, return_std=True)
        h = mlp.W_in(ln_y)
        phi_h = mlp.gelu(h)
        y = mlp.W_out(phi_h)
        y = y + x
        z_2 = sae_mlpout.encode(y)
        
        cache["ln_y"] = ln_y
        cache["ln_pre_y"] = ln_pre_y
        cache["ln_scale"] = ln_scale
        cache["h"] = h
        cache["phi_h"] = phi_h
        
        return z_2

    _ = full_block_with_skip(z)
    
    

    mlp_grads = egrad(mlp.gelu, cache["h"])


    jacobian_true = jacobian_exact(full_block_with_skip, z)
    jacobian_guess = jacobian_full_block(sae_mlpin, sae_mlpout, mlp, mlp_grads, ln, cache)

    torch.testing.assert_close(jacobian_guess, jacobian_true)
    print("Test full block with skip passed")

# %%
