from j_ln import egrad, DummySAE, MLP, LayerNorm, jacobian_exact, jacobian_full_block
import torch
from eindex import eindex
from dataclasses import dataclass
from torch import Tensor
from jaxtyping import Float, Int
import einops
import time
import numpy as np
from typing import Callable, Dict, Any
from opt_einsum import contract as opt_einsum


@torch.compile
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


def jacobian_full_block_sparse(out_idx : Int[Tensor, "batch seq k2"],
                               in_idx : Int[Tensor, "batch seq k1"],
                        sae_mlpin, 
                        sae_mlpout, 
                        mlp, 
                        mlp_grads : Float[Tensor, "batch seq d_mlp"],
                        gamma : Float[Tensor, "d_model"],
                        ln_pre_y : Float[Tensor, "batch seq d_model"],
                        ln_scale : Float[Tensor, "batch seq 1"]):
    # w_enc_active = eindex(sae_mlpout.W_enc.T, out_idx, "[batch seq k2] d_e -> batch seq k2 d_e")
    # w_dec_active = eindex(sae_mlpin.W_dec.T, in_idx, "d_e [batch seq k1] -> batch seq d_e k1")
    
    # print(f"Correct shapes: {w_enc_active.shape}, {w_dec_active.shape}")

    w_enc_active = sae_mlpout.W_enc.T[out_idx]
    w_dec_active = sae_mlpin.W_dec.T[:, in_idx]
    w_dec_active = einops.rearrange(w_dec_active, "d_e b s k -> b s d_e k")


    
    #w_enc_active = einops.repeat(sae_mlpout.W_enc.T, "d_e k2 -> b s d_e k2", b=batch_size, s=seq_len)
    #w_dec_active = einops.repeat(sae_mlpin.W_dec.T, "d_e k1 -> b s d_e k1", b=batch_size, s=seq_len)
    # "b s k2 d_e_1, d_e_1 d_mlp, b s d_mlp, d_mlp d_e_2 -> b s k2 d_e_2"
    #j_after = opt_einsum("bsod,dm,bsm,di->bsoi",w_enc_active, mlp.W_out.weight, mlp_grads, mlp.W_in.weight)
    j_after = opt_einsum("bsaw,wc,bsc,cd->bsad",w_enc_active, mlp.W_out.weight, mlp_grads, mlp.W_in.weight)
    #j_after = einops.einsum(w_enc_active, mlp.W_out.weight, mlp_grads, mlp.W_in.weight, "b s k2 d_e_1, d_e_1 d_mlp, b s d_mlp, d_mlp d_e_2 -> b s k2 d_e_2")
    j_before = w_dec_active
    jacobian = jacobian_fold_layernorm(j_after, j_before, ln_pre_y, ln_scale, gamma)
    
    jacobian_skip = w_enc_active @ w_dec_active
    
    return jacobian + jacobian_skip

def jacobian_full_block(sae_mlpin, 
                        sae_mlpout, 
                        mlp, 
                        mlp_grads, 
                        gamma,
                        cache):
    #w_enc_active = eindex(sae_mlpout.W_enc.T, out_idx, "[batch seq k2] d_e -> batch seq k2 d_e")
    #w_dec_active = eindex(sae_mlpin.W_dec.T, in_idx, "d_e [batch seq k1] -> batch seq d_e k1")
    
    batch_size, seq_len = mlp_grads.shape[:2]
    
    w_enc_active = einops.repeat(sae_mlpout.W_enc.T, "d_e k2 -> b s d_e k2", b=batch_size, s=seq_len)
    w_dec_active = einops.repeat(sae_mlpin.W_dec.T, "d_e k1 -> b s d_e k1", b=batch_size, s=seq_len)
    
    j_after = einops.einsum(w_enc_active, mlp.W_out.weight, mlp_grads, mlp.W_in.weight, "b s k2 d_e_1, d_e_1 d_mlp, b s d_mlp, d_mlp d_e_2 -> b s k2 d_e_2")
    j_before = w_dec_active
    jacobian = jacobian_fold_layernorm(j_after, j_before, cache["ln_pre_y"], cache["ln_scale"], gamma)
    
    jacobian_skip = w_enc_active @ w_dec_active
    
    return jacobian + jacobian_skip


@torch.compile(fullgraph=True, mode="max-autotune")
def jacobian_opt_compiled(
    # Inputs related to indexing and sparse weights
    w_enc_active: Float[Tensor, "batch seq k2 d_model"],
    w_dec_active: Float[Tensor, "batch seq d_model k1"],
    j_after: Float[Tensor, "batch seq k2 d_model"],
    # out_idx: Int[Tensor, "batch seq k2"],
    # in_idx: Int[Tensor, "batch seq k1"],
    # # SAE weights
    # sae_mlpout_W_enc_T_param: Float[Tensor, "n_feat_out d_model"], # Expects W_enc.T
    # sae_mlpin_W_dec_T_param: Float[Tensor, "d_model n_feat_in"],  # Expects W_dec.T
    # # MLP weights
    # mlp_W_out_weight: Float[Tensor, "d_model d_mlp"],
    # mlp_W_in_weight: Float[Tensor, "d_mlp d_model"],
    # # Activations and gradients
    # mlp_grads: Float[Tensor, "batch seq d_mlp"],
    # LayerNorm parameters and cached values
    ln_gamma: Float[Tensor, "d_model"],
    ln_pre_y: Float[Tensor, "batch seq d_model"],
    ln_scale: Float[Tensor, "batch seq 1"],
    # Constants
    D_model: int
) -> Float[Tensor, "batch seq k2 k1"]:
    """
    Optimized version using torch.compile, assuming specific input tensor forms.
    """

    # 1. Gather active weights for SAEs
    # Based on user confirmation: sae_mlpout.W_enc.T[out_idx] is the target.
    # So, sae_mlpout_W_enc_T_param is sae_mlpout.W_enc.T
  
    # Expected shape: (batch, seq, d_model, k1)

    # j_after_inlined = opt_einsum("bskd,dm,bsm,mi->bski",
    #                              w_enc_active,
    #                              mlp_W_out_weight,
    #                              mlp_grads,
    #                              mlp_W_in_weight)
    
    # 3. Inline jacobian_fold_layernorm logic
    id_scale_scalar_squeezed = (1.0 / ln_scale).squeeze(-1)

    id_term = torch.einsum("bsKi,i,bsir->bsKr",
                           j_after, ln_gamma, w_dec_active)
    #id_term = opt_einsum("bsKi,i,bsir->bsKr",
     #                    j_after, ln_gamma, w_dec_active)
    id_component = id_scale_scalar_squeezed.unsqueeze(-1).unsqueeze(-1) * id_term

    term1_off_ones = torch.einsum("bsKi,i->bsK", j_after, ln_gamma)
    term2_off_ones = w_dec_active.sum(dim=-2) 
    off_ones = torch.einsum("bsK,bsr->bsKr", term1_off_ones, term2_off_ones)

    gamma_ln_pre_y = ln_gamma * ln_pre_y
    off_y_left = torch.einsum("bsKi,bsi->bsK", j_after, gamma_ln_pre_y)
    off_y_right = torch.einsum("bsi,bsir->bsr", ln_pre_y, w_dec_active)
    off_y = torch.einsum("bsK,bsr->bsKr", off_y_left, off_y_right)
    
    off_scale_factor = (id_scale_scalar_squeezed / D_model).unsqueeze(-1).unsqueeze(-1)
    off_component = off_scale_factor * (off_ones + off_y)

    jacobian_mlp_path = id_component - off_component

    # 4. Skip connection part
    #jacobian_skip = torch.matmul(w_enc_active, w_dec_active)
    jacobian_skip = w_enc_active @ w_dec_active


    final_jacobian = jacobian_mlp_path + jacobian_skip
    return final_jacobian


def jacobian_block_ln(out_idx: Int[Tensor, "batch seq k2"],
                 in_idx: Int[Tensor, "batch seq k1"],
                 sae_mlpin: DummySAE,
                 sae_mlpout: DummySAE, 
                 mlp: MLP,
                 mlp_grads: Float[Tensor, "batch seq d_mlp"],
                 ln_weight: Float[Tensor, "d_model"],
                 ln_pre_y: Float[Tensor, "batch seq d_model"],
                 ln_scale: Float[Tensor, "batch seq 1"]
                ) -> Float[Tensor, "batch seq k2 k1"]:
    
    # Based on user confirmation for jacobian_full_block_sparse:
    # W_enc is (d_model, n_feat), so W_enc.T is (n_feat, d_model)
    # W_dec is (n_feat, d_model), so W_dec.T is (d_model, n_feat)

    mlp_W_out_weight = mlp.W_out.weight
    mlp_W_in_weight = mlp.W_in.weight
    
    D_model = ln_weight.shape[0]
    
    w_enc_active = sae_mlpout.W_enc.T[out_idx]
    # Expected shape: (batch, seq, k2, d_model)

    # sae_mlpin_W_dec_T_param is sae_mlpin.W_dec.T
    # sae_mlpin_W_dec_T_param[:, in_idx] gives (d_model, batch, seq, k1)
    w_dec_active = sae_mlpin.W_dec.T[:, in_idx].permute(1, 2, 0, 3).contiguous()
    k = sae_mlpin.k
    
    j_after = opt_einsum("bskd,dm,bsm,mi->bski",
                                w_enc_active,
                                mlp_W_out_weight,
                                mlp_grads,
                                mlp_W_in_weight)

    return jacobian_opt_compiled(
        w_enc_active,
        w_dec_active,
        j_after,
        ln_weight, ln_pre_y, ln_scale,
        D_model
    ).abs_().sum() / (k ** 2)
