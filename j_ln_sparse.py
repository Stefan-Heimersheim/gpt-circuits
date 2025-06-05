# %%
%load_ext autoreload
%autoreload 2

# %%

from j_ln import jacobian_fold_layernorm, egrad, DummySAE, MLP, LayerNorm, jacobian_exact, jacobian_full_block
import torch
from eindex import eindex
from dataclasses import dataclass
from torch import Tensor
from jaxtyping import Float, Int
import einops
# %

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 3
seq_len = 2
n_feat = 160
n_embd = 32

cache = {}

z = torch.randn(batch_size, seq_len, n_feat, device=device)

@dataclass
class GPTConfig:
    n_embd: int
    bias: bool = False

cfg = GPTConfig(n_embd=n_embd)


sae_mlpin = DummySAE(n_features=n_feat, n_embd=n_embd, k=5).to(device)
sae_mlpout = DummySAE(n_features=n_feat, n_embd=n_embd, k=5).to(device)
mlp = MLP(cfg).to(device)
ln = LayerNorm((n_embd,)).to(device)
ln.bias.data = torch.randn_like(ln.bias.data)
ln.weight.data = torch.randn_like(ln.weight.data)



def sandwich_mlp_block(z : Float[Tensor, "batch seq feat"],
             sae_mlpin : DummySAE,
             mlp : MLP,
             sae_mlpout : DummySAE,
             ln: LayerNorm,
             cache: dict) -> tuple[Float[Tensor, "batch seq feat"], Int[Tensor, "batch seq feat"]]:
    """
    takes sparse feature magnitudes and indices, and returns the sandwich product
    """
    
    # top_k_values, in_feat_idx = torch.topk(in_feat_mags, sae_mlpin.k, dim=-1, sorted=True)
    # #mask = in_feat_mags >= top_k_values[..., -1].unsqueeze(-1)
    # z = in_feat_mags #* mask.float()

    x = sae_mlpin.decode(z)

    x_hat, x_pre_hat, x_scale = ln(x, return_std=True)
    
    cache['ln_pre_y'] = x_pre_hat
    cache['ln_scale'] = x_scale
    
    h = mlp.W_in(x_hat)
    phi_h = mlp.gelu(h)
    y = mlp.W_out(phi_h)
    y = y + x
    
    cache['h'] = h
    cache['phi_h'] = phi_h

    mlp_act_grads = egrad(mlp.gelu, h)
    cache['mlp_act_grads'] = mlp_act_grads

    out_feat_mags = sae_mlpout.encode(y)
    
    top_k_values_out, out_feat_idx = torch.topk(out_feat_mags, sae_mlpout.k, dim=-1, sorted=True)
    mask_out = out_feat_mags >= top_k_values_out[..., -1].unsqueeze(-1)
    out_feat_mags_sparse = out_feat_mags * mask_out.float()

    return out_feat_mags_sparse, out_feat_mags, out_feat_idx


# %%

def jacobian_full_block(sae_mlpin, 
                        sae_mlpout, 
                        mlp, 
                        mlp_grads, 
                        gamma,
                        cache=cache):
    #w_enc_active = eindex(sae_mlpout.W_enc.T, out_idx, "[batch seq k2] d_e -> batch seq k2 d_e")
    #w_dec_active = eindex(sae_mlpin.W_dec.T, in_idx, "d_e [batch seq k1] -> batch seq d_e k1")
    w_enc_active = einops.repeat(sae_mlpout.W_enc.T, "d_e k2 -> b s d_e k2", b=batch_size, s=seq_len)
    w_dec_active = einops.repeat(sae_mlpin.W_dec.T, "d_e k1 -> b s d_e k1", b=batch_size, s=seq_len)
    
    j_after = einops.einsum(w_enc_active, mlp.W_out.weight, mlp_grads, mlp.W_in.weight, "b s k2 d_e_1, d_e_1 d_mlp, b s d_mlp, d_mlp d_e_2 -> b s k2 d_e_2")
    j_before = w_dec_active
    jacobian = jacobian_fold_layernorm(j_after, j_before, cache["ln_pre_y"], cache["ln_scale"], gamma)
    
    jacobian_skip = w_enc_active @ w_dec_active
    
    return jacobian + jacobian_skip


# out_feat_mags_sparse, in_feat_idx, out_feat_idx = sandwich_mlp_block(z, sae_mlpin, mlp, sae_mlpout, ln)

# jacobian_true = jacobian_exact(lambda z: sandwich_mlp_block(z, sae_mlpin, mlp, sae_mlpout, ln)[0], z)

# jacobian_true_dense = eindex(jacobian_true, out_feat_idx, in_feat_idx, "batch seq [batch seq k2] [batch seq k1] -> batch seq k2 k1")



#jacobian_guess_naive = jacobian_full_block(sae_mlpin, sae_mlpout, mlp, mlp_grads, ln, cache)
# %%
torch.manual_seed(42)
z  = torch.randn(batch_size, seq_len, n_feat, device=device)
z_top, z_idx = torch.topk(z, 5, dim=-1, sorted=True)
mask = z >= z_top[..., -1].unsqueeze(-1)
z_sparse = z * mask.float()

# %%
cache = {}

z_2_sparse, z_2, z_2_idx = sandwich_mlp_block(z_sparse, sae_mlpin, mlp, sae_mlpout, ln, cache)

true_cache = {}
jacobian_true = jacobian_exact(lambda z: sandwich_mlp_block(z, sae_mlpin, mlp, sae_mlpout, ln, true_cache)[0], z_sparse)
jacobian_true_dense = eindex(jacobian_true, z_2_idx, z_idx, "batch seq [batch seq k2] [batch seq k1] -> batch seq k2 k1")



# %%

mlp_grads = egrad(mlp.gelu, cache['h'])

jacobian_guess_naive = jacobian_full_block(sae_mlpin, sae_mlpout, mlp, mlp_grads, ln.weight, cache)
jacobian_guess_naive_dense = eindex(jacobian_guess_naive, z_2_idx, z_idx, "batch seq [batch seq k2] [batch seq k1] -> batch seq k2 k1")

torch.testing.assert_close(jacobian_guess_naive_dense, jacobian_true_dense)
print("Test dense naive jacobian passed")


# torch.testing.assert_close(jacobian_guess_sparse, jacobian_true_dense)
# print("Test sparse jacobian passed")
# %%
def jacobian_full_block_sparse(out_idx : Int[Tensor, "batch seq k2"],
                               in_idx : Int[Tensor, "batch seq k1"],
                        sae_mlpin, 
                        sae_mlpout, 
                        mlp, 
                        mlp_grads : Float[Tensor, "batch seq d_mlp"],
                        gamma : Float[Tensor, "d_model"],
                        ln_pre_y : Float[Tensor, "batch seq d_model"],
                        ln_scale : Float[Tensor, "batch seq 1"]):
    w_enc_active = eindex(sae_mlpout.W_enc.T, out_idx, "[batch seq k2] d_e -> batch seq k2 d_e")
    w_dec_active = eindex(sae_mlpin.W_dec.T, in_idx, "d_e [batch seq k1] -> batch seq d_e k1")
    #w_enc_active = einops.repeat(sae_mlpout.W_enc.T, "d_e k2 -> b s d_e k2", b=batch_size, s=seq_len)
    #w_dec_active = einops.repeat(sae_mlpin.W_dec.T, "d_e k1 -> b s d_e k1", b=batch_size, s=seq_len)
    
    j_after = einops.einsum(w_enc_active, mlp.W_out.weight, mlp_grads, mlp.W_in.weight, "b s k2 d_e_1, d_e_1 d_mlp, b s d_mlp, d_mlp d_e_2 -> b s k2 d_e_2")
    j_before = w_dec_active
    jacobian = jacobian_fold_layernorm(j_after, j_before, ln_pre_y, ln_scale, gamma)
    
    jacobian_skip = w_enc_active @ w_dec_active
    
    return jacobian + jacobian_skip
# %%

jacobian_guess_dense_fast = jacobian_full_block_sparse(z_2_idx, z_idx, sae_mlpin, sae_mlpout, mlp, mlp_grads, ln.weight, cache["ln_pre_y"], cache["ln_scale"])
torch.testing.assert_close(jacobian_guess_dense_fast, jacobian_true_dense)
print("Test dense fast jacobian passed")
# %%


# %%