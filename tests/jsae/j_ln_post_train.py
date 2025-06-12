import os

while not os.getcwd().endswith("gpt-circuits"):
    os.chdir("..")
print(os.getcwd())

# %%

from config.sae.training import options
from models.jsaeblockparsified import JBlockSparsifiedGPT
from config.gpt.models import NormalizationStrategy
from safetensors.torch import load_model
from models.factorysparsified import FactorySparsified
from david.utils import generate_with_saes
from data.tokenizers import ASCIITokenizer
import torch
from config.sae.models import HookPoint

from jaxtyping import Float, Int
from torch import Tensor
import einops

from eindex import eindex

from utils.jsae import jacobian_mlp_block_ln
# %%

config = options['jsae.mlp_ln.shk_64x4']

gpt_mlp = FactorySparsified.load("checkpoints_ln/jln.shk_64x4.sp-1e-1", device="cuda")

generate_with_saes(gpt_mlp, ASCIITokenizer(), "Today I thought,", activations_to_patch=gpt_mlp.saes.keys())

# %%
torch.manual_seed(42)
z = torch.randn(3,2,64)
sparse_z, idx_in = gpt_mlp.saes[f'{1}_{HookPoint.RESID_MID.value}'].encode(z, return_topk_indices=True)

layer_idx = 1

def sandwich(sparse_z, layer_idx = layer_idx):
    resid_mid = gpt_mlp.saes[f'{layer_idx}_{HookPoint.RESID_MID.value}'].decode(sparse_z)
    block = gpt_mlp.gpt.transformer.h[layer_idx]
    mlp = block.mlp
    resid_mid_normed, ln_pre_y, ln_scale = block.ln_2(resid_mid, return_std=True)
    
    h = mlp.c_fc(resid_mid_normed)
    phi_h = mlp.gelu(h)
    mlp_out = mlp.c_proj(phi_h)

    resid_post = resid_mid + mlp_out
    #resid_post = resid_mid + block.mlp(resid_mid_normed)
    z_post, out_idx = gpt_mlp.saes[f'{layer_idx}_{HookPoint.RESID_POST.value}'].encode(resid_post, return_topk_indices=True)
    aux = {'resid_mid_normed': resid_mid_normed, 
           'ln_pre_y': ln_pre_y, 
           'ln_scale': ln_scale, 
           'out_idx': out_idx,
           'h': h,
           'phi_h': phi_h,
           'mlp_out': mlp_out}
    return z_post, aux


sparse_z_out, aux = sandwich(sparse_z, layer_idx)


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

j_exact = jacobian_exact(lambda z: sandwich(z, layer_idx)[0], sparse_z)
j_exact_sparse = eindex(j_exact, aux['out_idx'], idx_in, "b s [b s k2] [b s k1] -> b s k2 k1")


def egrad(func, x):
    with torch.enable_grad():
        if not x.requires_grad:
            x.requires_grad = True
        f_x = func(x)  # [batch, pos, d_mlp]
        grad_f_x = torch.autograd.grad(
            outputs=f_x, inputs=x,
            grad_outputs=torch.ones_like(f_x), retain_graph=True
        )[0]
    return grad_f_x

# Compute MLP gradients using autograd
mlp_grads = egrad(gpt_mlp.gpt.transformer.h[layer_idx].mlp.gelu, aux['h'])



# %%

j_guess = jacobian_mlp_block_ln(
    sae_residmid=gpt_mlp.saes[f'{layer_idx}_{HookPoint.RESID_MID.value}'],
    sae_residpost=gpt_mlp.saes[f'{layer_idx}_{HookPoint.RESID_POST.value}'],
    mlp=gpt_mlp.gpt.transformer.h[layer_idx].mlp,
    topk_indices_residmid=idx_in,
    topk_indices_residpost=aux['out_idx'],
    mlp_grads = mlp_grads,
    ln_weight=gpt_mlp.gpt.transformer.h[1].ln_2.weight,
    ln_pre_y=aux['ln_pre_y'],
    ln_scale=aux['ln_scale'],
    return_scalar=False
)

torch.testing.assert_close(j_guess, j_exact_sparse)
# Calculate absolute and relative errors
abs_error = (j_guess - j_exact_sparse).abs()
rel_error = abs_error / (j_exact_sparse.abs())

print(f"Max absolute error: {abs_error.max().item():.6e}")
print(f"Mean absolute error: {abs_error.mean().item():.6e}")
print(f"Max relative error: {rel_error.max().item():.6e}")
print(f"Mean relative error: {rel_error.mean().item():.6e}")

print("Jacobian LN post train passed")
# %%