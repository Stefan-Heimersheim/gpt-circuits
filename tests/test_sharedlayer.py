# %%
%load_ext autoreload
%autoreload 2
# %%
import os
import torch
# Change current working directory to parent
if not os.getcwd().endswith("gpt-circuits"):
    os.chdir('..')
print(os.getcwd())


from models.sae.sharedlayer import SharedLayer
from config.sae.models import SAEConfig, SAEVariant
from config.sae.training import LossCoefficients
import json
from config.gpt.models import GPTConfig

gpt_config = GPTConfig(n_layer = 2, n_embd = 4)


saes = SharedLayer(
    trainable_layers=[0, 1, 2],
    config=SAEConfig(
        n_features=tuple(n for n in (8, 16, 24)),
        sae_variant=SAEVariant.TOPK,
        top_k=(10,10,10),
        shared_layers=True,
        gpt_config=gpt_config,
    ),
    loss_coefficients=None,
)
# %%
print(dict(saes['0'].named_parameters()))
# verify weight sharing works
for l in range(3):
    W_dec_temp = torch.randn_like(saes[str(l)].W_dec)
    saes[str(l)].W_dec.data = W_dec_temp.data
    feat_size = saes.config.n_features[l]
    assert torch.allclose(saes.W_dec[:feat_size, :].data, saes[str(l)].W_dec.data), f"Weight sharing failed for layer {l}"

for l in range(3):
    W_enc_temp = torch.randn_like(saes[str(l)].W_enc)
    saes[str(l)].W_enc.data = W_enc_temp.data
    feat_size = saes.config.n_features[l]
    assert torch.allclose(saes.W_enc[:, :feat_size].data, saes[str(l)].W_enc.data), f"Weight sharing failed for layer {l}"
    
for l in range(3):
    feat_size = saes.config.n_features[l]
    assert torch.allclose(saes.W_dec[:feat_size, :].data, saes[str(l)].W_dec[:feat_size, :].data), f"Weight sharing failed for layer {l}"
    assert torch.allclose(saes.W_enc[:, :feat_size].data, saes[str(l)].W_enc[:, :feat_size].data), f"Weight sharing failed for layer {l}"
# %%

# %%