# %%
# %load_ext autoreload
# %autoreload 2
# %%
import os

# Change current working directory to parent
os.chdir('..')
print(os.getcwd())


from models.sae.sharedlayer import SharedLayer
from config.sae.models import SAEConfig, SAEVariant
from config.sae.training import LossCoefficients

loss_coefficients = LossCoefficients()

saes = SharedLayer(
    trainable_layers=[0, 1, 2],
    config=SAEConfig(
        n_features=tuple(64 * n for n in (8, 16, 24)),
        sae_variant=SAEVariant.TOPK,
        top_k=(10,10,10),
        shared_layers=True,
    ),
    loss_coefficients=loss_coefficients,
)
# %%
