"""
python -m training.sae.jsae_block [--config=jsae.shakespeare_64x4] [--load_from=shakespeare_64x4_dyt] [--sparsity=0.02|0.1,0.2,0.3,0.4]
"""
# %%
import os

while not os.getcwd().endswith("gpt-circuits"):
    os.chdir("..")
print(os.getcwd())

# %%

from config.sae.training import options
from models.jsaeblockparsified import JBlockSparsifiedGPT
from config.gpt.models import NormalizationStrategy
from safetensors.torch import load_model



config = options['jsae.shakespeare_64x4']
load_from = "checkpoints/jblock.shk_64x4-sparse-3.3e-03"

config.sae_config.gpt_config.normalization = NormalizationStrategy.DYNAMIC_TANH
config.sae_config.compile = False

model = JBlockSparsifiedGPT(config.sae_config, 
                            config.loss_coefficients, 
                            config.trainable_layers)


load_model(model.gpt, os.path.join(load_from, "model.safetensors"), device=config.device.type)


for idx, sae_key in enumerate(model.saes.keys()):
    #print(f"loading sae {sae_key} from {os.path.join(load_from, f'sae.{idx}.safetensors')}")
    load_model(model.saes[sae_key], os.path.join(load_from, f"sae.{idx}.safetensors"), device=config.device.type)
# %%

print(model)



# %%