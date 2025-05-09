# %%
import torch
import os

# Get current directory and keep going up until we find gpt-circuits root
while not os.getcwd().endswith("gpt-circuits"):
    os.chdir("..")
print(os.getcwd())
# %%

from models.factorysparsified import FactorySparsified
from huggingface_hub import snapshot_download

#Pulled from my download code
model_name = 'davidquarel/topk-staircase-share.shakespeare_64x4'
local_dir = 'checkpoints/topk-staircase-share.shakespeare_64x4' #Or whatever location you want
snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)

#pulled from my compute_attributions
model_path = local_dir
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = FactorySparsified.load(model_path, device=device)
#THIS LINE FAILS
# %%
# fixed via model.staircasesparsified.py
# %%
