import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

from config.sae.models import SAEConfig
from config.sae.training import LossCoefficients
from models.gpt import GPT
from models.sae import EncoderOutput, SparseAutoencoder
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput

from typing import Literal

from jaxtyping import Float
from torch import Tensor

class MLPSparsifiedGPT(SparsifiedGPT):
    def __init__(
        self, 
        config: SAEConfig,
        loss_coefficients: Optional[LossCoefficients] = None,
        trainable_layers: Optional[tuple] = None,
    ):
        #don't actually want to call SparsifiedGPT.__init__, but we want to inherit from it
        nn.Module.__init__(self) 
        self.config = config
        self.loss_coefficients = loss_coefficients
        self.gpt = GPT(config.gpt_config)
        assert len(config.n_features) == self.gpt.config.n_layer * 2
        self.layer_idxs = trainable_layers if trainable_layers else list(range(self.gpt.config.n_layer))
        sae_keys = [f'{x}_{y}' for x in self.layer_idxs for y in ['mlpin', 'mlpout']] # index of the mlpin and mlpout activations
        
        sae_class: Type[SparseAutoencoder] = self.get_sae_class(config)
        
        self.saes = nn.ModuleDict(dict([(key, sae_class(idx, config, loss_coefficients, self)) 
                                        for idx, key in enumerate(sae_keys)]))
        self.sae_keys = sae_keys
       
    def get_sae_logits(self, 
                       sae_key: str, 
                       activations: dict[int, torch.Tensor], 
                       encoder_outputs: dict[int, EncoderOutput]) -> torch.Tensor:
        layer_idx, hook_loc = self.split_sae_key(sae_key)
        recon_act = encoder_outputs[sae_key].reconstructed_activations
        resid_mid = activations[f'{layer_idx}_residmid']
        
        if hook_loc == 'mlpin':
            recon_act = self.gpt.transformer.h[layer_idx].mlp(recon_act)
        elif hook_loc == 'mlpout':
            pass
        else:
            raise ValueError(f"Invalid hook location: {hook_loc}")
        
        resid_post = recon_act + resid_mid
        # forward through transformer blocks starting with the specified layer
        return self.gpt.forward(resid_post, start_at_layer=layer_idx+1).logits
       
    @contextmanager
    def record_activations(self):
        """
        Context manager for recording residual stream activations.

        :yield activations: Dictionary of activations.
        activations[f'{layer_idx}_mlpin'] = h[layer_idx].mlpin
        activations[f'{layer_idx}_mlpout'] = h[layer_idx].mlpout
        # NOTE: resid_mid is stored in self.resid_mid_cache, not yielded directly
        """
        act: dict[str, torch.Tensor] = {}

        # Register hooks
        hooks = []
        for layer_idx in self.layer_idxs:
            mlp = self.gpt.transformer.h[layer_idx].mlp
            ln2 = self.gpt.transformer.h[layer_idx].ln_2
            
            self.make_cache_post_hook(hooks, act, mlp, key_in = f"{layer_idx}_mlpin", key_out = f"{layer_idx}_mlpout")
            self.make_cache_pre_hook(hooks, act, ln2, key_in = f"{layer_idx}_residmid")        
    
            # MLP Sparsifies doesn't use grad hooks, but we can still use it for JSAE
        try:
            yield act

        finally:
            # Unregister hooks
            for hook_fn in hooks:
                hook_fn.remove()
            
    @contextmanager
    def use_saes(self, activations_to_patch: Iterable[str] = ()):
        """
        Context manager for using SAE layers during the forward pass.

        :param activations_to_patch: Layer indices and hook locations for patching residual stream activations with reconstructions.
        :yield encoder_outputs: Dictionary of encoder outputs.
        key = f"{layer_idx}_{hook_loc}" e.g. 0_mlpin, 0_mlpout, 1_mlpin, 1_mlpout, etc.
        """
        encoder_outputs: dict[str, EncoderOutput] = {}

        hooks = []
        for layer_idx in self.layer_idxs:

            mlp = self.gpt.transformer.h[layer_idx].mlp

            sae_key = f'{layer_idx}_mlpin'
            self.make_sae_pre_hook(hooks, encoder_outputs, mlp, sae_key, activations_to_patch)
            
            sae_key = f'{layer_idx}_mlpout'
            self.make_sae_post_hook(hooks, encoder_outputs, mlp, sae_key, activations_to_patch)
            
        try:
            yield encoder_outputs

        finally:
            for hook_fn in hooks:
                hook_fn.remove()
                
    # def post_init(self):
    #     pass
        # While a nice idea, it might break other code as TopKSAE
        # has a different save format 
        # for sae_key, sae in self.saes.items():
        #     self.saes[sae_key].sae_key = sae_key
        
        # def sae_save(self, dirpath: Path):
        #     """
        #     Save the sparse autoencoder to a file in the specified directory.
        #     """
        #     sae_key = self.sae_key
        #     weights_path = dirpath / f"sae.{sae_key}.safetensors"
        #     save_model(self, str(weights_path))

        # def sae_load(self, dirpath: Path, device: torch.device):
        #     """
        #     Load the sparse autoencoder from a file in the specified directory.
        #     """
        #     sae_key = self.sae_key
        #     weights_path = dirpath / f"sae.{sae_key}.safetensors"
        #     load_model(self, weights_path, device=device.type)
        
        # for sae_key, sae in self.saes.items():
        #     sae.save = lambda dirpath, current_sae=sae: sae_save(current_sae, dirpath)
        #     sae.load = lambda dirpath, device, current_sae=sae: sae_load(current_sae, dirpath, device)
        