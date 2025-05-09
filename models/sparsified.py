import dataclasses
import json
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Type

import torch
import torch.nn as nn
from torch.nn import functional as F

from config.sae.models import SAEConfig, SAEVariant, gen_sae_keys, HookPoint
from config.sae.training import LossCoefficients

from models.gpt import GPT
from models.sae import EncoderOutput, SAELossComponents, SparseAutoencoder
from models.sae.gated import GatedSAE, GatedSAE_V2
from models.sae.jumprelu import JumpReLUSAE, StaircaseJumpReLU
from models.sae.standard import StandardSAE, StandardSAE_V2
from models.sae.topk import StaircaseTopKSAE, StaircaseTopKSAEDetach, TopKSAE

from jaxtyping import Tensor, Float

@dataclasses.dataclass
class SparsifiedGPTOutput:
    """
    Output from the forward pass of a sparsified GPT model.
    """

    logits: torch.Tensor
    cross_entropy_loss: torch.Tensor
    # Residual stream activations at every layer
    activations: dict[int, torch.Tensor]
    ce_loss_increases: Optional[torch.Tensor]
    # Compound cross-entropy loss increase if using SAE reconstructions for all trainable layers
    compound_ce_loss_increase: Optional[torch.Tensor]
    sae_loss_components: dict[int, SAELossComponents]
    feature_magnitudes: dict[int, torch.Tensor]
    reconstructed_activations: dict[int, torch.Tensor]
    indices: dict[int, torch.Tensor] = None
    sparsity_losses: dict[int, torch.Tensor] = None # for storing sparsity losses
    aux_losses: dict[int, torch.Tensor] = None # for any other losses

    @property
    def sae_losses(self) -> torch.Tensor:
        """
        SAE losses for each trainable layer.
        """
        return torch.stack([loss.total for loss in self.sae_loss_components.values()])
    
    @property
    def recon_losses(self) -> torch.Tensor:
        """
        Reconstruction losses for each trainable layer.
        """
        return torch.stack([loss.reconstruct for loss in self.sae_loss_components.values()])
    


class SparsifiedGPT(nn.Module):
    """
    GPT Model with sparsified activations using sparse autoencoders.
    """

    def __init__(
        self,
        config: SAEConfig,
        loss_coefficients: Optional[LossCoefficients] = None,
        trainable_layers: Optional[tuple] = None,
    ):
        super().__init__()
        self.config = config
        self.loss_coefficients = loss_coefficients
        self.gpt = GPT(config.gpt_config)

        if self.config.sae_keys is None:
            # assume by default we train an sae across activations between transformer blocks
            warnings.warn("SparsifiedGPT: No SAE keys provided. Go check sae.json, were keys missing?", UserWarning)
            self.config.sae_keys = gen_sae_keys(config.n_features, loc='standard')
            warnings.warn(f"SparsifiedGPT: Using default keys: {self.config.sae_keys}", UserWarning)

        # Construct sae layers
        sae_class: Type[SparseAutoencoder] = self.get_sae_class(config)
        self.sae_idxs = trainable_layers if trainable_layers else list(range(len(self.config.sae_keys)))
        self.saes = nn.ModuleDict(
            dict([(self.config.sae_keys[i], sae_class(i, config, loss_coefficients, self)) for i in self.sae_idxs])
        )

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, is_eval: bool = False
    ) -> SparsifiedGPTOutput:
        """
        Forward pass of the sparsified model.

        :param idx: Input tensor.
        :param targets: Target tensor.
        :param is_eval: Whether the model is in evaluation mode.
        """
        with self.record_activations() as activations:
            with self.use_saes() as encoder_outputs:
                logits, cross_entropy_loss = self.gpt(idx, targets)

        # If targets are provided during training evaluation, gather more metrics
        ce_loss_increases = None
        compound_ce_loss_increase = None
        if is_eval and targets is not None:
            ce_loss_increases = []
            
            if self.config.sae_variant == SAEVariant.JSAE:
                
                for layer_idx in self.layer_idxs:
                    recon_pre_mlp = encoder_outputs[f'{layer_idx}_{HookPoint.MLP_IN}'].reconstructed_activations
                    resid_mid = activations[f'{layer_idx}_{HookPoint.RESID_MID}']

                    sae_logits = self.forward_with_patched_pair(recon_pre_mlp, resid_mid, layer_idx)
                    sae_ce_loss = F.cross_entropy(sae_logits.view(-1, sae_logits.size(-1)), targets.view(-1))
                    ce_loss_increases.append(sae_ce_loss - cross_entropy_loss)
                ce_loss_increases = torch.stack(ce_loss_increases)
            
            else:
            # Calculate cross-entropy loss increase for each SAE layer
                for sae_key, output in encoder_outputs.items():
                    recon_act = output.reconstructed_activations
                    sae_logits = self.forward_with_sae(sae_key, recon_act, activations)
                    sae_ce_loss = F.cross_entropy(sae_logits.view(-1, sae_logits.size(-1)), targets.view(-1))
                    ce_loss_increases.append(sae_ce_loss - cross_entropy_loss)
                ce_loss_increases = torch.stack(ce_loss_increases)

            # Calculate compound cross-entropy loss as a result of patching activations.
            with self.use_saes(activations_to_patch=self.sae_idxs):
                _, compound_cross_entropy_loss = self.gpt(idx, targets)
                compound_ce_loss_increase = compound_cross_entropy_loss - cross_entropy_loss

        return SparsifiedGPTOutput(
            logits=logits,
            cross_entropy_loss=cross_entropy_loss,
            activations=activations,
            ce_loss_increases=ce_loss_increases,
            compound_ce_loss_increase=compound_ce_loss_increase,
            sae_loss_components={i: output.loss for i, output in encoder_outputs.items() if output.loss},
            feature_magnitudes={i: output.feature_magnitudes for i, output in encoder_outputs.items()},
            reconstructed_activations={i: output.reconstructed_activations for i, output in encoder_outputs.items()},
            indices={i: output.indices for i, output in encoder_outputs.items()},
        )
        
    
    def forward_with_patched_pair(self, 
                                recon_pre_mlp: Float[Tensor, "B T n_embd"], 
                                resid_mid: Float[Tensor, "B T n_embd"],
                                layer_idx: int) -> torch.Tensor:
        """
        Forward pass of the model with patched activations, using a pair of reconstructed activations.
        :param recon_pre_mlp: Reconstructed activations just before the MLP. Shape: (B, T, n_embd)
        :param recon_post_mlp: Reconstructed activations just after the MLP. Shape: (B, T, n_embd)
        :param resid_mid: Residual stream activations at the middle of the transformer block. Shape: (B, T, n_embd)
        :param layer_idx: Layer index. 0 patches activations just before the first transformer block.
        """
        assert isinstance(recon_pre_mlp, torch.Tensor), f"recon_pre_mlp: {recon_pre_mlp}"
        assert isinstance(resid_mid, torch.Tensor), f"resid_mid: {resid_mid}"
        
        post_mlp = self.gpt.transformer.h[layer_idx].mlp(recon_pre_mlp)
        post_mlp_recon = self.saes[f'{layer_idx}_{HookPoint.MLP_OUT}'](post_mlp).reconstructed_activations
        
        resid_post = post_mlp_recon + resid_mid
        
        return self.gpt.forward(resid_post, start_at_layer=layer_idx+1).logits
        
    def forward_with_sae(self, sae_key: str, recon_act: torch.Tensor, activations: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the sparsified model with a single SAE layer.
        """
        layer_idx, hook_location = self.split_sae_key(sae_key)
        block = self.gpt.transformer.h[layer_idx]
        attn, ln_2, mlp = block.attn, block.ln_2, block.mlp
        # recall that match/case do not fall through by default, unlike other languages!      
        match hook_location:
            
            case HookPoint.RESID_PRE: # before the transformer blocks
                return self.gpt.forward(recon_act, start_at_layer=layer_idx).logits
            
            case HookPoint.MLP_OUT: # output of mlp, add back to resid_mid, pass through remaining transformer blocks
                resid_mid = activations[f"{layer_idx}_{HookPoint.RESID_MID}"]
                resid_post = resid_mid + recon_act
            
            case HookPoint.MLP_IN: #feed into mlp, add back to resid_mid, pass through remaining transformer blocks
                resid_mid = activations[f"{layer_idx}_{HookPoint.RESID_MID}"]
                resid_post = resid_mid + mlp(recon_act)
            
            case HookPoint.RESID_MID: # between the transformer blocks
                resid_mid = activations[f"{layer_idx}_{HookPoint.RESID_MID}"]
                resid_post = recon_act + mlp(ln_2(resid_mid))
                
            case HookPoint.RESID_POST: # after the transformer block
                resid_post = recon_act
                
            case HookPoint.ATTN_OUT: # output of attention
                resid_pre = activations[f"{layer_idx}_{HookPoint.RESID_PRE}"]
                resid_mid = resid_pre + recon_act
                resid_post = resid_mid + mlp(ln_2(resid_mid))
                
            case HookPoint.ATTN_IN: # input to attention
                resid_pre = activations[f"{layer_idx}_{HookPoint.RESID_PRE}"]
                resid_mid = resid_pre + attn(recon_act)
                resid_post = resid_mid + mlp(ln_2(resid_mid))
                
            case _:
                raise ValueError(f"Invalid hook location: {hook_location}. Must be one of {HookPoint.all()}")
            
        # run the rest of the model
        return self.gpt.forward(resid_post, start_at_layer=layer_idx+1).logits
        
    def split_sae_key(self, sae_key: str) -> tuple[int, HookPoint]:
        """
        Split a SAE key into a layer index and hook location.
        """
        items = sae_key.split('_')
        if len(items) == 2:
            layer_idx = int(items[0])
            hook_location = HookPoint(items[1])
            return layer_idx, hook_location
        else:
            raise ValueError(f"Invalid SAE key format: {sae_key}. Must be in format '{{layer_idx}}_{{hook_location}}'")
        
    def make_grad_hook(self,
                       hooks : list[torch.utils.hooks.RemovableHandle],
                       cache : dict[str, torch.Tensor],
                       target : nn.Module,
                       key : str):
        """
        Hook for computing the gradient of any elementwise function.
        e.g. gelu, relu, DynamicTanh, etc.
        """
        @torch.compiler.disable(recursive=False)
        def activation_grad_hook(module, inputs, outputs, key = key):
            # Want to still run this even if grads are disabled
            pre_actfn = inputs[0]
            post_actfn = outputs
            
            pre_actfn_copy = pre_actfn.detach().requires_grad_(True)
            
            with torch.enable_grad():
                recomputed_post_actfn = module.forward(pre_actfn_copy)
            
                grad_of_actfn = torch.autograd.grad(
                    outputs=recomputed_post_actfn, 
                    inputs=pre_actfn_copy,
                    grad_outputs=torch.ones_like(recomputed_post_actfn), 
                    retain_graph=False,
                    create_graph=False)[0]

            cache[key] = grad_of_actfn.detach()
            return outputs
        
        hooks.append(target.register_forward_hook(activation_grad_hook))

    def make_cache_pre_hook(self,
                            hooks : list[torch.utils.hooks.RemovableHandle],
                            cache : dict[str, torch.Tensor],
                            target : nn.Module,
                            key : str):
        """
        Cache the input of a module.
        """
        return self.make_cache_hook(hooks, cache, target, key, is_prehook = True)
    
    def make_cache_post_hook(self,
                            hooks : list[torch.utils.hooks.RemovableHandle],
                            cache : dict[str, torch.Tensor],
                            target : nn.Module,
                            key : str):
        """
        Cache the output of a module.
        """
        return self.make_cache_hook(hooks, cache, target, key, is_prehook = False)
                        
    def make_cache_hook(self,
                        hooks : list[torch.utils.hooks.RemovableHandle],
                        cache : dict[str, torch.Tensor],
                        target : nn.Module,
                        key : str,
                        is_prehook : bool = True):
        """
        Cache the input or output of a module.
        """
        @torch.compiler.disable(recursive=False)
        def pre_hook_fn(module, inputs):
            cache[key] = inputs[0]
            return inputs
        
        @torch.compiler.disable(recursive=False)
        def post_hook_fn(module, inputs, outputs):
            cache[key] = outputs
            return outputs
        
        if is_prehook:
            hooks.append(target.register_forward_pre_hook(pre_hook_fn))
        else:
            hooks.append(target.register_forward_hook(post_hook_fn))
        
    def make_sae_hook(self,
                      hooks : list[torch.utils.hooks.RemovableHandle],
                      cache : dict[str, EncoderOutput],
                      target : nn.Module,
                      sae_key : str,
                      activations_to_patch : Iterable[str] = (),
                      is_prehook : bool = True):

        @torch.compiler.disable(recursive=False)  # type: ignore
        def sae_prehook_fn(module, inputs):
            cache[sae_key] = self.saes[sae_key](inputs[0])
            if sae_key in activations_to_patch:
                return cache[sae_key].reconstructed_activations
            
        @torch.compiler.disable(recursive=False)  # type: ignore
        def sae_posthook_fn(module, inputs, outputs):
            cache[sae_key] = self.saes[sae_key](outputs)
            if sae_key in activations_to_patch:
                return cache[sae_key].reconstructed_activations
            
        if is_prehook:
            hooks.append(target.register_forward_pre_hook(sae_prehook_fn))
        else:
            hooks.append(target.register_forward_hook(sae_posthook_fn))

    @contextmanager
    def record_activations(self):
        """
        Context manager for recording activations.

        :yield activations: Dictionary of activations.
        """
        # Dictionary for storing results
        activations: dict[int, torch.Tensor] = {}
        processed_resid_pre = set()
        processed_resid_mid = set()
        processed_mlpgrad = set()
        # Register hooks
        hooks = []
        for sae_key in self.saes.keys():
           
            layer_idx, hook_loc = self.split_sae_key(sae_key)
            target, is_prehook = self.get_hook_target(sae_key)
            self.make_cache_hook(hooks, activations, target, key = sae_key, is_prehook = is_prehook)
            
            
            if layer_idx not in processed_resid_pre:
                if hook_loc == HookPoint.ATTN_OUT or hook_loc == HookPoint.ATTN_IN:
                    # require to cache resid_pre to reconstruct resid_mid
                    resid_pre_target = self.gpt.transformer.h[layer_idx]
                    self.make_cache_pre_hook(hooks, activations, resid_pre_target, key_in = f"{layer_idx}_{HookPoint.RESID_PRE}")
                    processed_resid_pre.add(layer_idx)
                
            if layer_idx not in processed_resid_mid:
                if hook_loc == HookPoint.MLP_IN or hook_loc == HookPoint.MLP_OUT:
                    # require to cache resid_mid to reconstruct resid_post
                    resid_mid_target = self.gpt.transformer.h[layer_idx].ln_2
                    self.make_cache_pre_hook(hooks, activations, resid_mid_target, key_in = f"{layer_idx}_{HookPoint.RESID_MID}")
                    processed_resid_mid.add(layer_idx)
            
            if "jsae" in self.config.sae_variant:
                if layer_idx not in processed_mlpgrad:
                    grad_target = self.gpt.transformer.h[layer_idx].mlp.gelu
                    self.make_grad_hook(hooks, activations, grad_target, key = f"{layer_idx}_mlpgrad")
                    processed_mlpgrad.add(layer_idx)
        try:
            yield activations

        finally:
            # Unregister hooks
            for hook in hooks:
                hook.remove()

    @contextmanager
    def use_saes(self, activations_to_patch: Iterable[str] = ()):
        """
        Context manager for using SAE layers during the forward pass.

        :param activations_to_patch: Layer indices for patching residual stream activations with reconstructions.
        :yield encoder_outputs: Dictionary of encoder outputs.
        """
        # Dictionary for storing results
        encoder_outputs: dict[int, EncoderOutput] = {}

        # Register hooks
        hooks = []
        for sae_key in self.saes.keys():
            target, is_prehook = self.get_hook_target(sae_key)
            self.make_sae_hook(hooks, encoder_outputs, target, sae_key, activations_to_patch, is_prehook)

        try:
            yield encoder_outputs

        finally:
            # Unregister hooks
            for hook in hooks:
                hook.remove()


    def get_hook_target(self, sae_key: str) -> tuple[nn.Module, bool]:
        """
        SAE layer -> Targeted module for forward pre-hook.
        """
        layer_idx, hook_loc = self.split_sae_key(sae_key)
        
        if layer_idx == self.config.gpt_config.n_layer and hook_loc == HookPoint.ACT:
            return self.gpt.transformer.ln_f, True
        
        if layer_idx < self.config.gpt_config.n_layer:
            block = self.gpt.transformer.h[layer_idx]
            
            match hook_loc:
                case HookPoint.RESID_PRE:
                    target, is_prehook = block, True
                case HookPoint.MLP_IN:
                    target, is_prehook = block.mlp, True
                case HookPoint.MLP_OUT:
                    target, is_prehook = block.mlp, False
                case HookPoint.RESID_MID:
                    target, is_prehook = block.ln_2, True
                case HookPoint.RESID_POST:
                    target, is_prehook = block, False
                case HookPoint.ATTN_OUT:
                    target, is_prehook = block.attn, False
                case HookPoint.ATTN_IN:
                    target, is_prehook = block.attn, True
                case _:
                    raise ValueError(f"Invalid hook location: {hook_loc}")
            
            return target, is_prehook
        
        if layer_idx > self.config.gpt_config.n_layer:
            raise ValueError(f"Invalid layer index: {layer_idx}")
        
    
    @classmethod
    def load(cls, dir, loss_coefficients=None, trainable_layers=None, device: torch.device = torch.device("cpu")):
        """
        Load a sparsified GPT model from a directory.
        """
        # Load GPT model
        gpt = GPT.load(dir, device=device)

        # Load SAE config
        meta_path = os.path.join(dir, "sae.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        config = SAEConfig(**meta)
        config.gpt_config = gpt.config

        # Create model using saved config
        print(cls)
        model = cls(config, loss_coefficients, trainable_layers)
        model.gpt = gpt

        # Load SAE weights
        for module in model.saes.values():
            assert isinstance(module, SparseAutoencoder)
            module.load(Path(dir), device=device)

        return model

    def load_gpt_weights(self, dir):
        """
        Load just the GPT model weights without loading SAE weights.
        """
        device = next(self.gpt.lm_head.parameters()).device
        self.gpt = GPT.load(dir, device=device)

    def save(self, dir, layers_to_save: Optional[list[str]] = None):
        """
        Save the sparsified GPT model to a directory.

        :param dir: Directory for saving weights.
        :param layers_to_save: Module names for SAE layers to save. If None, all layers will be saved.
        """
        # Save GPT model
        self.gpt.save(dir)

        # Save SAE config
        meta_path = os.path.join(dir, "sae.json")
        meta = dataclasses.asdict(self.config, dict_factory=SAEConfig.dict_factory)
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        # Which layers should we save?
        layers_to_save = layers_to_save or list(self.saes.keys())

        # Save SAE modules
        for layer_name, module in self.saes.items():
            if layer_name in layers_to_save:
                assert isinstance(module, SparseAutoencoder)
                module.save(Path(dir))

    def get_sae_class(self, config: SAEConfig) -> Type[SparseAutoencoder]:
        """
        Maps the SAE variant to the actual class.
        """
        match config.sae_variant:
            case SAEVariant.STANDARD:
                return StandardSAE
            case SAEVariant.STANDARD_V2:
                return StandardSAE_V2
            case SAEVariant.GATED:
                return GatedSAE
            case SAEVariant.GATED_V2:
                return GatedSAE_V2
            case SAEVariant.JUMP_RELU:
                return JumpReLUSAE
            case SAEVariant.JUMP_RELU_STAIRCASE:
                return StaircaseJumpReLU
            case SAEVariant.TOPK:
                return TopKSAE
            case SAEVariant.TOPK_STAIRCASE:
                return StaircaseTopKSAE
            case SAEVariant.TOPK_STAIRCASE_DETACH:
                return StaircaseTopKSAEDetach
            case SAEVariant.JSAE:
                return TopKSAE
            case _:
                raise ValueError(f"Unrecognized SAE variant: {config.sae_variant}")
