import dataclasses
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Type, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from config.sae.models import SAEConfig, SAEVariant
from config.sae.training import LossCoefficients
from models.gpt import GPT
from models.sae import EncoderOutput, SAELossComponents, SparseAutoencoder
from models.sae.gated import GatedSAE, GatedSAE_V2
from models.sae.jumprelu import JumpReLUSAE, StaircaseJumpReLU
from models.sae.standard import StandardSAE, StandardSAE_V2
from models.sae.topk import StaircaseTopKSAE, StaircaseTopKSAEDetach, TopKSAE
from config.sae.models import HookPoint

from jaxtyping import Float
from torch import Tensor

import warnings
@dataclasses.dataclass
class SparsifiedGPTOutput:
    """
    Output from the forward pass of a sparsified GPT model.
    """

    logits: Float[Tensor, "batch seq vocab"]
    cross_entropy_loss: Float[Tensor, ""]
    # Residual stream activations at every layer
    activations: dict[int, Float[Tensor, "batch seq n_embd"]]
    ce_loss_increases: Optional[Float[Tensor, "n_layer"]]
    # Compound cross-entropy loss increase if using SAE reconstructions for all trainable layers
    compound_ce_loss_increase: Optional[Float[Tensor, ""]]
    sae_loss_components: dict[int, SAELossComponents]
    feature_magnitudes: dict[int, Float[Tensor, "batch seq feature_size"]]
    reconstructed_activations: dict[int, Float[Tensor, "batch seq n_embd"]]
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

        # Construct sae layers
        sae_class: Type[SparseAutoencoder] = self.get_sae_class(config)
        self.layer_idxs = trainable_layers if trainable_layers else list(range(len(config.n_features)))
        warnings.warn("SparsifiedGPT: Use of self.saes[i] is deprecated. Use self.saes[f'{i}_{HookPoint.ACT.value}'] instead.")
        self.saes = nn.ModuleDict(dict([(f"{i}_{HookPoint.ACT.value}", sae_class(i, config, loss_coefficients, self)) for i in self.layer_idxs]))
        if self.config.sae_keys is None:
            self.config.sae_keys = tuple(self.saes.keys())
        
        assert config.sae_variant != SAEVariant.JSAE_LAYER, f"JSAE not supported for SparsifiedGPT. See JSparsifiedGPT."
        assert config.sae_variant != SAEVariant.JSAE_BLOCK, f"JSAE_BLOCK not supported for SparsifiedGPT. See JBlockSparsifiedGPT."
        
    @property
    def eval_keys(self) -> Union[list[str], list[int]]:
        return self.saes.keys()
        
    def get_sae_logits(self, 
                       eval_key: int | str,
                       activations: dict[int, torch.Tensor], 
                       encoder_outputs: dict[int, EncoderOutput]) -> torch.Tensor:
        assert isinstance(eval_key, str), "eval_key must be a string for SparsifiedGPT"
        layer_idx, _ = self.split_sae_key(eval_key)
        resid = encoder_outputs[eval_key].reconstructed_activations
        sae_logits = self.gpt.forward(resid, start_at_layer=layer_idx).logits
        return sae_logits

    @contextmanager
    def record_activations(self):
        """
        Context manager for recording residual stream activations.

        :yield activations: Dictionary of activations.
        """
        # Dictionary for storing results
        activations: dict[int, torch.Tensor] = {}

        # Register hooks
        hooks = []
        for layer_idx in list(range(len(self.config.n_features))):
            target = self.get_hook_target(layer_idx)
            self.make_cache_pre_hook(hooks, activations, target, key_in = f"{layer_idx}_residmid")

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
        for layer_idx in self.layer_idxs:
            target = self.get_hook_target(layer_idx)
            self.make_sae_pre_hook(hooks, encoder_outputs, target, f"{layer_idx}_{HookPoint.ACT.value}", activations_to_patch)


        try:
            yield encoder_outputs

        finally:
            # Unregister hooks
            for hook in hooks:
                hook.remove()

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, is_eval: bool = False, stop_at_layer: Optional[int] = None
    ) -> SparsifiedGPTOutput:
        """
        Forward pass of the sparsified model.

        :param idx: Input tensor.
        :param targets: Target tensor.
        :param is_eval: Whether the model is in evaluation mode.
        :param stop_at_layer: Optional layer index to stop the forward pass at. Exclusive.
                              If specified, returns early with limited output. Defaults to None (run full model).
        """
        with self.record_activations() as activations:
            with self.use_saes() as encoder_outputs:
                gpt_output = self.gpt(idx, targets, stop_at_layer=stop_at_layer)
                
                # Handle early termination case
                if stop_at_layer is not None:
                    # GPT returns just the residual stream tensor when stop_at_layer is used
                    return SparsifiedGPTOutput(
                        logits=None,  # No logits when stopping early
                        cross_entropy_loss=None,  # No loss when stopping early
                        activations=activations,
                        ce_loss_increases=None,
                        compound_ce_loss_increase=None,
                        sae_loss_components={i: output.loss for i, output in encoder_outputs.items() if output.loss},
                        feature_magnitudes={i: output.feature_magnitudes for i, output in encoder_outputs.items()},
                        reconstructed_activations={i: output.reconstructed_activations for i, output in encoder_outputs.items()},
                        indices={i: output.indices for i, output in encoder_outputs.items()},
                    )
                
                # Normal case - GPT returns GPTOutput(logits, loss)
                logits, cross_entropy_loss = gpt_output

        # If targets are provided during training evaluation, gather more metrics
        ce_loss_increases = None
        compound_ce_loss_increase = None
        if is_eval and targets is not None:
            # Calculate cross-entropy loss increase for each SAE layer
            with torch.no_grad():
                self.eval()
                ce_loss_increases = []
                for eval_key in self.eval_keys:
                    sae_logits = self.get_sae_logits(eval_key, activations, encoder_outputs)
                    sae_ce_loss = F.cross_entropy(sae_logits.view(-1, sae_logits.size(-1)), targets.view(-1))
                    ce_loss_increases.append(sae_ce_loss - cross_entropy_loss)
                ce_loss_increases = torch.stack(ce_loss_increases)

                # Calculate compound cross-entropy loss as a result of patching activations.
                with self.use_saes(activations_to_patch=self.saes.keys()):
                    _, compound_cross_entropy_loss = self.gpt(idx, targets)
                    compound_ce_loss_increase = compound_cross_entropy_loss - cross_entropy_loss
                self.train()

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
        
    def split_sae_key(self, sae_key: str) -> tuple[int, str]:
        """
        Split a SAE key into a layer index and hook location.
        """
        items = sae_key.split('_')
        if len(items) == 1:
            return int(items[0]), "resid_mid"
        elif len(items) == 2:
            return int(items[0]), items[1]
        else:
            raise ValueError(f"Invalid SAE key: {sae_key}")
        
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
                        *,
                        key_in : Optional[str] = None, 
                        key_out : Optional[str] = None):
        
        @torch.compiler.disable(recursive=False)
        def pre_hook_fn(module, inputs):
            if key_in is not None:
                cache[key_in] = inputs[0]
            return inputs
        
        hooks.append(target.register_forward_pre_hook(pre_hook_fn))
            
    def make_cache_post_hook(self,
                            hooks : list[torch.utils.hooks.RemovableHandle],
                            cache : dict[str, torch.Tensor],
                            target : nn.Module,
                            *,
                            key_in : Optional[str] = None, 
                            key_out : Optional[str] = None):
               
        @torch.compiler.disable(recursive=False)
        def post_hook_fn(module, inputs, outputs):
            if key_in is not None:
                cache[key_in] = inputs[0]
            if key_out is not None:
                cache[key_out] = outputs
            return outputs
        
        hooks.append(target.register_forward_hook(post_hook_fn))
            
    def make_sae_pre_hook(self,
                      hooks : list[torch.utils.hooks.RemovableHandle],
                      cache : dict[str, EncoderOutput],
                      target : nn.Module,
                      sae_key : str,
                      activations_to_patch : Iterable[str] = ()):

        @torch.compiler.disable(recursive=False)  # type: ignore
        def sae_prehook_fn(module, inputs):
            cache[sae_key] = self.saes[sae_key](inputs[0])
            if sae_key in activations_to_patch:
                return cache[sae_key].reconstructed_activations

        hooks.append(target.register_forward_pre_hook(sae_prehook_fn))
        

    def make_sae_post_hook(self,
                           hooks : list[torch.utils.hooks.RemovableHandle],
                          cache : dict[str, EncoderOutput],
                          target : nn.Module,
                          sae_key : str,
                          activations_to_patch : Iterable[str] = ()):
        
        @torch.compiler.disable(recursive=False)  # type: ignore
        def sae_posthook_fn(module, inputs, outputs):
            cache[sae_key] = self.saes[sae_key](outputs)
            if sae_key in activations_to_patch:
                return cache[sae_key].reconstructed_activations
            
        hooks.append(target.register_forward_hook(sae_posthook_fn))

    def get_hook_target(self, layer_idx) -> nn.Module:
        """
        SAE layer -> Targeted module for forward pre-hook.
        """
        if layer_idx < self.config.gpt_config.n_layer:
            return self.gpt.transformer.h[layer_idx]  # type: ignore
        elif layer_idx == self.config.gpt_config.n_layer:
            return self.gpt.transformer.ln_f  # type: ignore
        raise ValueError(f"Invalid layer index: {layer_idx}")


    @classmethod
    def make(cls, config, loss_coefficients, trainable_layers):
        return cls(config, loss_coefficients, trainable_layers)

    @classmethod
    def load(cls, dir, loss_coefficients=None, trainable_layers=None, device: torch.device | str = torch.device("cpu")):
        """
        Load a sparsified GPT model from a directory.
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        # Load GPT model
        gpt = GPT.load(dir, device=device)

        # Load SAE config
        meta_path = os.path.join(dir, "sae.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        config = SAEConfig(**meta)
        config.gpt_config = gpt.config
        # Create model using saved config
        print(f"Loading {cls.__name__} with config: {config}")
        model = cls.make(config, loss_coefficients, trainable_layers)
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


    def save_meta(self, dir):
        """
        Save the SAE config to the output directory.
        """
        meta_path = os.path.join(dir, "sae.json")
        meta = dataclasses.asdict(self.config, dict_factory=SAEConfig.dict_factory)
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    def save(self, dir, sae_keys_to_save: Optional[list[str]] = None):
        """
        Save the sparsified GPT model to a directory.

        :param dir: Directory for saving weights.
        :param sae_keys_to_save: Module names for SAE layers to save. If None, all layers will be saved.
        """
        # Save GPT model
        self.gpt.save(dir)
        # Save SAE config
        self.save_meta(dir)

        # Which layers should we save?
        if sae_keys_to_save is None:
            sae_keys_to_save = list(self.saes.keys())

        # Save SAE modules
        print(f"Saving SAEs: {sae_keys_to_save}")
        for layer_name, module in self.saes.items():
            if layer_name in sae_keys_to_save:
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
            case SAEVariant.JSAE_LAYER:
                return TopKSAE
            case SAEVariant.JSAE_BLOCK:
                return TopKSAE
            case SAEVariant.STAIRCASE_BLOCK:
                return StaircaseTopKSAE
            case _:
                raise ValueError(f"Unrecognized SAE variant: {config.sae_variant}")
