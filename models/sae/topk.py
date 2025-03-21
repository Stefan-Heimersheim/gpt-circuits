from typing import Optional

import torch
import torch.nn as nn

from config.sae.models import SAEConfig
from config.sae.training import LossCoefficients
from models.sae import EncoderOutput, SAELossComponents, SparseAutoencoder


class TopKSAE(nn.Module, SparseAutoencoder):
    """
    Top-k sparse autoencoder as described in:
    https://arxiv.org/pdf/2406.04093v1
    """
    def __init__(self, layer_idx: int, 
                 config: SAEConfig, 
                 loss_coefficients: Optional[LossCoefficients],
                 parent: Optional[nn.Module] = None):
        super(TopKSAE, self).__init__()
        
        feature_size = config.n_features[layer_idx]  # SAE dictionary size.
        self.feature_size = feature_size
        embedding_size = config.gpt_config.n_embd  # GPT embedding size.
        self.embedding_size = embedding_size
        
        assert config.top_k is not None, "Top-k must be provided. Verify checkpoints/<model_name>/sae.json contains a 'top_k' key."
        assert not config.shared_layers or parent is not None, "Parent must be provided if shared_layers is True."
        assert loss_coefficients is None, "Loss coefficients must be None for Top-k SAE."
        
        self.k = config.top_k[layer_idx]
        self._parent = parent
        self.config = config
        
        #models share encoder/decoder weights with parent model
        if not config.shared_layers:
            self.W_dec = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(feature_size, embedding_size)))
            
            #self.b_enc = nn.Parameter(torch.zeros(feature_size))
            
            try:
                # NOTE: Subclass might define these properties.
                self.W_enc = nn.Parameter(torch.empty(embedding_size, feature_size))
                self.W_enc.data = self.W_dec.data.T.detach().clone()  # initialize W_enc from W_dec
            except KeyError:
                assert False, "W_enc must be defined in subclass. TOP_k SAE how did I get here?"
        else:
            self.W_dec = nn.Parameter(
                torch.as_strided(self._parent.W_dec, 
                                 (feature_size, embedding_size), 
                                 (self._parent.W_dec.stride(0), self._parent.W_dec.stride(1))))
            self.W_enc = nn.Parameter(
                torch.as_strided(self._parent.W_enc, 
                                 (embedding_size, feature_size), 
                                 (self._parent.W_enc.stride(0), self._parent.W_enc.stride(1))))
    
        # each model gets it's own biases 
        self.b_enc = nn.Parameter(torch.zeros(feature_size))
        self.b_dec = nn.Parameter(torch.zeros(embedding_size))
        
    def __repr__(self):
        # to avoid infinite recursion as children point to parent
        parent_ref = self._parent
        self._parent = None
        base_repr = super().__repr__()  # Get standard PyTorch repr
        self._parent = parent_ref
        parent_class_name = parent_ref.__class__.__name__ if parent_ref else "None"
        return base_repr + f"\n(Parent: {parent_class_name})"

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: GPT model activations (B, T, embedding size)
        """
        latent = (x - self.b_dec) @ self.W_enc + self.b_enc

        # Zero out all but the top-k activations
        top_k_values, _ = torch.topk(latent, self.k, dim=-1)
        mask = latent >= top_k_values[..., -1].unsqueeze(-1)
        latent_k_sparse = latent * mask.float()

        return latent_k_sparse

    def decode(self, feature_magnitudes: torch.Tensor) -> torch.Tensor:
        """
        feature_magnitudes: SAE activations (B, T, feature size)
        """
        return feature_magnitudes @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Returns a reconstruction of GPT model activations and feature magnitudes.
        Also return loss components if loss coefficients are provided.

        x: GPT model activations (B, T, embedding size)
        """
        feature_magnitudes = self.encode(x)
        x_reconstructed = self.decode(feature_magnitudes)
        output = EncoderOutput(x_reconstructed, feature_magnitudes)
        if self.k:
            sparsity_loss = 0 # no need for sparsity loss for top-k SAE
            output.loss = SAELossComponents(x, x_reconstructed, feature_magnitudes, sparsity_loss)

        return output
