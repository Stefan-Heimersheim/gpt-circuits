from typing import Optional

import torch
import torch.nn as nn

from config.sae.models import SAEConfig
from config.sae.training import LossCoefficients


class SharedLayer():
    """
    Wrapper for multiple SAE layers with shared weights.
    Each indexed layer behaves like a virtual SAE with a different top-k value.
    Can be used as a drop-in replacement for nn.ModuleDict
    """
    def __init__(self, 
                 trainable_layers: list[int], 
                 config: SAEConfig, 
                 loss_coefficients: Optional[LossCoefficients]):
        super(SharedLayer, self).__init__()
        self.config = config
        self.loss_coefficients = loss_coefficients
        self.trainable_layers = trainable_layers
        self.sae_variant = config.sae_variant
        from models.sparsified import get_sae_class
        SAEClass = get_sae_class(config)
        
        feature_size = max(config.n_features)
        embedding_size = config.gpt_config.n_embd
        print(f"feature_size: {feature_size}, embedding_size: {embedding_size}")
        
        # Shared parameters across all layers
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(feature_size, embedding_size)))
        self.b_enc = nn.Parameter(torch.zeros(feature_size))
        self.b_dec = nn.Parameter(torch.zeros(embedding_size))
        try:
            # NOTE: Subclass might define these properties.
            self.W_enc = nn.Parameter(torch.empty(embedding_size, feature_size))
            self.W_enc.data = self.W_dec.data.T.detach().clone()  # initialize W_enc from W_dec
        except KeyError:
            pass
        
        self.saes = nn.ModuleDict(dict([(f'{i}', SAEClass(i, config, loss_coefficients, self)) 
                                        for i in self.trainable_layers]))
    
    def __getitem__(self, key):
        """ Allow sae['0'](x) access, ensuring key is an integer index """
        return self.saes[key]

    def __iter__(self):
        return iter(self.saes)

    def items(self):
        return self.saes.items()

    def keys(self):
        return self.saes.keys()

    def values(self):
        return self.saes.values()