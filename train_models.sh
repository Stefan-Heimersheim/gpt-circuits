#!/bin/bash


# Train the LLM
python -m training.gpt --config=shakespeare_64x4

# Train the SAEs


# FF Layer
# ------------------------

# Train JSAEs across FF Block
for optional_parameter in "1.0e-03" "1.2e-03" "1.5e-03" "1.8e-03" "2.2e-03" "2.7e-03" "3.3e-03" "3.9e-03" "4.7e-03" "5.6e-03" "6.8e-03" "1.0e-02"
    python -m training.sae.jsae_concurrent --sparsity=$sparsity --load_from=shakespeare_64x4 --name=ff_layer_jsae_$optional_parameter 
done
python -m training.sae.jsae_concurrent --sparsity=0 --load_from=shakespeare_64x4 --name=ff_layer_jsae_0.0ep00 


# FF Block
# ------------------------
# Equivalent to training standard TopK SAEs across FF Block
python -m training.sae.jsae_block --sparsity=0 --load_from=shakespeare_64x4 --name=ff_block_topk

# Train Staircase SAEs across FF Block
python -m training.sae.staircase_concurrent --config=staircase-pairsx8.shakespeare_64x4 --load_from=shakespeare_64x4 --name=ff_block_staircase


# Transformer Block
# ------------------------
# Train SAEs across Transformer Block

# Standard TopK
python -m training.sae.concurrent --config=topk-10-x8.shakespeare_64x4 --load_from=shakespeare_64x4 --name=trans_block_topk

# Staircase TopK
python -m training.sae.staircase_concurrent --config=topk-staircase-10-x8-share.shakespeare_64x4 --load_from=shakespeare_64x4 --name=trans_block_staircase