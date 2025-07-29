#!/bin/bash

curl -d "Starting SAE feature extraction for layers 6,8,9,10 on topk and staircase models" ntfy.sh/david_spar_server

# python -m david.interp_sae --config staircase.tblock.gpt2 layers 8
# curl -d "✓ Completed staircase.tblock.gpt2 layers 6,8,9,10 - All jobs finished!" ntfy.sh/david_spar_server 
# .
# python -m david.interp_sae --config topk.tblock.gpt2 layers 8
# curl -d "✓ Completed topk.tblock.gpt2 layers 6,8,9,10" ntfy.sh/david_spar_server


python -m david.interp_sae --config staircase.tblock.gpt2 --layers 8
curl -d "✓ Completed staircase.tblock.gpt2 layers 7 - All jobs finished!" ntfy.sh/david_spar_server 

python -m david.interp_sae --config topk.tblock.gpt2  --layers 8
curl -d "✓ Completed topk.tblock.gpt2 layers 7" ntfy.sh/david_spar_server