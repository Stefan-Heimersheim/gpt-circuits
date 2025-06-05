/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_block --sparsity=0 --config=topk.mlpblock.shk_64x4 --k=5 --max_steps=20000 --name=top5-mlpblock --load_from=shakespeare_64x4
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_block --sparsity=0 --config=topk.mlpblock.shk_64x4 --k=10 --max_steps=20000 --name=top10-mlpblock --load_from=shakespeare_64x4
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_block --sparsity=0 --config=topk.mlpblock.shk_64x4 --k=20 --max_steps=20000 --name=top20-mlpblock --load_from=shakespeare_64x4

/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --sparsity=0 --config=topk.mlpblock.shk_64x4 --k=5 --max_steps=20000 --name=top5-mlplayer --load_from=shakespeare_64x4
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --sparsity=0 --config=topk.mlpblock.shk_64x4 --k=10 --max_steps=20000 --name=top10-mlplayer --load_from=shakespeare_64x4
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --sparsity=0 --config=topk.mlpblock.shk_64x4 --k=20 --max_steps=20000 --name=top20-mlplayer --load_from=shakespeare_64x4
