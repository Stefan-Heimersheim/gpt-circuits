
# %%
import torch
from eindex import eindex
import einops

batch, seq, k, feat = 2,3,5,7

jacobian = torch.randn((batch,seq,feat,batch,seq,feat)) 
# model has no cross batch/seq interactions, so only
# the diagonal terms along b and s are non zero in practice
in_idx = torch.randint(low=0, high = feat, size = (batch, seq, k))
out_idx = torch.randint(low=0, high = feat, size = (batch, seq, k))


# Want output[b,s,k2,k1] = A[b,s,out_idx[b,s,k2],b,s,in_idx[b,s,k1]]

result_slow = torch.zeros((batch,seq,k,k))
for b in range(batch):
    for s in range(seq):
        for k1 in range(k):
            for k2 in range(k):
                result_slow[b,s,k2,k1] = jacobian[b,s,out_idx[b,s,k2],b,s,in_idx[b,s,k1]]


# Works if we take diagonals along batch and seq first 
jacobian_diag = einops.einsum(jacobian, "b s k2 b s k1 -> b s k2 k1") #take diagonal along batch and seq
result_fast = eindex(jacobian_diag, out_idx, in_idx, "b s [b s k2] [b s k1] -> b s k2 k1")

torch.testing.assert_close(result_slow, result_fast, rtol=1e-4, atol=1e-4)
print("Results are close!")

# Doesn't work if we get eindex to do it all
result_super_fast = eindex(jacobian, out_idx, in_idx, "b s [b s k2] b s [b s k1] -> b s k2 k1")

# %%
