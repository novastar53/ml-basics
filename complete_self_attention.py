# This self-attention module is based on the implementation
# provided here: https://peterbloem.nl/blog/transformers


import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, k, heads=4, mask=False):
        """
        k: word embedding length
        heads: number of attention heads
        mask: ?
        """ 

        super().__init__()

        # make sure you can split your input embedding
        # into the desired number of attention heads
        assert k % heads == 0

        self.k = k
        self.heads = heads


        # set up matrices for queries, keys and values
        self.to_keys = nn.Linear(k, k, bias=False)
        self.to_queries = nn.Linear(k, k, bias=False)
        self.to_values = nn.Linear(k, k, bias=False)

        # to combine the outputs from the attention heads
        self.unify_heads = nn.Linear(k, k)

    
    def forward(self, x):

        b, t, k = x.size()
        h = self.heads()

        # compute the queries, keys and values
        queries = self.to_queries(x)
        keys = self.to_keys(x)
        values = self.to_values(x)


        # divide them up amongst the attention heads
        s = k // h

        keys = keys.view(b, t, h, s)
        

