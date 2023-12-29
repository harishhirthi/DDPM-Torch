import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, Embedding_dim: int):
        super(MultiHeadAttention, self).__init__()

        self.inf = 1e9
        self.dmodel = Embedding_dim
        self.h = num_heads
        self.dk = self.dv = self.dmodel // self.h
        self.Wo = nn.Linear(self.h * self.dv, self.dmodel, bias = False)
        self.Wq = nn.Linear(self.dmodel, self.h * self.dk, bias = False)
        self.Wk = nn.Linear(self.dmodel, self.h * self.dk, bias = False)
        self.Wv = nn.Linear(self.dmodel, self.h * self.dv, bias = False)

    
    # Function to perfrom attention
    def attention(self, Wq: nn.Module, Wk: nn.Module, Wv: nn.Module, x: torch.Tensor, mask = None) -> torch.Tensor:

        """
        An attention function can be described as mapping a query and a set of key-value pairs to an output,
        where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
        of the values, where the weight assigned to each value is computed by a compatibility function of the
        query with the corresponding key.

        Instead of performing a single attention function with dmodel-dimensional keys, values and queries,
        we found it beneficial to linearly project the queries, keys and values h times with different, learned
        linear projections to dk, dk and dv dimensions, respectively.

        Args:
        Wq -> Query Weight Matrix.
        Wk -> Key Weight Matrix.
        Wv -> Value Weight Matrix.
        x -> Input sequence with embeddings.
        mask -> Attention scores to be masked.

        """

        q = Wq(x)
        k = Wk(x)
        v = Wv(x)

        q = q.view(x.size(0), x.size(1), self.h, self.dk).transpose(1, 2)
        k = k.view(x.size(0), x.size(1), self.h, self.dk).transpose(1, 2)
        v = v.view(x.size(0), x.size(1), self.h, self.dv).transpose(1, 2)

        attn_scores = q @ k.transpose(-2, -1) / np.sqrt(self.dk) # Calculation of Attention Scores

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill_(mask == 0, -self.inf)
        
        attn_scores = attn_scores.softmax(dim = -1)
        attn_values = attn_scores @ v # Calculation of Attention values.
        attn_values = attn_values.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.h * self.dv)

        return attn_values, attn_scores
    
    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
    
        x, mask = x, mask
        # Calculation of attention values for number of heads
        attn_values, attn_scores = self.attention(self.Wq, self.Wk, self.Wv, x, mask)
        multiheadattn_values = self.Wo(attn_values) 
    
        return multiheadattn_values     