import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, Tuple, Union, Dict, List
from functools import partial

Past = Tuple[torch.Tensor, torch.Tensor]

class PositionalEmbedding(nn.Embedding):
    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)

    def _load_from_state_dict(self,
                              state_dict: Dict[str, torch.Tensor],
                              prefix: str,
                              *args,
                              **kwargs):
        weight = state_dict[f'{prefix}weight']

        # Reduce or expand the positional embedding matrix to increase or
        # decrease the total sequence length.
        if weight.size(0) < self.num_embeddings:
            weight = torch.cat((weight, self.weight[weight.size(0):]), dim=0)
        elif weight.size(0) > self.num_embeddings:
            weight = weight[:self.num_embeddings]

        state_dict[f'{prefix}weight'] = weight
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        position = torch.arange(offset, offset + x.size(-1),
                                dtype=torch.long, device=x.device)
        position = position.view((1,) * (x.ndim - 1) + (-1,)).expand_as(x)

        return super().forward(position)


class TokenEmbedding(nn.Embedding):
    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)

    def forward(self,
                x: torch.Tensor,
                transposed: bool = False) -> torch.Tensor:
        if transposed:
            return torch.matmul(x, self.weight.transpose(0, 1))
        else:
            return super().forward(x)

class BaseAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))

        if mask is not None:
            x += mask.type_as(x) * x.new_tensor(-1e4)
        x = self.dropout(x.softmax(-1))

        return torch.matmul(x, v)


class MultiHeadAttention(BaseAttention):
    def __init__(self, heads: int, dropout: float = 0.1):
        super().__init__(dropout)
        self.heads = heads

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Split the tensors to multi-heads.
        q = q.view(q.size()[:-1] + (self.heads, q.size(-1) // self.heads))
        k = k.view(k.size()[:-1] + (self.heads, k.size(-1) // self.heads))
        v = v.view(v.size()[:-1] + (self.heads, v.size(-1) // self.heads))

        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)

        if mask is not None:
            mask = mask.unsqueeze(-3)

        # Calculate multi-headed attentions and merge them into one.
        return (super().forward(q, k, v, mask)
                .transpose(-3, -2)
                .contiguous()
                .view(q.size()[:-3] + (q.size(-2), v.size(-1) * self.heads)))


class AttentionLayer(nn.Module):
    def __init__(self, heads: int, dims: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(heads, dropout)
        self.proj_q = nn.Linear(dims, dims)
        self.proj_k = nn.Linear(dims, dims)
        self.proj_v = nn.Linear(dims, dims)
        self.linear = nn.Linear(dims, dims)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                past: Optional[Past] = None,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Past]:
        q, k, v = self.proj_q(q), self.proj_k(k), self.proj_v(v)

        # Reuse attention keys and values by concatenating to the current ones.
        if past is not None:
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)

        x = self.linear(self.attn(q, k, v, mask))
        return x, (k, v)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)


class PositionwiseFeedForward(nn.Sequential):
    def __init__(self, dims: int, rate: int = 4, dropout: float = 0.1):
        super().__init__(
            nn.Linear(dims, dims * rate),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dims * rate, dims))
        
class PadMasking(nn.Module):
    def __init__(self, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        is_pad = (x == self.pad_idx).unsqueeze(-2)
        shifted = torch.zeros(x.size()[:-1] + (1, offset,),
                              dtype=torch.bool, device=x.device)

        mask = torch.cat((shifted, is_pad), dim=-1)
        return mask.expand(x.shape + mask.shape[-1:])


class FutureMasking(nn.Module):
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq_len = x.size(-1)

        # Create shifted upper triangular matrix.
        future = torch.ones((seq_len, seq_len + offset),
                            dtype=torch.bool, device=x.device)
        future = future.triu(offset + 1)

        mask = future.view((1,) * (x.ndim - 1) + future.size())
        return mask.expand(x.shape + mask.shape[-1:])
    
class TransformerLayer(nn.Module):
    def __init__(self,
                 heads: int,
                 dims: int,
                 rate: int,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = AttentionLayer(heads, dims, dropout)
        self.ff = PositionwiseFeedForward(dims, rate, dropout)
        self.ln_attn = nn.LayerNorm(dims)
        self.ln_ff = nn.LayerNorm(dims)

    def forward(self,
                x: torch.Tensor,
                past: Optional[Past] = None,
                mask: Optional[torch.Tensor] = None,
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, Past]]:
        # Layer normalizations are performed before the layers respectively.
        a = self.ln_attn(x)
        a, past = self.attn(a, a, a, past, mask)

        x = x + a
        x = x + self.ff(self.ln_ff(x))

        return x if self.training else (x, past)
    
class Transformer(nn.Module):
    def __init__(self,
                 layers: int,
                 pad_idx: int,
                 words: int,
                 seq_len: int,
                 heads: int,
                 dims: int,
                 rate: int = 4,
                 dropout: float = 0.1,
                 bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
        self.pad_masking = PadMasking(pad_idx)
        self.future_masking = FutureMasking()

        self.positional_embedding = PositionalEmbedding(seq_len, dims)
        self.token_embedding = TokenEmbedding(words, dims)
        self.dropout_embedding = nn.Dropout(dropout)

        self.transformers = nn.ModuleList([
            TransformerLayer(heads, dims, rate, dropout)
            for _ in range(layers)])
        self.ln_head = nn.LayerNorm(dims)

    def forward(self,
                x: torch.Tensor,
                past: Optional[List[Past]] = None,
                use_grad_ckpt: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Past]]]:
        offset = past[0][0].size(-2) if past is not None else 0

        # Create masking tensor.
        mask = self.pad_masking(x, offset)
        if not self.bidirectional:
            mask = mask + self.future_masking(x, offset)

        # Use token embedding and positional embedding layers.
        x = self.token_embedding(x) + self.positional_embedding(x, offset)
        x = self.dropout_embedding(x)

        # Apply transformer layers sequentially.
        present = []
        for i, transformer in enumerate(self.transformers):
            if self.training and use_grad_ckpt:
                transformer = partial(torch.utils.checkpoint.checkpoint,
                                      transformer)

            x = transformer(x, past[i] if past is not None else None, mask)

            if not self.training:
                present.append(x[1])
                x = x[0]

        x = self.ln_head(x)
        x = self.token_embedding(x, transposed=True)

        return x if self.training else (x, present)

def get_gpt3_13b(vocab_size, pad_idx):
    # https://github.com/affjljoo3581/GPT2/tree/master
    return Transformer(
        layers=40,
        pad_idx=pad_idx,  # this is probably wrong
        words=vocab_size,
        seq_len=2048,
        heads=40,
        dims=5140,
        rate=4,
        dropout=0.1,
        bidirectional=True
    )

def get_gpt3_175b(vocab_size, pad_idx):
    return Transformer(
        layers=96,
        pad_idx=pad_idx,
        words=vocab_size,
        seq_len=2048,
        heads=96,
        dims=12288,
        rate=4, # used in pwise ff
        dropout=0.0,
        bidirectional=True
    )