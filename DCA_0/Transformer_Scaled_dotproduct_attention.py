import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.linear_final = nn.Linear(300, 300)

    def forward(self, k, q, v, one_key, num_heads, scale=None, tok_mask=None, entity_mask=None, num_layers=1, tmp_num=0):
        """forward.

        Args:
            q: Queries shape [B, L_q, D_q]
            k: Keys shape [B, L_k, D_k]
            v: Values shape [B, L_v, D_v]，similar to the k
            scale: a zoom factor ，float scalar
            attn_mask: Masking shape[B, L_q, L_k]

        Returns:
            context vector and attetention vector
        """
        attention = torch.bmm(q, k.transpose(1, 2))

        if scale is not None:
            attention = attention * scale

        if tmp_num == num_layers - 1:

            if tok_mask is not None:
                attention = (attention * tok_mask.repeat(num_heads, 1, 1)).add_((tok_mask.repeat(num_heads, 1, 1) - 1).mul_(1e10))

            if entity_mask is not None:
                attention = (attention * entity_mask.repeat(num_heads, 1, 1)).add_((entity_mask.repeat(num_heads, 1, 1) - 1).mul_(1e10))
        else:

            if tok_mask is not None:
                attention = (attention * tok_mask.repeat(num_heads, 1, 1)).add_((tok_mask.repeat(num_heads, 1, 1) - 1).mul_(1e10))

            if entity_mask is not None:
                attention = (attention * entity_mask.repeat(num_heads, 1, 1)).add_((entity_mask.repeat(num_heads, 1, 1) - 1).mul_(1e10))

        # compute softmax
        soft_attention = self.softmax(attention)
        top_soft_attention, _ = torch.max(soft_attention, dim=1, keepdim=True)
        context = torch.bmm(top_soft_attention, one_key)
        sorce = torch.bmm(v, context.permute(0, 2, 1))
        q = v * sorce

        # if tmp_num != num_layers - 1:
        q = q.view(q.size(0) // num_heads, -1, 300)
        q = self.linear_final(q)
        k = k.view(k.size(0) // num_heads, -1, 300)

        if tmp_num == num_layers - 1:

            attention = torch.bmm(q,  k.transpose(1, 2))

            if scale is not None:
                attention = attention * scale

            if tok_mask is not None:
                attention = (attention * tok_mask).add_((tok_mask - 1).mul_(1e10))

            if entity_mask is not None:
                attention = (attention * entity_mask).add_((entity_mask - 1).mul_(1e10))

            top_soft_attention, _ = torch.max(attention, dim=1)

            return q, top_soft_attention

        else:

            return q, top_soft_attention
