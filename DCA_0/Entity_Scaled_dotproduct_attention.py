import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.linear_final = nn.Linear(300, 300)

    def forward(self, k, q, v, one_key, num_heads, scale=None, entity_mask=None, num_layers=1, tmp_num=0):

        attention = torch.mm(q, k.permute(1, 0))

        if scale is not None:
            attention = attention * scale

        if tmp_num == num_layers - 1:

            if entity_mask is not None:
                attention = (attention * entity_mask.repeat(num_heads, 1)).add_((entity_mask.repeat(num_heads, 1) - 1).mul_(1e10))

        else:

            if entity_mask is not None:
                attention = (attention * entity_mask.repeat(num_heads, 1)).add_((entity_mask.repeat(num_heads, 1) - 1).mul_(1e10))

        softmax_attetnion = self.softmax(attention)
        top_soft_attention, _ = torch.max(softmax_attetnion, dim=0, keepdim=True)
        context = torch.mm(top_soft_attention, one_key)
        sorce = torch.mm(v, context.permute(1, 0))
        q = v * sorce

        # if tmp_num != num_layers - 1:

        q = q.view(q.size(0) // num_heads, 300)
        q = self.linear_final(q)
        k = k.view(k.size(0) // num_heads, 300)

        if tmp_num == num_layers - 1:

            attention = torch.mm(q, k.permute(1, 0))

            if scale is not None:
                attention = attention * scale

            if entity_mask is not None:
                attention = (attention * entity_mask).add_((entity_mask - 1).mul_(1e10))

            top_soft_attention, _ = torch.max(attention, dim=0)

            return q, top_soft_attention

        else:

            return q, top_soft_attention
