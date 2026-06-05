import torch
import torch.nn as nn
from Transformer_Scaled_dotproduct_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=300, num_heads=5, dropout=0.0, num_layers=1, tmp_num=0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final_2 = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.num_layers = num_layers
        self.tmp_num = tmp_num

    def forward(self, key, query, value, one_key, tok_mask=None, entity_mask=None):

        num_layers = self.num_layers
        tmp_num = self.tmp_num

        dim_per_head = self.dim_per_head
        batch_size = key.size(0)
        num_heads = self.num_heads

        # linear projection
        query = self.linear_q(query)
        key = self.linear_k(key)

        if tmp_num == num_layers - 1:

            key = key.view(batch_size * num_heads, -1, dim_per_head)
            query = query.view(batch_size * num_heads, -1, dim_per_head)
            value = value.view(batch_size * num_heads, -1, dim_per_head)
            one_key = one_key.view(batch_size * num_heads, -1, dim_per_head)

            # scaled dot product attention
            scale = (key.size(-1) // 25) ** -0.5
            output_2, ctx_attention = self.dot_product_attention(
            key, query, value, one_key, num_heads, scale, tok_mask, entity_mask, num_layers, tmp_num)

            return output_2, ctx_attention

        else:

            key = key.view(batch_size * num_heads, -1, dim_per_head)
            query = query.view(batch_size * num_heads, -1, dim_per_head)
            value = value.view(batch_size * num_heads, -1, dim_per_head)
            one_key = one_key.view(batch_size * num_heads, -1, dim_per_head)

            # scaled dot product attention
            scale = (key.size(-1) // 25) ** -0.5
            output_2, ctx_attention = self.dot_product_attention(
            key, query, value, one_key, num_heads, scale, tok_mask, entity_mask, num_layers, tmp_num)

            output_2 = self.linear_final_2(output_2)

            return output_2, ctx_attention
