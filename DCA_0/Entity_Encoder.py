import torch.nn as nn
from Entity_Multi_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    """Encoder的一层。"""

    def __init__(self, model_dim=300, num_heads=1, dropout=0.0, num_layers=1, tmp_num=0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout, num_layers, tmp_num)

    def forward(self, cumulative_entity_vecs, entity_vecs, entity_vecs_, entity_mask=None):
        # self attention
        output_2, ctx_attention = self.attention(cumulative_entity_vecs, entity_vecs, entity_vecs_, entity_mask)

        return output_2, ctx_attention


class Encoder_entity(nn.Module):
    """多层EncoderLayer组成Encoder。"""

    def __init__(self,
                 # vocab_size,
                 # max_seq_len,
                 num_layers=4,
                 model_dim=300,
                 num_heads=4,
                 dropout=0.0):
        super(Encoder_entity, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, dropout, num_layers, tmp_num) for tmp_num in
             range(num_layers)])

    # , inputs_len, inputs_len_nomask
    def forward(self, cumulative_entity_vecs, entity_vecs, entity_mask):
        output_2 = entity_vecs

        for encoder in self.encoder_layers:
            # one layer
            output_2, ctx_attention = encoder(cumulative_entity_vecs, output_2, entity_vecs, entity_mask)

        return output_2, ctx_attention
