import torch
import torch.nn as nn
from DCA_0.abstract_word_entity import AbstractWordEntity
from DCA_0.Transformer_Scaled_dotproduct_attention import ScaledDotProductAttention
from Transformer_Encoder import Encoder
import torch.nn.functional as F


class LocalCtxMultiAttRanker(AbstractWordEntity, ScaledDotProductAttention):
    """
    local model with context token attention (from G&H's EMNLP paper)
    """

    def __init__(self, config):
        print('--- create LocalCtxAttRanker model ---')

        config['word_embeddings_class'] = nn.Embedding # (self.word_voca.size(), self.emb_dims)
        config['entity_embeddings_class'] = nn.Embedding # (self.entity_voca.size(), self.emb_dims)
        super(LocalCtxMultiAttRanker, self).__init__(config)

        self.hid_dims = config['hid_dims']  # 100
        self.tok_top_n = config['tok_top_n']  # 50 number of top contextual words for the local model
        self.margin = config['margin']  # 0.01

        self.encoder = Encoder()
        self.tok_score_mat_diag = nn.Parameter(torch.ones(self.emb_dims))

        self.local_ctx_dr = nn.Dropout(p=0.0)

    def forward(self, token_ids, tok_mask, entity_ids, entity_mask, p_e_m=None):
        # batchsize: len(list(mentions)), n_words: len(list(lctx_ids + rctx_ids))
        batchsize, n_words = token_ids.size()
        # 8
        n_entities = entity_ids.size(1)
        # (len(list(mentions)), 1, list(lctx_ids + rctx_ids))
        tok_mask = tok_mask.view(batchsize, 1, -1)

        # tok_vecs == context_emb.shape(len(mention), len(list(lctx_ids + rctx_ids)), emb_dims)
        tok_vecs = self.word_embeddings(token_ids)
        # entity_vecs.shape(len(mention), len(list(cands_id)), emb_dims)
        entity_vecs = self.entity_embeddings(entity_ids)

        _, selfattention = self.encoder(tok_vecs, entity_vecs, tok_vecs, tok_mask, entity_mask.view(batchsize, -1, 1))

        top_tok_att_scores, top_tok_att_ids = torch.topk(selfattention, dim=1, k=min(self.tok_top_n, n_words))
        att_probs = F.softmax(top_tok_att_scores, dim=1).view(batchsize, -1, 1)
        selected_tok_vecs = torch.gather(tok_vecs, dim=1, index=top_tok_att_ids.view(batchsize, -1, 1).repeat(1, 1, tok_vecs.size(2)))

        ctx_vecs = torch.sum((selected_tok_vecs * self.tok_score_mat_diag) * att_probs, dim=1, keepdim=True)
        ctx_vecs = self.local_ctx_dr(ctx_vecs)

        ent_ctx_scores = torch.bmm(entity_vecs, ctx_vecs.permute(0, 2, 1))
        scores = ent_ctx_scores.view(batchsize, n_entities)
        scores = (scores * entity_mask).add_((entity_mask - 1).mul_(1e10))
        return scores

    def print_weight_norm(self):
        print('tok_score_mat_diag', self.tok_score_mat_diag.data.norm())



