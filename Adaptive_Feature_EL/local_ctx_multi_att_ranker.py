import torch
import torch.nn as nn
from DCA.abstract_word_entity import AbstractWordEntity
from abstract_word_entity import AbstractWordEntity
# from DCA.Transformer_Scaled_dotproduct_attention import ScaledDotProductAttention

class LocalCtxMultiAttRanker(AbstractWordEntity):
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
        self.dr = config['dr']  # 0.2

        self.score_combine_ = torch.nn.Sequential(
            torch.nn.Linear(2, self.hid_dims),  # 100 --hongbin
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dr),  # 0.2
            torch.nn.Linear(self.hid_dims, 1))  # 100

        '''self.encoder = Encoder()'''
        self.local_A = torch.nn.Parameter(torch.ones(self.emb_dims))
        self.local_B = torch.nn.Parameter(torch.ones(self.emb_dims))
        self.local_C = torch.nn.Parameter(torch.ones(self.emb_dims))
        # self.local_D = torch.nn.Parameter(torch.ones(self.emb_dims))

        self.local_score_combine = torch.nn.Sequential(
            torch.nn.Linear(2, self.hid_dims),  # 100 --hongbin
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dr),  # 0.2
            torch.nn.Linear(self.hid_dims, 2))  # 100

        self.local_softmax = nn.Softmax(dim=2)

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

        '''_, ctx_vecs = self.encoder(tok_vecs, entity_vecs, tok_mask, entity_mask.view(batchsize, -1, 1), tok_vecs)'''

        attention = torch.bmm(entity_vecs * self.local_A, tok_vecs.permute(0, 2, 1))
        if tok_mask is not None:
            attention = (attention * tok_mask).add_((tok_mask - 1).mul_(1e10))
        if entity_mask is not None:
            attention = (attention * entity_mask.view(batchsize, -1, 1)).add_((entity_mask.view(batchsize, -1, 1) - 1).mul_(1e10))
        top_soft_attention, _ = torch.max(attention, dim=1, keepdim=True)

        token_att = torch.bmm(tok_vecs * self.local_B, tok_vecs.permute(0, 2, 1))
        if tok_mask is not None:
            token_att = (token_att * tok_mask).add_((tok_mask - 1).mul_(1e10))
            token_att = (token_att * tok_mask.view(batchsize, -1, 1)).add_((tok_mask.view(batchsize, -1, 1) - 1).mul_(1e10))
        top_token_attention, _ = torch.max(token_att, dim=1, keepdim=True)

        inputs = torch.cat([top_soft_attention, top_token_attention], dim=1).permute(0, 2, 1)
        top_dual_attention = torch.sum(self.local_score_combine(inputs), dim=2, keepdim=True).permute(0, 2, 1)

        token_att_two = torch.bmm(tok_vecs * self.local_D, tok_vecs.permute(0, 2, 1))
        if tok_mask is not None:
            token_att_two = (token_att_two * tok_mask).add_((tok_mask - 1).mul_(1e10))
            token_att_two = (token_att_two * tok_mask.view(batchsize, -1, 1)).add_((tok_mask.view(batchsize, -1, 1) - 1).mul_(1e10))
        top_token_attention_two, _ = torch.max(token_att_two, dim=1, keepdim=True)
        att_probs_self = self.local_softmax(top_token_attention_two)

        top_tok_att_scores, top_tok_att_ids = torch.topk(top_dual_attention * att_probs_self, dim=2, k=min(self.tok_top_n, n_words))
        selected_tok_vecs = torch.gather(tok_vecs, dim=1, index=top_tok_att_ids.view(batchsize, -1, 1).repeat(1, 1, tok_vecs.size(2)))
        att_probs = self.local_softmax(top_tok_att_scores)
        dual_context = torch.bmm(att_probs, selected_tok_vecs * self.local_C).permute(0, 2, 1)

        ent_ctx_scores = torch.bmm(entity_vecs, dual_context)

        scores = ent_ctx_scores.view(batchsize, n_entities)
        scores = (scores * entity_mask).add_((entity_mask - 1).mul_(1e10))
        return scores

    def print_weight_norm(self):

        print('self.local_A', self.local_A.data.norm())
        print('self.local_B', self.local_B.data.norm())
        print('self.local_C', self.local_C.data.norm())
        # print('self.local_D', self.local_D.data.norm())

        # print('pre_f - l1.w, b', self.score_combine_[0].weight.data.norm(), self.score_combine_[0].bias.data.norm())
        # print('pre_f - l2.w, b', self.score_combine_[3].weight.data.norm(), self.score_combine_[3].bias.data.norm())

        # print('self.local_f - l1.w, b', self.local_score_combine[0].weight.data.norm(), self.local_score_combine[0].bias.data.norm())
        # print('self.local_f - l2.w, b', self.local_score_combine[3].weight.data.norm(), self.local_score_combine[3].bias.data.norm())

    # def local_regularize(self, max_norm=11):
    #     # super(MulRelRanker, self).regularize(max_norm)
    #     # print("----MulRelRanker Regularization----")
    #
    #     l1_w_norm = self.local_score_combine[0].weight.norm()
    #     l1_b_norm = self.local_score_combine[0].bias.norm()
    #     l2_w_norm = self.local_score_combine[3].weight.norm()
    #     l2_b_norm = self.local_score_combine[3].bias.norm()
    #
    #     if (l1_w_norm > max_norm).data.all():
    #         self.local_score_combine[0].weight.data = self.local_score_combine[0].weight.data * max_norm / l1_w_norm.data
    #     if (l1_b_norm > max_norm).data.all():
    #         self.local_score_combine[0].bias.data = self.local_score_combine[0].bias.data * max_norm / l1_b_norm.data
    #     if (l2_w_norm > max_norm).data.all():
    #         self.local_score_combine[3].weight.data = self.local_score_combine[3].weight.data * max_norm / l2_w_norm.data
    #     if (l2_b_norm > max_norm).data.all():
    #         self.local_score_combine[3].bias.data = self.local_score_combine[3].bias.data * max_norm / l2_b_norm.data




