import torch
import torch.nn as nn
import torch.nn.functional as F
from DCA_0.abstract_word_entity import AbstractWordEntity


class NTEE(AbstractWordEntity):
    """
    NTEE model, proposed in Yamada et al. "Learning Distributed Representations of Texts and Entities from Knowledge Base"

    hongbin -- select candidates in prerank
    """

    def __init__(self, config):
        print('--- create NTEE model ---')

        # integrate the look-up table into embedding
        config['word_embeddings_class'] = nn.EmbeddingBag
        # input is list of index, output is embedding
        config['entity_embeddings_class'] = nn.Embedding
        super(NTEE, self).__init__(config)
        # matrix multiplication
        self.linear = nn.Linear(self.emb_dims, self.emb_dims)

    # token_ids: list(lctx_ids, ment_ids(no exist), rctx_ids, ...)
    # token_offsets: list(0 0 + len(list(lctx_ids + ment_ids + rctx_ids)) ...)
    # use_sum: True
    # https://blog.csdn.net/shunaoxi2313/article/details/103115260 gain result of interation
    def compute_sent_vecs(self, token_ids, token_offsets, use_sum=False):
        # shape(len(token_offsets),300)
        sum_vecs = self.word_embeddings(token_ids, token_offsets)
        # True
        if use_sum:
            return sum_vecs

        sum_vecs = F.normalize(sum_vecs)
        sent_vecs = self.linear(sum_vecs)
        return sent_vecs

    # token_ids: list(lctx_ids, ment_ids(no exist), rctx_ids, ...)
    # token_offsets: list(0 0 + len(list(lctx_ids + ment_ids + rctx_ids)) ...)
    # entity_ids: list(list(cands))
    # use_sum: True
    def forward(self, token_ids, token_offsets, entity_ids, use_sum=False):
        # 2D shape(len(token_offsets), 300)
        sent_vecs = self.compute_sent_vecs(token_ids, token_offsets, use_sum)
        # 3D shape(len(entity_ids), len(cands), 300)
        entity_vecs = self.entity_embeddings(entity_ids)

        # compute scores
        batchsize, dims = sent_vecs.size()
        n_entities = entity_vecs.size(1)
        # torch.bmm: tensor multiplication -> (batchsize, n_entities, 1)
        # https://blog.csdn.net/foneone/article/details/103876519 input（p,m,n) * mat2(p,n,a) ->output(p,m,a)
        scores = torch.bmm(entity_vecs, sent_vecs.view(batchsize, dims, 1)).view(batchsize, n_entities)
        # dim=1 represent according to row to compute
        log_probs = F.log_softmax(scores, dim=1)
        # shape(sent_vecs.size(0), entity_vecs.size(1))
        return log_probs

    def predict(self, token_ids, token_offsets, entity_ids, gold_entity_ids=None):
        log_probs = self.forward(token_ids, token_offsets, entity_ids)
        _, pred_entity_ids = torch.max(log_probs, dim=1)

        acc = None
        if gold_entity_ids is not None:
            acc = torch.eq(gold_entity_ids, pred_entity_ids).sum()
        return pred_entity_ids, acc

    def loss(self, log_probs, true_pos):
        return F.nll_loss(log_probs, true_pos)
