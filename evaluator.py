from copy import deepcopy
from dictionary_utils import get_candidates, build_dictionary


class Evaluator(object):

    def __init__(self, trainer):
        self.src_embed = trainer.src_embed
        self.tgt_embed = trainer.tgt_embed
        self.netG = trainer.netG
        self.netD = trainer.netD
        self.args = trainer.args
        self.logger = trainer.logger

    def dist_mean_cosine(self):
        args = self.args

        src_emb = self.netG(self.src_embed.weight).data
        tgt_emb = self.tgt_embed.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

        for dico_method in ['nn', 'csls_knn_10']:
            dico_build = 'S2T'
            dico_max_size = 1000
            _args = deepcopy(args)
            _args.dico_method = dico_method
            _args.dico_build = dico_build
            _args.dico_threshold = 0
            _args.dico_max_rank = 10000
            _args.dico_min_size = 0
            _args.dico_max_size = dico_max_size
            s2t_candidates = get_candidates(src_emb, tgt_emb, _args)
            t2s_candidates = get_candidates(tgt_emb, src_emb, _args)
            dico = build_dictionary(src_emb,
                                    tgt_emb,
                                    _args,
                                    s2t_candidates,
                                    t2s_candidates)

            if dico is None:
                mean_cosine = -1e9
            else:
                mean_cosine = (src_emb[dico[:dico_max_size, 0]] *
                               tgt_emb[dico[:dico_max_size, 1]]).sum(1).mean()

            return mean_cosine
