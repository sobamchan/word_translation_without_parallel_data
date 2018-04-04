from copy import deepcopy
from dictionary_utils import get_candidates, build_dictionary
from word_translation import get_word_translation_accuracy


class Evaluator(object):

    def __init__(self, trainer):
        self.src_embed = trainer.src_embed
        self.tgt_embed = trainer.tgt_embed
        self.netG = trainer.netG
        self.netD = trainer.netD
        self.src_dico = trainer.src_dico
        self.tgt_dico = trainer.tgt_dico
        self.logger = trainer.logger
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

    def word_translation(self):
        src_emb = self.netG(self.src_embed.weight).data
        tgt_emb = self.tgt_embed.weight.data

        for method in ['nn', 'csls_knn_10']:
            results = get_word_translation_accuracy(
                    self.src_dico.lang, self.src_dico.w2i, src_emb,
                    self.tgt_dico.lang, self.tgt_dico.w2i, tgt_emb,
                    method=method,
                    args=self.args
                    )
            for r in results:
                self.logger.dump(r)
