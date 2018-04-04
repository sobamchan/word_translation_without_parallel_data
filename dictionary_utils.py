import torch
from utils import get_nn_avg_dist


def get_candidates(emb1, emb2, args):
    bs = 128

    all_scores = []
    all_targets = []
    n_src = emb1.size(0)
    if args.dico_method == 'nn':

        for i in range(0, n_src, bs):
            scores = emb2.mm(emb1[i:min(n_src, i + bs)]
                             .transpose(0, 1)).transpose(0, 1)
            best_scores, best_targets = scores.topk(2,
                                                    dim=1,
                                                    largest=True,
                                                    sorted=True)
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        # all_scores: for each emb1 vocab, contains top-2 "close" word's scores
        all_scores = torch.cat(all_scores, 0)
        # all_targets:
        #   for each emb1 vocab, contains top-2 "close" word's indices
        all_targets = torch.cat(all_targets, 0)

    if args.dico_method.startswith('csls_knn_'):
        knn = args.dico_method[len('csls_knn_'):]
        knn = int(knn)

        ave_dist1 = torch.from_numpy(get_nn_avg_dist(emb2, emb1, knn))
        ave_dist2 = torch.from_numpy(get_nn_avg_dist(emb1, emb2, knn))
        ave_dist1 = ave_dist1.type_as(emb1)
        ave_dist2 = ave_dist2.type_as(emb2)

        for i in range(0, n_src, bs):
            scores = emb2.mm((emb1[i:min(n_src, i + bs)])
                             .transpose(0, 1)).transpose(0, 1)
            scores.mul_(2)

            scores.sub_(ave_dist1[i:min(n_src, i + bs)][:, None]
                        + ave_dist2[None, :])

            best_scores, best_targets =\
                scores.topk(2, dim=1, largest=True, sorted=True)

            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    all_pairs = torch.cat([
        torch.arange(0, all_targets.size(0)).long().unsqueeze(1),
        all_targets[:, 0].unsqueeze(1)
        ], 1)

    assert all_scores.size() == all_pairs.size() == (n_src, 2)

    # sort paris by score confidence
    diff = all_scores[:, 0] - all_scores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    all_scores = all_scores[reordered]
    all_pairs = all_pairs[reordered]

    if args.dico_max_rank > 0:
        selected = all_pairs.max(1)[0] <= args.dico_max_rank
        mask = selected.unsqueeze(1).expand_as(all_scores).clone()
        all_scores = all_scores.masked_select(mask).view(-1, 2)
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    if args.dico_max_size > 0:
        all_scores = all_scores[:args.dico_max_size]
        all_pairs = all_pairs[:args.dico_max_size]

    diff = all_scores[:, 0] - all_scores[:, 1]
    if args.dico_min_size > 0:
        diff[:args.dico_min_size] = 1e9

    if args.dico_threshold > 0:
        mask = diff > args.dico_threshold
        mask = mask.unsqueeze(1).expand_as(all_pairs).clone()
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    return all_pairs


def build_dictionary(src_emb, tgt_emb, args,
                     s2t_candidates=None, t2s_candidates=None):
    s2t = 'S2T' in args.dico_build
    t2s = 'T2S' in args.dico_build

    if s2t:
        if s2t_candidates is None:
            s2t_candidates = get_candidates(src_emb, tgt_emb, args)

    if t2s:
        if t2s_candidates is None:
            t2s_candidates = get_candidates(src_emb, tgt_emb, args)
        t2s_candidates =\
            torch.cat([t2s_candidates[:, 1:], t2s_candidates[:, :1]], 1)

    if args.dico_build == 'S2T':
        dico = s2t_candidates
    elif args.dico_build == 'T2S':
        dico = t2s_candidates
    else:
        s2t_candidates = set([(a, b) for a, b in s2t_candidates])
        t2s_candidates = set([(a, b) for a, b in t2s_candidates])
        if args.dico_build == 'S2T|T2S':
            final_pairs = s2t_candidates | t2s_candidates
        else:
            assert args.dico_build == 'S2T&T2S'
            final_pairs = s2t_candidates & t2s_candidates
            if len(final_pairs) == 0:
                print("Empty intersection ...")
                return None
        dico = torch.LongTensor(list([[a, b] for (a, b) in final_pairs]))

    return dico.cuda() if args.use_cuda else dico
