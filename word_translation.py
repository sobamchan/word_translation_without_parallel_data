import io
import numpy as np
import torch
from dictionary_utils import get_nn_avg_dist


def load_dictionary(path, w2i1, w2i2, args):
    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with io.open(path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            assert line == line.lower()
            word1, word2 = line.rstrip().split()
            if word1 in w2i1 and word2 in w2i2:
                pairs.append((word1, word2))
            else:
                not_found += 1
                not_found1 += int(word1 not in w2i1)
                not_found2 += int(word2 not in w2i2)

    pairs = sorted(pairs, key=lambda x: w2i1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = w2i1[word1]
        dico[i, 1] = w2i2[word2]
    return dico


def get_word_translation_accuracy(lang1, w2i1, emb1, lang2,
                                  w2i2, emb2, method, args):
    logger = args.logger
    path = './data/%s-%s.5000-6500.txt' % (lang1, lang2)
    dico = load_dictionary(path, w2i1, w2i2, args)
    dico = dico.cuda() if emb1.is_cuda else dico

    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

    if method == 'nn':
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))

    elif method.startswith('csls_knn_'):
        knn = method[len('csls_knn_'):]
        knn = int(knn)
        ave_dist1 = get_nn_avg_dist(emb2, emb1, knn)
        ave_dist2 = get_nn_avg_dist(emb1, emb2, knn)
        ave_dist1 = torch.from_numpy(ave_dist1).type_as(emb1)
        ave_dist2 = torch.from_numpy(ave_dist2).type_as(emb2)
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(ave_dist1[dico[:, 0]][:, None] + ave_dist2[None, :])

    results = []
    top_matches = scores.topk(100, 1, True)[1]
    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
        _matching = (top_k_matches == dico[:, 1][:, None]
                     .expand_as(top_k_matches)).sum(1)
        matching = {}
        for i, src_id in enumerate(dico[:, 0]):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        precision_at_k = 100 * np.mean(list(matching.values()))
        logger.log('%i source words - %s - Precision at k = %i: %f' %
                   (len(matching), method, k, precision_at_k))
        results.append({'precision_at_%i' % k: precision_at_k})

    return results
