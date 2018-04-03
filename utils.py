import os
import json
import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from dictionary import Dictionary


def load_word_vec_list(fpath, lang):
    wlistpath = os.path.join(os.path.split(fpath), '.wordlist')
    with open(wlistpath, 'r') as f:
        wlist = json.load(f)
    w2i = {}
    for w in wlist:
        if w not in w2i.keys():
            w2i[w] = len(w2i)
    i2w = {v: k for k, v in w2i.items()}
    dico = Dictionary(i2w, w2i, lang)

    if fpath.endswith('.npy'):
        vec = np.load(fpath)
    else:
        vec = KeyedVectors.load_word2vec_format(fpath, binary=False).vectors
    return vec, dico


def dict2vocab(d):
    vocab = list(d.keys())
    w2i = {}
    for w in vocab:
        if w not in w2i.keys():
            w2i[w] = len(w2i)
    i2w = {v: k for k, v in w2i.items()}
    return vocab, w2i, i2w


def read_file_batch_by_line_idx(fpath, indexes):
    f = open(fpath, 'r')
    next(f)
    line = f.readline()
    d = {}
    idx = 0
    while line:
        if idx in indexes:
            items = line.strip().split(' ')
            word = items[0]
            elems = items[1:]
            d[word] = np.array(elems, dtype=np.float64)
        idx += 1
        line = f.readline()
    return d


def get_file_line_n(fpath):
    f = open(fpath, 'r')
    line = f.readline()
    line = f.readline()
    line_n = 0
    while line:
        line = f.readline()
        line_n += 1
    return line_n


def sample_real_smooth_label(batch_size):
    labels = np.array([np.random.choice(np.arange(0.7, 1.2, 0.1))
                       for _ in range(batch_size)])
    return labels.reshape(batch_size, 1)


def sample_fake_smooth_label(batch_size):
    labels = np.array([np.random.choice(np.arange(0.0, 0.3, 0.1))
                       for _ in range(batch_size)])
    return labels.reshape(batch_size, 1)


def get_embeds_from_numpy(src_vecs, tgt_vecs):
    if isinstance(src_vecs, np.ndarray):
        src_vecs = torch.from_numpy(src_vecs)
    if isinstance(tgt_vecs, np.ndarray):
        tgt_vecs = torch.from_numpy(tgt_vecs)
    src_embed = nn.Embedding(len(src_vecs), 300)
    tgt_embed = nn.Embedding(len(tgt_vecs), 300)
    src_embed.weight.data.copy_(src_vecs)
    tgt_embed.weight.data.copy_(tgt_vecs)
    return src_embed, tgt_embed


def get_nn_avg_dist(emb, query, knn):
    bs = 1024
    all_distances = []
    emb = emb.transpose(0, 1).contiguous()
    for i in range(0, query.shape[0], bs):
        distances = query[i:i + bs].mm(emb)
        best_distances, _ = distances.topk(knn,
                                           dim=1,
                                           largest=True,
                                           sorted=True)
        all_distances.append(best_distances.mean(1).cpu())
    all_distances = torch.cat(all_distances)
    return all_distances.cpu()
