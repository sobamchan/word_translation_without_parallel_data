import numpy as np
from gensim.models import KeyedVectors


def load_word_vec_dict(fpath):
    f = open(fpath, 'r')
    d = {}
    line = f.readline()
    line = f.readline()
    while line:
        items = line.split()
        word = items[0]
        elems = items[1:]
        # d[word] = np.array(elems, dtype=np.float64)
        d[word] = elems
        line = f.readline()
    f.close()
    return d


def load_word_vec_list(fpath):
    w2v = KeyedVectors.load_word2vec_format(fpath, binary=False)
    return w2v.vectors


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
