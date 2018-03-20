import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import model
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir',
                        type=str,
                        default='./lab/test')
    parser.add_argument('--source-vec-file',
                        type=str,
                        default='./data/wiki.ja.vec.200k')
    parser.add_argument('--target-vec-file',
                        type=str,
                        default='./data/wiki.en.vec.200k')
    parser.add_argument('--source-word',
                        type=str,
                        default='男')
    return parser.parse_args()


def main(args):

    # load model
    netG = model.netG()
    load_path = os.path.join(args.load_dir, 'netG_state.pth')
    netG.load_state_dict(torch.load(load_path))

    source_w2v = KeyedVectors.load_word2vec_format(args.source_vec_file,
                                                   binary=False)
    target_w2v = KeyedVectors.load_word2vec_format(args.target_vec_file,
                                                   binary=False)
    target_vocab = list(target_w2v.vocab.keys())

    # get target vec
    source_vec = source_w2v.get_vector(args.source_word)
    source_vec = Variable(torch.from_numpy(source_vec).type(torch.FloatTensor))

    # conver source vector
    transfered_source_vec = netG(source_vec).data[0]

    # search most similar
    target_vocab_n, embed_n = target_w2v.vectors.shape
    distances = []
    for idx in range(target_vocab_n):
        distance = 1 - cosine(target_w2v.vectors[idx, :],
                              transfered_source_vec)
        distances.append(distance)
    top_n_indexes = list(reversed(np.argsort(distances)[-5:]))
    for i in top_n_indexes:
        print(target_vocab[i])


if __name__ == '__main__':
    args = get_args()
    main(args)
