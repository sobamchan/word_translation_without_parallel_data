# import torch
import torch.nn as nn


def build_dictionary(emb1, emb2, params):
    '''
    in:
      emb1, emb2: FloatTensor
    out:
      dico: dictionary emb1 -> emb2
            {emb1_widx: [emb2_widx * k]}
    '''


def test_build_dictionary():
    emb1 = nn.Embedding(30, 10)
    emb2 = nn.Embedding(30, 10)
    build_dictionary(emb1, emb2)
