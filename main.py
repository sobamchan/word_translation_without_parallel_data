import argparse
import torch
from trainer import Trainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--use-cuda', type=bool, default=True)
    parser.add_argument('--multi-gpus', type=bool, default=True)
    parser.add_argument('--source-vec-file',
                        type=str,
                        default='./data/wiki.ja.vec.200k')
    parser.add_argument('--target-vec-file',
                        type=str,
                        default='./data/wiki.en.vec.200k')
    parser.add_argument('--output-dir',
                        type=str,
                        default='test')
    return parser.parse_args()


def main(args):
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    args = get_args()
    if args.use_cuda and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False
    main(args)
