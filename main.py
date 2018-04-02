import argparse
from distutils.util import strtobool
import torch
from trainer import Trainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--map-beta', type=float, default=0.001)
    parser.add_argument('--dis-input-dropout', type=float, default=0.1)
    parser.add_argument('--dis-dropout', type=float, default=0.0)
    parser.add_argument('--dis-step', type=int, default=5)
    parser.add_argument('--use-criteria', type=strtobool, default='1')
    parser.add_argument('--use-cuda', type=strtobool, default='1')
    parser.add_argument('--multi-gpu', type=strtobool, default='1')
    parser.add_argument('--source-vec-file',
                        type=str,
                        default='./data/wiki.en.vec.200k.npy')
    parser.add_argument('--target-vec-file',
                        type=str,
                        default='./data/wiki.es.vec.200k.npy')
    parser.add_argument('--output-dir',
                        type=str,
                        default='./lab/test')
    parser.add_argument('--log-inter',
                        type=int,
                        default=10)
    parser.add_argument('--slack-output', type=bool, default=True)
    return parser.parse_args()


def main(args):
    trainer = Trainer(args)
    trainer.train()
    trainer.save_netG_state()


if __name__ == '__main__':
    args = get_args()
    if args.use_cuda and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False
    main(args)
