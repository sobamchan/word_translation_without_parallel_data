import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import utils
import model


class Trainer(object):

    def __init__(self, args):
        np.random.seed(0)

        self.args = args
        s_vec = utils.load_word_vec(args.source_vec_file)
        s_vocab, s_w2i, s_i2w = utils.dict2vocab(s_vec)
        self.s_vec = s_vec
        self.s_vocab = s_vocab
        self.s_w2i = s_w2i
        self.s_i2w = s_i2w
        t_vec = utils.load_word_vec(args.target_vec_file)
        t_vocab, t_w2i, t_i2w = utils.dict2vocab(t_vec)
        self.t_vec = t_vec
        self.t_vocab = t_vocab
        self.t_w2i = t_w2i
        self.t_i2w = t_i2w

        netD = model.netD()
        netG = model.netG()
        if args.use_cuda:
            netD = netD.cuda()
            netG = netG.cuda()
        self.netD = netD
        self.netG = netG
        self.optimizer_D = optim.Adam(self.netD.parameters(),
                                      lr=args.lr,
                                      betas=(args.beta1, 0.999))
        self.optimizer_G = optim.Adam(self.netG.parameters(),
                                      lr=args.lr,
                                      betas=(args.beta1, 0.999))
        self.criterion = nn.BCELoss()

    def train(self):
        args = self.args
        batch_size = args.batch_size
        s_words = np.array(list(self.s_vec.keys()))
        t_words = np.array(list(self.t_vec.keys()))
        if len(s_words) < len(t_words):
            max_indx = len(s_words)
        else:
            max_indx = len(t_words)

        for i_epoch in range(1, args.epoch + 1):
            s_perm_indxes = np.random.permutation(len(s_words))
            t_perm_indxes = np.random.permutation(len(t_words))

            for idx in range(0, max_indx, batch_size):
                ss = s_words[s_perm_indxes[idx:idx+batch_size]]
                ts = t_words[t_perm_indxes[idx:idx+batch_size]]
                D_G_z1, error_D = self.train_D((ss, ts))

    def train_D(self, batch):
        args = self.args
        ss, ts = batch

        s_vecs_np = np.array([np.array(self.s_vec[sw]).astype(np.float)
                              for sw in ss])
        t_vecs_np = np.array([np.array(self.t_vec[tw]).astype(np.float)
                              for tw in ts])

        batch_size = len(ss)

        self.netD.zero_grad()

        x = Variable(torch.from_numpy(t_vecs_np).type(torch.FloatTensor))
        label = Variable(torch.FloatTensor(batch_size).fill_(1))
        if args.use_cuda:
            x = x.cuda()
            label = label.cuda()
        output = self.netD(x)
        error_D_real = self.criterion(output, label)
        error_D_real.backward()

        # train with fake data (projected source data)
        x = Variable(torch.from_numpy(s_vecs_np).type(torch.FloatTensor))
        label = Variable(torch.FloatTensor(batch_size).fill_(0))
        if args.use_cuda:
            x = x.cuda()
            label = label.cuda()
        fake = self.netG(x)
        output = self.netD(fake.detach())
        error_D_fake = self.criterion(output, label)
        error_D_fake.backward()
        D_G_z1 = output.data.mean()

        self.optimizer_D.step()
        error_D = error_D_real + error_D_fake

        return D_G_z1, error_D.data[0]

    def train_G(self, batch):
        pass
