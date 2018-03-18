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
        self.source_vocab_n = utils.get_file_line_n(args.source_vec_file)
        self.target_vocab_n = utils.get_file_line_n(args.target_vec_file)

        print('setting models')
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
        s_fpath = args.source_vec_file
        t_fpath = args.target_vec_file
        batch_size = args.batch_size

        if self.source_vocab_n > self.target_vocab_n:
            max_indx = self.target_vocab_n
        else:
            max_indx = self.source_vocab_n

        for i_epoch in range(1, args.epoch + 1):
            s_perm_indxes = np.random.permutation(self.source_vocab_n)
            t_perm_indxes = np.random.permutation(self.target_vocab_n)

            D_G_z1_list = []
            D_G_z2_list = []
            error_D_list = []
            error_G_list = []
            for idx in range(0, max_indx, batch_size):
                s_inds = s_perm_indxes[idx:idx+batch_size]
                t_inds = t_perm_indxes[idx:idx+batch_size]
                s_d = utils.read_file_batch_by_line_idx(s_fpath, s_inds)
                t_d = utils.read_file_batch_by_line_idx(t_fpath, t_inds)

                D_G_z1, error_D = self.train_D((s_d, t_d))
                D_G_z2, error_G = self.train_G((s_d, t_d))

                D_G_z1_list.append(D_G_z1)
                D_G_z2_list.append(D_G_z2)
                error_D_list.append(error_D)
                error_G_list.append(error_G)
            print('%d th epoch: loss d: %.4f, loss g: %.4f' %
                  (i_epoch,
                   np.mean(error_D_list),
                   np.mean(error_G_list)))

    def train_D(self, batch):
        args = self.args
        s_d, t_d = batch
        s_vecs_np = np.array(list(s_d.values()))
        t_vecs_np = np.array(list(t_d.values()))

        batch_size = len(s_d)

        self.netD.zero_grad()

        # train with real data
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
        args = self.args
        s_d, _ = batch

        s_vecs_np = np.array(list(s_d.values()))

        batch_size = len(s_d)

        self.netG.zero_grad()
        x = Variable(torch.from_numpy(s_vecs_np).type(torch.FloatTensor))
        label = Variable(torch.FloatTensor(batch_size).fill_(1))
        if args.use_cuda:
            x = x.cuda()
            label = label.cuda()
        fake = self.netG(x)
        output = self.netD(fake)
        error_D_fake = self.criterion(output, label)
        error_D_fake.backward()
        D_G_z2 = output.data.mean()
        self.optimizer_G.step()

        return D_G_z2, error_D_fake.data[0]
