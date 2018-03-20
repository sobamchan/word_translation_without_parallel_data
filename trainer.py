import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import utils
import slack_utils
import model
import logger


class Trainer(object):

    def __init__(self, args):
        np.random.seed(0)

        self.args = args

        self.logger = logger.Logger(args.output_dir)

        print('load source vec')
        self.source_vecs = utils.load_word_vec_list(args.source_vec_file)
        print('load target vec')
        self.target_vecs = utils.load_word_vec_list(args.target_vec_file)

        print('setting models')
        netD = model.netD()
        netG = model.netG()
        if args.multi_gpus:
            netD = nn.DataParallel(netD)
            netG = nn.DataParallel(netG)
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

        # if self.source_vocab_n > self.target_vocab_n:
        #     max_indx = self.target_vocab_n
        # else:
        #     max_indx = self.source_vocab_n

        for i_epoch in range(1, args.epoch + 1):
            s_200k_perm_indxes = np.random.permutation(200000)
            s_50k_perm_indxes = np.random.permutation(50000)
            t_50k_perm_indxes = np.random.permutation(50000)

            D_G_z1_list = []
            D_G_z2_list = []
            error_D_list = []
            error_G_list = []
            total = 50000
            for idx in tqdm(range(0, total, batch_size),
                            total=int(total/batch_size)):
                s_vecs = self.source_vecs[
                        s_50k_perm_indxes[idx:idx+batch_size]]
                t_vecs = self.target_vecs[
                        t_50k_perm_indxes[idx:idx+batch_size]]
                D_G_z1, error_D = self.train_D((s_vecs, t_vecs))

                s_vecs = self.source_vecs[
                        s_200k_perm_indxes[idx:idx+batch_size]]
                D_G_z2, error_G = self.train_G(s_vecs)

                D_G_z1_list.append(D_G_z1)
                D_G_z2_list.append(D_G_z2)
                error_D_list.append(error_D)
                error_G_list.append(error_G)
            print('%d th epoch: loss d: %.4f, loss g: %.4f' %
                  (i_epoch,
                   np.mean(error_D_list),
                   np.mean(error_G_list)))

            result_ = {'error_D': np.mean(error_D_list),
                       'error_G': np.mean(error_G_list)}
            self.logger.dump(result_)
            if i_epoch % 10 == 0:
                progress_path = os.path.join(args.output_dir,
                                             'progress.json')
                imgpaths = slack_utils.output_progress(progress_path,
                                                       args.output_dir)
                for imgpath in imgpaths:
                    slack_utils.send_slack_img(imgpath)

    def train_D(self, batch):
        args = self.args
        s_vecs, t_vecs = batch

        batch_size = len(s_vecs)

        self.netD.zero_grad()

        # train with real data
        x = Variable(torch.from_numpy(t_vecs).type(torch.FloatTensor))
        # label = Variable(torch.FloatTensor(batch_size).fill_(1)).unsqueeze(1)
        label = utils.sample_real_smooth_label(batch_size)
        label = Variable(torch.from_numpy(label).type(torch.FloatTensor))
        if args.use_cuda:
            x = x.cuda()
            label = label.cuda()
        output = self.netD(x)
        error_D_real = self.criterion(output, label)
        error_D_real.backward()

        # train with fake data (projected source data)
        x = Variable(torch.from_numpy(s_vecs).type(torch.FloatTensor))
        # label = Variable(torch.FloatTensor(batch_size).fill_(0)).unsqueeze(1)
        label = utils.sample_fake_smooth_label(batch_size)
        label = Variable(torch.from_numpy(label).type(torch.FloatTensor))
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

    def train_G(self, s_vecs):
        args = self.args

        batch_size = len(s_vecs)

        self.netG.zero_grad()
        x = Variable(torch.from_numpy(s_vecs).type(torch.FloatTensor))
        # label = Variable(torch.FloatTensor(batch_size).fill_(1)).unsqueeze(1)
        label = utils.sample_real_smooth_label(batch_size)
        label = Variable(torch.from_numpy(label).type(torch.FloatTensor))
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

    def convert(self, s_vecs):
        args = self.args
        # batch_size = len(s_vecs)
        self.netG.train()
        x = Variable(torch.from_numpy(s_vecs).type(torch.FloatTensor))
        if args.use_cuda:
            x = x.cuda()
            # label = label.cuda()
        results = self.netG(x)

        return results
