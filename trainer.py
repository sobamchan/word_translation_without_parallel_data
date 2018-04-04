import os
import subprocess
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
from evaluator import Evaluator
from dictionary_utils import build_dictionary


class Trainer(object):

    def __init__(self, args):
        np.random.seed(args.seed)

        self.args = args

        self.logger = logger.Logger(args.output_dir)
        self.args.logger = self.logger

        current_commit_hash =\
            subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        self.logger.log('current git commit hash: %s' % current_commit_hash)

        print('load vec')
        source_vecs, source_dico =\
            utils.load_word_vec_list(args.source_vec_file, args.source_lang)
        target_vecs, target_dico =\
            utils.load_word_vec_list(args.target_vec_file, args.target_lang)

        self.src_dico = source_dico
        self.tgt_dico = target_dico
        args.src_dico = source_dico
        args.tgt_dico = target_dico

        src_embed, tgt_embed =\
            utils.get_embeds_from_numpy(source_vecs, target_vecs)
        if args.use_cuda:
            self.src_embed = src_embed.cuda()
            self.tgt_embed = tgt_embed.cuda()
        else:
            self.src_embed = src_embed
            self.tgt_embed = tgt_embed

        print('setting models')
        netD = model.netD(self.args)
        netG = model.netG()
        netG.W.weight.data.copy_(torch.diag(torch.ones(300)))
        if args.multi_gpu:
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
        self.prefix = os.path.basename(args.output_dir)

        self.evaluator = Evaluator(self)

    def train(self):
        args = self.args

        for i_epoch in range(1, args.epoch + 1):

            error_D_list = []
            error_G_list = []

            for niter in tqdm(range(10000)):

                for _ in range(args.dis_step):
                    error_D = self.train_D()
                    error_D_list.append(error_D)

                error_G = self.train_G()

                error_G_list.append(error_G)

                if niter % 500 == 0:
                    print('error_D: ', np.mean(error_D_list))
                    print('error_G: ', np.mean(error_G_list))
                    if args.use_criteria:
                        print('dist cosine mean: ',
                              self.evaluator.dist_mean_cosine())
                        print('caluclating word translation accuracy...')
                        self.evaluator.word_translation()

            result_ = {'epoch': i_epoch,
                       'error_D': np.mean(error_D_list),
                       'error_G': np.mean(error_G_list)}
            self.logger.dump(result_)
            if i_epoch % args.log_inter == 0:
                progress_path = os.path.join(args.output_dir,
                                             'progress.json')
                imgpaths = slack_utils.output_progress(progress_path,
                                                       args.output_dir,
                                                       self.prefix)
                for imgpath in imgpaths:
                    slack_utils.send_slack_img(imgpath)

    def get_batch_for_disc(self, volatile):
        args = self.args
        batch_size = args.batch_size
        src_ids = torch.LongTensor(batch_size).random_(75000)
        tgt_ids = torch.LongTensor(batch_size).random_(75000)
        if args.use_cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()
        src_embed = self.src_embed(Variable(src_ids, volatile=True))
        tgt_embed = self.tgt_embed(Variable(tgt_ids, volatile=True))
        src_embed = self.netG(Variable(src_embed.data, volatile=volatile))
        tgt_embed = Variable(tgt_embed.data, volatile=volatile)

        x = torch.cat([src_embed, tgt_embed], 0)
        y = torch.FloatTensor(2 * batch_size).zero_()
        y[:batch_size] = 1 - 0.1
        y[batch_size:] = 0.1
        y = Variable(y.cuda() if args.use_cuda else y)
        return x, y

    def train_D(self):
        self.netD.train()
        self.netG.eval()

        x, y = self.get_batch_for_disc(volatile=True)
        preds = self.netD(Variable(x.data))
        loss = self.criterion(preds, y)
        self.optimizer_D.zero_grad()
        loss.backward()
        self.optimizer_D.step()
        return loss.data[0]

    def train_G(self):
        self.netD.eval()
        self.netG.train()

        x, y = self.get_batch_for_disc(volatile=False)
        preds = self.netD(x)
        loss = self.criterion(preds, 1 - y)

        self.optimizer_G.zero_grad()
        loss.backward()
        self.optimizer_G.step()
        self.orthogonalize()

        return loss.data[0]

    def build_dictionary(self):
        src_emb = self.netG(self.src_embed.weight).data
        tgt_emb = self.tgt_embed.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self.dico = build_dictionary(src_emb, tgt_emb, self.args)

    def orthogonalize(self):
        if self.args.map_beta > 0:
            W = self.netG.W.weight.data
            beta = self.args.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def save_netG_state(self):
        multi_gpu = self.arg.multi_gpus
        odir = self.args.output_dir
        fpath = os.path.join(odir, 'netG_state.pth')
        self.logger.log('saving netG state to', fpath)
        if multi_gpu:
            state_dict = self.netG.module.state_dict()
        else:
            state_dict = self.netG.state_dict()
        torch.save(state_dict, fpath)
