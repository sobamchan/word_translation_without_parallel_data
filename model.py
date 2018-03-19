# import torch
import torch.nn as nn


class netD(nn.Module):

    def __init__(self, ngpu=1):
        super(netD, self).__init__()
        self.ngpu = ngpu

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(300, 2048)
        self.leaky_relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()

    def main(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def forward(self, x):
        return self.main(x)
        # if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        #     x = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        # else:
        #     x = self.main(x)
        # return x


class netG(nn.Module):

    def __init__(self, ngpu=1):
        super(netG, self).__init__()
        self.ngpu = ngpu
        self.W = nn.Linear(300, 300, bias=False)

    def main(self, x):
        x = self.W(x)
        return x

    def forward(self, x):
        return self.main(x)
        # if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        #     x = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        # else:
        #     x = self.main(x)
        # return x
