# import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class netD(nn.Module):

    def __init__(self):
        super(netD, self).__init__()

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(300, 2048)
        self.leaky_relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()

        weights_init(self)

    def main(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def forward(self, x):
        return self.main(x)


class netG(nn.Module):

    def __init__(self):
        super(netG, self).__init__()
        self.W = nn.Linear(300, 300, bias=False)

        weights_init(self)

    def main(self, x):
        x = self.W(x)
        return x

    def forward(self, x):
        return self.main(x)
