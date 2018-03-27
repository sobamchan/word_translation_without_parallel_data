# import torch
import torch.nn as nn
import torch.nn.functional as F


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

        layers = [nn.Dropout((0.1))]
        for i in range(3):
            input_dim = 300 if i == 0 else 2048
            output_dim = 1 if i == 2 else 2048
            layers.append(nn.Linear(input_dim, output_dim))
            if i < 2:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(0.1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


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


class complexNetG(nn.Module):

    def __init__(self):
        super(complexNetG, self).__init__()
        self.fc1 = nn.Linear(300, 400)
        self.fc2 = nn.Linear(400, 300)

    def main(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x):
        return self.main(x)
