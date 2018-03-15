import torch.nn as nn


class netD(nn.Module):

    def __init__(self):
        super(netD, self).__init__()

        self.fc1 = nn.Linear(300, 2048)
        self.leaky_relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class netG(nn.Module):

    def __init__(self):
        super(netG, self).__init__()
        self.W = nn.Linear(300, 300, bias=False)

    def forward(self, x):
        x = self.W(x)
        return x
