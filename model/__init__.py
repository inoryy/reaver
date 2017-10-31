import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from torch.autograd import Variable


class CNNBlock(nn.Module):
    def __init__(self, cat_channels_in, cat_channels_out, num_channels):
        super().__init__()
        assert cat_channels_out + num_channels > 0
        if cat_channels_in > 0:
            # TODO should I do this separately for every feature?
            self.embed = nn.Conv2d(cat_channels_in, cat_channels_out, 1)
        self.conv1 = nn.Conv2d(cat_channels_out + num_channels, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 3)

    def forward(self, x):
        cat_x, num_x = x
        cat_x = Variable(torch.from_numpy(cat_x.transpose(0,3,1,2)).type(FloatTensor))
        # num_x = torch.from_numpy(num_x)

        if self.embed:
            cat_x = self.embed(cat_x)
            # x = torch.cat(cat_x, num_x, 3)
            x = cat_x
        else:
            x = num_x

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError
