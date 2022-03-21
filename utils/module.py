import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.utils import init_weights


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_scale_factor, projection_size, init_method='He', activation='relu'):
        super().__init__()

        mlp_hidden_size = round(mlp_scale_factor * in_channels)
        if activation == 'relu':
            non_linear_layer = nn.ReLU(inplace=True)
        elif activation == 'leaky relu':
            non_linear_layer = nn.LeakyReLU(inplace=True)
        elif activation == 'tanh':
            non_linear_layer = nn.Tanh()
        else:
            raise AssertionError(f'{activation} is not supported yet.')

        self.mlp_head = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            non_linear_layer,
            nn.Linear(mlp_hidden_size, projection_size)
        )
        init_weights(self.mlp_head, init_method)

    def forward(self, x):
        return self.mlp_head(x)


