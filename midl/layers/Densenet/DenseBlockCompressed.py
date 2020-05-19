import torch.nn as nn
import torch.nn.functional as F

from midl.layers.ActFunc import ActFunc


class DenseBlockCompressed(nn.Module):
    def __init__(self, nb_layers, in_channels, growth_rate, block, drop_rate=0.0):
        super(DenseBlockCompressed, self).__init__()
        self.layer = self._make_layer(block, in_channels, growth_rate, nb_layers, drop_rate)

    def _make_layer(self, block, in_channels, growth_rate, nb_layers, drop_rate):
        layers = []

        layers.append(nn.Conv3d(in_channels=in_channels,
                              out_channels=growth_rate,
                              kernel_size=3,
                              stride=1,
                              padding=1))
        layers.append(nn.BatchNorm3d(growth_rate))
        layers.append(ActFunc('ReLU'))

        for i in range(nb_layers-1):
            layers.append(block((i+1)*growth_rate, growth_rate, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
