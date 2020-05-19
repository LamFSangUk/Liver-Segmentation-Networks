import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_channels, growth_rate, block, drop_rate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_channels, growth_rate, nb_layers, drop_rate)

    def _make_layer(self, block, in_channels, growth_rate, nb_layers, drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_channels+i*growth_rate, growth_rate, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
