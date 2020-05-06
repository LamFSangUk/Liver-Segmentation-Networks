import torch
import torch.nn as nn

class ActFunc(nn.Module):
    def __init__(self, activation, **kwargs):
        super(ActFunc, self).__init__()

        self.activation_function = nn.ModuleDict([
            ['ReLU', nn.ReLU()],
            ['PReLU', nn.PReLU(kwargs['num_parameters'])],
            ['None', nn.Identity()]
        ])[activation]

    def forward(self, x):
        x = self.activation_function(x)
        return x
