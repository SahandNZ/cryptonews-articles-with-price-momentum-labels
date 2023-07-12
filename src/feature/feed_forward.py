from typing import Callable, List

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, layers: List[int], dropout: float, activation_fn: nn.Module, prediction_fn: Callable = None,
                 show_tqdm: bool = True):
        super().__init__(activation_fn=activation_fn, prediction_fn=prediction_fn, show_tqdm=show_tqdm)
        self.layers: List[int] = layers

        modules = [nn.Flatten()]
        for index in range(len(self.layers) - 2):
            modules.append(nn.Linear(layers[index], layers[index + 1]))
            modules.append(nn.BatchNorm1d(layers[index + 1]))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(layers[-2], layers[-1]))
        modules.append(nn.BatchNorm1d(layers[-1]))
        if activation_fn is not None:
            modules.append(activation_fn)

        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.tensor):
        return self.net(x)
