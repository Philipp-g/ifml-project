import copy

import torch.nn as nn
import torch

class BombNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 9), # + 1 for current step
            nn.ReLU(),
            nn.Linear(9, output_dim))

    def forward(self, inp):
        state, step = inp
        state = state.view(state.shape[0], -1)
        x = torch.cat((state, step), 1)
        x = self.fc(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(15*15, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
