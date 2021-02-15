import copy

import torch.nn as nn

class BombNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        _, c, h, w = input_dim
        if h != w and h != 17:
            print("Input must be 17x17")
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1600, 1600),
            nn.ReLU(),
            nn.Linear(1600, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
