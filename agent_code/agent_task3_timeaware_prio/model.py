import torch.nn as nn
import torch

class BombNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        _, c, h, w = input_dim
        if h != w and h != 15:
            print("Input must be 15x15")
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten())
        self.fc = nn.Sequential(
            nn.Linear(2593, 1024), # + 1 for current step
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim))

    def forward(self, inp):
        state, step = inp
        x = self.convs(state)
        x = torch.cat((x, step), 1)
        x = self.fc(x)
        return x
