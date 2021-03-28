import torch.nn as nn
import torch

# Grad-CAM code inspired by https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
class BombNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(2593, 1024), # + 1 for current step
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 6))
        self.gradients = None

    def activation_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, inp):
        state, step = inp
        return self.convs(state)

    def forward(self, inp):
        state, step = inp
        x = self.convs(state)
        h = x.register_hook(self.activation_hook)
        x = x.view(state.shape[0], -1)
        x = torch.cat((x, step), 1)
        x = self.fc(x)
        return x
