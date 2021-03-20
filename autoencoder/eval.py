from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms

from model import AutoEncoder
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


import random
from typing import Sequence

class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)

class CustomTensorDataset(Dataset):
    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            tuple(self.transform(tensor[index]) for tensor in self.tensors)
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


model = AutoEncoder().cuda()
checkpoint = torch.load("autoencoder.pth")
criterion = nn.L1Loss()
model.load_state_dict(checkpoint)
model.eval()

with open("agent_code/rule_based_agent_auto_encode/states.npy", "rb") as f:
    data = np.load(f, allow_pickle=False)
data_t = torch.Tensor(data).unsqueeze(1)
data_t = data_t.cuda()

#transformations = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), MyRotateTransform((90,180,270))])


#dataset = CustomTensorDataset(data_t, transform=transformations)
dataset = TensorDataset(data_t)
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

eval_loss = []
for data in tqdm(dataloader):
    img = data[0]
    img = img.view(img.shape[0], -1)
    output = model(img)
    img_nn = (img * 12) - 1
    output_nn = ((output * 12) - 1).round()
    loss = criterion(output_nn, img_nn)
    #torch.sum(torch.all(img_nn == output_nn, 0))
    eval_loss.append(loss.item())
print(np.mean(eval_loss))

#torch.save(model.state_dict(), './autoencoder.pth')
