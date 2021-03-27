from torch.utils.data import TensorDataset, DataLoader

from model import AutoEncoder
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm


model = AutoEncoder().cuda()
checkpoint = torch.load("autoencoder.pth")
criterion = nn.L1Loss()
model.load_state_dict(checkpoint)
model.eval()

with open("../agent_code/rule_based_agent_auto_encode/states.npy", "rb") as f:
    data = np.load(f, allow_pickle=False)

data = data[:2_500_000]
data_t = torch.Tensor(data)
data_t = data_t.cuda()
dataset = TensorDataset(data_t)

train_len = int(0.7*len(dataset))
valid_len = (len(dataset)-train_len)//2
test_len = len(dataset) - train_len - valid_len

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, valid_len, test_len], generator=torch.Generator().manual_seed(42))

batch_size = 32
dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

test_loss = []
for data in tqdm(dataloader_test):
    img = data[0]
    img = img.view(img.shape[0], -1)
    output = model(img)
    loss = criterion(output, img)
    test_loss.append(loss.item())
print(np.mean(test_loss))
