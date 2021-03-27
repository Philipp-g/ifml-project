from torch.utils.data import TensorDataset, DataLoader

from model import AutoEncoder
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./runs/")

model = AutoEncoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dataloader_valid = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(50):
    train_loss = []
    valid_loss = []
    for data in tqdm(dataloader_train):
        img = data[0]
        img = img.view(img.shape[0], -1)
        output = model(img)
        loss = criterion(output, img)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for data in tqdm(dataloader_valid):
        img = data[0]
        img = img.view(img.shape[0], -1)
        output = model(img)
        loss = criterion(output, img)
        valid_loss.append(loss.item())
    print(f"epoch: {epoch}, Train Loss: {np.mean(train_loss)}")
    print(f"epoch: {epoch}, Valid Loss: {np.mean(valid_loss)}")
    writer.add_scalar("Train loss", np.mean(train_loss), epoch)
    writer.add_scalar("Valid loss", np.mean(valid_loss), epoch)

torch.save(model.state_dict(), './autoencoder.pth')
