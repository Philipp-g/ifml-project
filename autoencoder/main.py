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

with open("agent_code/rule_based_agent_auto_encode/states_no_dup.npy", "rb") as f:
    data = np.load(f, allow_pickle=False)
data = data[:1_000_000]
data_t = torch.Tensor(data)#.unsqueeze(1)
data_t = data_t.cuda()

dataset = TensorDataset(data_t)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(100):
    epoch_loss = []
    for data in tqdm(dataloader):
        img = data[0]
        img = img.view(img.shape[0], -1)
        output = model(img)
        loss = criterion(output, img)
        epoch_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch: {epoch}, Loss: {np.mean(epoch_loss)}")
    writer.add_scalar("Loss", np.mean(epoch_loss), epoch)

torch.save(model.state_dict(), './autoencoder.pth')
