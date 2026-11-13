from random import randint

import torch as t
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

layer1 = nn.Linear(3, 2)
layer2 = nn.Linear(2, 1)


class NetGirl(nn.Module):
    def __init__(self, inp_dim, num_hidden, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(inp_dim, num_hidden)
        self.layer2 = nn.Linear(num_hidden, out_dim)

    def forward(self, x):
        x = self.layer1.forward(x)
        x = F.tanh(x)
        x = self.layer2.forward(x)
        x = F.tanh(x)

        return x


model = NetGirl(3, 2, 1)

x_train = t.FloatTensor([
    (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)])
y_train = t.FloatTensor([-1, 1, -1, 1, -1, 1, -1, -1])
total = len(y_train)

optimizer = optim.RMSprop(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

model.train()

for epoch in range(200):
    k = randint(0, total - 1)
    y = model(x_train[k])
    y = y.squeeze()
    loss = loss_fn(y, y_train[k])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with t.no_grad():
            pred = (model(x_train).squeeze(-1) > 0).float() * 2 - 1  # в {-1, +1}
            acc = (pred == y_train).float().mean() * 100
        print(f"Epoch {epoch}: acc: {acc.item():.2f}%")
        model.train()

model.eval()

# Глобально выключить
# model.requires_grad_(False)
# model.requires_grad_(True)

for x, d in zip(x_train, y_train):
    with t.no_grad():
        y = model(x)
        print(f"Output value: {y.data} => {d}")
