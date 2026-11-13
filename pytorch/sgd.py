import torch as t
from random import randint
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import SGD, Adam


def model(X, w):
    return X @ w


N = 2
w = t.FloatTensor(N).uniform_(-1e-5, 1e-5)
w.requires_grad_(True)
x = t.arange(0, 3, 0.1)

y_train = 0.5 * x + 0.2 * t.sin(2 * x) - 3.0
x_train = t.tensor([[_x ** _n for _n in range(N)] for _x in x])

total = len(x)
lr = t.tensor([0.1, 0.01])

# loss_fn = nn.MSELoss()
loss_fn = nn.L1Loss()
# optimizer = SGD(params=[w], lr=0.01, momentum=0.8, nesterov=True)
optimizer = Adam(params=[w], lr=0.01)

for _ in range(1000):
    k = randint(0, total - 1)
    y = model(x_train[k], w)

    loss = loss_fn(y, y_train[k])

    # loss = ((y - y_train) ** 2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # w.data = w.data - lr * w.grad
    # w.grad.zero_()

print(w)
predict = model(x_train, w)

plt.plot(x, y_train.numpy())
plt.plot(x, predict.data.numpy())
plt.grid()
plt.show()
