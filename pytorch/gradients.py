import torch as t
from random import randint


def act(z):
    return t.tanh(z)


def df(z):
    s = act(z)
    return 1 - s * s


def go_forward(x_inp, w1, w2):
    z1 = t.mv(w1[:, :3], x_inp) + w1[:, 3]
    s1 = act(z1)

    z2 = t.dot(w2[:2], s1) + w2[2]
    y = act(z2)

    return y, z1, s1, z2


t.manual_seed(0)

W1 = t.rand(8).view(2, 4) - 0.5
W2 = t.rand(3) - 0.5

x_train = t.FloatTensor([
    (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)])
y_train = t.FloatTensor([-1, 1, -1, 1, -1, 1, -1, -1])

lmd = 0.05
N = 1000
total = len(y_train)

for _ in range(N):
    k = randint(0, total - 1)
    x = x_train[k]
    y, z1, s1, out = go_forward(x, W1, W2)

    e = y - y_train[k]

    delta = e * df(out)
    delta2 = W2[:2] * delta * df(z1)

    W2[:2] = W2[:2] - lmd * delta * s1
    W2[2] = W2[2] - lmd * delta

    W1[0, :3] = W1[0, :3] - lmd * delta2[0] * x
    W1[1, :3] = W1[1, :3] - lmd * delta2[1] * x

    W1[0, 3] = W1[0, 3] - lmd * delta2[0]
    W1[1, 3] = W1[1, 3] - lmd * delta2[1]

for x, d in zip(x_train, y_train):
    y, z1, s1, out = go_forward(x, W1, W2)
    print(f"Output value: {y.item():.3f} => {d.item()}")

print(W1)
print(W2)