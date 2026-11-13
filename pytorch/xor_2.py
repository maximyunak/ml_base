import torch as t
import matplotlib.pyplot as plt

t.manual_seed(0)


def act(x):
    return 0 if x <= 0 else 1


N = 5
b1 = -0.5
b2 = 1.5

# x
x1 = t.randn(N)
# x2 =

# w
w_hidden = t.FloatTensor([[1, 1, -1.5], [1, 1, -0.5]])
w_out = t.FloatTensor([-1, 1, -0.5])

data_x = [1, 1]
x = t.FloatTensor(data_x + [1])

z_hidden = t.matmul(w_hidden, x)
print(f"z_hidden: {z_hidden}")

u_hidden = t.FloatTensor([act(x) for x in z_hidden] + [1])
print(f"u_hidden: {u_hidden}")

z_out = t.dot(w_out, u_hidden)

y = act(z_out)

print(f"y: {y}")
