import torch as t

x = t.tensor([2.0], requires_grad=True)
y = t.tensor([-4.0], requires_grad=True)

f = (x + y) ** 2 + 2 * x * y
f.backward()



print(f)
print(x.data, x.grad)
print(y.data, y.grad)