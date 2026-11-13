import torch as t
import matplotlib.pyplot as plt

t.manual_seed(0)

N = 5
b = 3

x1 = t.rand(N)
tr = t.randint(1, 10, [N]) / 10
x2 = x1 + tr + b
c1 = t.vstack([x1, x2, t.ones(N)]).mT
print(f"x1 = {x1}, \nx2 = {x2} \nc1 = {c1}")
print(f"tr1 {tr}")

x1 = t.rand(N)
tr = t.randint(1, 10, [N]) / 10
x2 = x1 - tr + b
c2 = t.vstack([x1, x2, t.ones(N)]).mT
print(f"x1 = {x1}, \nx2 = {x2} \nc1 = {c2}")
print(f"tr2 {tr}")

f = [0 + b, 1 + b]
w1 = -0.5
w2 = -w1
w3 = -b * w2
w = t.FloatTensor([w1, w2, w3])

# w = t.FloatTensor([-0.3, 0.3])

for i in range(N):
    x = c2[:][i]
    y = t.dot(w, x)

    if y >= 0:
        print("c1 class")
    else:
        print("c2 class")

plt.scatter(c1[:, 0], c1[:, 1], s=10, color="red")
plt.scatter(c2[:, 0], c2[:, 1], s=10, color="blue")
plt.plot(f)
plt.grid()
plt.show()
