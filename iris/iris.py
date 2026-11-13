import torch as t
from torch.nn import MSELoss, BCELoss, CrossEntropyLoss, Sequential, Linear, ReLU
from torch.optim import SGD

X = t.load("data.pt")
y = t.load("target.pt")

X_train, X_val = X[:100], X[100:]
y_train, y_val = y[:100], y[100:]

model = Sequential(
    Linear(4,50),
    ReLU(),
    Linear(50,3)
)
loss_fn = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)


for epoch in range(200):
    y_hat = model(X_train)
    loss = loss_fn(y_hat, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        model.eval()
        with t.no_grad():
            acc = (model(X_val).argmax(1) == y_val).sum() / len(y_val) * 100
            # acc = (model(X_val).argmax(1) == y_val).float().mean()
        print(f"Epoch {epoch}: acc: {acc}, ")
        model.train()

score = (model(X_val).argmax(1) == y_val).sum() / len(y_val) * 100
print(f"Accuracy: {score:.2f}%")

# model.eval()
# with t.no_grad():
#     probs = model(X_val).softmax(dim=1)   # вероятности по 3 классам
#     pred = probs.argmax(dim=1)            # номер предсказанного класса
#
# names = ["setosa", "versicolor", "virginica"]
# for i in range(len(X_val)):
#     ok = "✓" if pred[i] == y_val[i] else "✗"
#     print(f"{ok} предсказано: {names[pred[i]]:<11} "
#           f"(уверенность {probs[i].max():.0%}), "
#           f"на самом деле: {names[y_val[i]]}")