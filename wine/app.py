import torch as t
from torch.nn import CrossEntropyLoss, Sequential, Linear, ReLU
from sklearn.model_selection import train_test_split
from torch.optim import SGD

dataset = t.load("wine.pt")

data = dataset['data']
target = dataset['target']

X_train, X_val, y_train, y_val = train_test_split(
    data,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

mean = X_train.mean(dim=0)
std = X_train.std(dim=0)

print(X_train)
print(X_val)
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

model = Sequential(
    Linear(13, 40),
    ReLU(),
    Linear(40, 3),
)

lost_fn = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)

for epoch in range(150):
    y_hat = model(X_train)
    loss = lost_fn(y_hat, y_train)
    #
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        model.eval()
        #
        with t.no_grad():
            acc = (model(X_val).argmax(1) == y_val).sum() / len(y_val) * 100
        print(f"epoch: {epoch}, loss: {loss.item()}, acc: {acc}%")
        model.train()

score = (model(X_val).argmax(1) == y_val).sum() / len(y_val) * 100
print(f"Accuracy: {score}%")