from random import randint

import torch as t
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, random_split, DataLoader

from embedding.dataset import word_to_idx, sentences, labels

# embedding = nn.Embedding(100, 3)

d_size = len(word_to_idx)
w_size = len(sentences[0].split())
epochs = 500
# print(linear(result))

def sentence_to_vec(sentence):
    return t.tensor([[word_to_idx[k] for k in sentence.split()]])

# learn

class ClassicModel(nn.Module):
    def __init__(self, dict_size, word_dim):
        super().__init__()

        self.embedding = nn.Embedding(dict_size, word_dim)
        self.linear = nn.Linear(word_dim, 2)

    def forward(self, x):
        result = self.embedding(x).mean(dim=1)

        return self.linear(result)

embedding_dim = 8
model = ClassicModel(d_size, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()


v = t.cat(
    [sentence_to_vec(sentence) for sentence in sentences],
    dim=0
)

y = t.tensor(labels, dtype=t.long)

dataset = TensorDataset(v, y)

train_data, test_data = random_split(dataset, [150,50], generator=t.Generator().manual_seed(0))
train_loader = DataLoader(dataset=train_data, batch_size=100, shuffle=True)

model.train()
for epoch in range(epochs):
     for x_train, y_train in train_loader:
        predict = model(x_train)

        loss = loss_fn(predict, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            # print(f"predict: {predict}, fact: {y_train}")
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")


model.eval()

Q = 0
for x,y in test_data:
    with t.no_grad():
        x = x.unsqueeze(0)
        p = model(x)
        p = t.argmax(p, dim=1)
        Q += t.sum(p==y.item()).item()

Q /= len(test_data)

print(f"Accuracy: {Q * 100:.2f}%")


def predict_text(text):
    model.eval()

    idxs = sentence_to_vec(text)

    with t.no_grad():
        prediction = model(idxs)
        print(prediction)
        prediction = t.argmax(prediction, dim=1)

    return "Положительное" if prediction.item() == 1 else "Отрицательное"


while True:
    text = input("Enter text: ")
    if text.lower() == "exit":
        break
    print(predict_text(text))
