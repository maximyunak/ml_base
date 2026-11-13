import torch as t
import csv

data = []
target = []

with open("wine.data") as file:
    for row in csv.reader(file):
        target.append(int(row[0]) - 1)
        data.append(row[1:])

data = [[float(value) for value in row] for row in data]

x = t.tensor(data, dtype=t.float32)
y = t.tensor(target, dtype=t.long)

t.save({
    "data": x,
    "target": y
}, "wine.pt")