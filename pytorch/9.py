import torch as t

device = t.device("cuda" if t.cuda.is_available() else "cpu")


def act(x):
    return 0 if x < 0.5 else 1


def go(house, rock, attr):
    X = t.tensor([house, rock, attr], dtype=t.float32, device=device)
    Wh = t.tensor([[0.3, 0.3, 0], [0.4, -0.5, 1]], device=device)
    Wout = t.tensor([-1.0, 1.0], device=device)

    Zh = t.mv(Wh, X)
    print(f"Знач сумм нейронов скрытого слоя: {Zh}")

    Uh = t.tensor([act(x) for x in Zh], dtype=t.float32, device=device)
    print(f"Знач на выходах нейронов скрытого слоя: {Uh}")

    Zout = t.dot(Wout, Uh)
    Y = act(Zout)
    print(f"Out value: {Y}")

    return Y


house = 1
rock = 1
attr = 1

res = go(house, rock, attr)
if res == 1:
    print("I like u")
else:
    print("I do not like u")
