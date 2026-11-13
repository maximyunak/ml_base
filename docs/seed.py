import torch as t
from torch.nn import Linear

# автоматом ставит везде
t.manual_seed(0)
# только для GPU
t.cuda.manual_seed(0)

# Для некотоырх библиотек nvidia
# Только для отладки, для обучения не надо, будет медленнее
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False

x = Linear(2,3, device='cuda')

print(x.weight)