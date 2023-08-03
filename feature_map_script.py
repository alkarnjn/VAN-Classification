import torch

dic = torch.load('features.pt')


for key, value in dic.items():
    print(key, len(value))
