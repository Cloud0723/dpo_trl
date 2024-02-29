import torch

print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))
print(torch.tensor([1, 2, 3, 4]).to(device="cuda:0"))