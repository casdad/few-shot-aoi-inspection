import torch

print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA     :", torch.version.cuda)
print("GPU            :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Device count   :", torch.cuda.device_count())
