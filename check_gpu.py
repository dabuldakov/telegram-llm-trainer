import torch
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступно: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")  # Должно показать 'NVIDIA A100'
print(f"Архитектура: {torch.cuda.get_device_capability(0)}")  # Должно быть (8, 0) для Ampere