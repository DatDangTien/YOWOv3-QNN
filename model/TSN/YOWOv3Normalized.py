import torch
import torch.nn as nn

class YOWOv3Normalized(nn.Module):
    """
    A wrapper around the YOWOv3 model that moves host-side preprocessing into the model graph.
    It expects a float32 input of shape [B, C, T, H, W] in the range [0.0, 1.0],
    and applies mean/std normalization.
    """
    def __init__(self, base_model, mean, std):
        super().__init__()
        self.base_model = base_model
        # Register mean and std as buffers to package them into the exported model
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1, 1))

    def forward(self, x):
        # Apply standard normalization
        x = (x - self.mean) / self.std
        # Forward through the main detector model
        return self.base_model(x)
