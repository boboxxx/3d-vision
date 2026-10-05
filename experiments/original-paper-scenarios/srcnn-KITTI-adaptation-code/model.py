"""Batched float32 trainable translation, initialized from six author arrays."""
import torch
from torch import nn
from torch.nn import functional as F


class Srcnn(nn.Module):
    def __init__(self, author):
        super().__init__()
        assert author.kernel_sizes == (9, 5, 5)
        self.kernel_sizes = author.kernel_sizes
        for key, value in author.state_dict().items():
            self.register_parameter(key, nn.Parameter(value.detach().float().clone()))
        assert len(self.state_dict()) == len(list(self.parameters())) == 6

    def forward(self, value):
        assert value.dtype == torch.float32 and value.ndim == 4 and value.shape[1] == 1
        for index, kernel in enumerate(self.kernel_sizes, 1):
            value = F.conv2d(F.pad(value, (kernel // 2,) * 4, mode='replicate'),
                            getattr(self, 'weight' + str(index)), getattr(self, 'bias' + str(index)))
            if index < 3:
                value = F.relu(value)
        return value


def central_loss(predicted, target):
    assert predicted.shape == target.shape and tuple(predicted.shape[1:]) == (1, 49, 49)
    return (predicted[:, :, 8:-8, 8:-8] - target[:, :, 8:-8, 8:-8]).square().mean()
