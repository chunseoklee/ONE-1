import torch
import torch.nn as nn


# model
class net_ConstantPad2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.ConstantPad2d(0, 0.0)

    def forward(self, input):
        return self.op(input)


_model_ = net_ConstantPad2d()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
