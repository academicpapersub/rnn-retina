import torch
import torch.nn as nn
import numpy as np

class Conv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=0, dilation=1, 
            bias=False, activation=True, padding_mode='zeros'):
        super().__init__()
        self.activation = activation
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, 
                padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x
        out = self.relu(self.bn(self.conv(out))) if self.activation else self.conv(out)
        return out
