import torch.nn as nn

from .utils import conv


class DispnetContextEncoder(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.conv_redir = conv(in_channels, 32, kernel_size=1, stride=1)

    def forward(self, conv3):
        conv_redir = self.conv_redir(conv3)
        return conv_redir
