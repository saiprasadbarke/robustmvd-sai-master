import torch.nn as nn

from .utils import conv


class DispnetEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        C_curr = 64
        self.conv1 = conv(3, C_curr, kernel_size=7, stride=2)

        C_last = C_curr  # 64
        C_curr *= 2  # 128
        self.conv2 = conv(C_last, C_curr, kernel_size=5, stride=2)

        C_last = C_curr  # 128
        C_curr *= 2  # 256
        self.conv3 = conv(
            C_last, C_curr, kernel_size=3, stride=2
        )  # TODO: rename to conv3a

    def forward(self, image):
        conv1 = self.conv1(image)  # images_key: torch.Size([1, 3, 128, 160])
        conv2 = self.conv2(conv1)
        conv3a = self.conv3(conv2)
        return {"conv1": conv1, "conv2": conv2, "conv3a": conv3a}, conv3a


# {
# 'all_enc_key[conv1]': torch.Size([1, 64, 64, 80]), * 327680
# 'all_enc_key[conv2]': torch.Size([1, 128, 32, 40]), ** 163840
# 'all_enc_key[conv3a]': torch.Size([1, 256, 16, 20])81920
# }
