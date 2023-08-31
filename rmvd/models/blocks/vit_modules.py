import torch
import torch.nn as nn
import torch.nn.functional as F


# copy-pasta with small modifications from MVSFormer repo
class VITDecoderStage4Single(nn.Module):
    def __init__(self, mode="add"):
        super(VITDecoderStage4Single, self).__init__()
        ch, vit_ch = 64, 384
        self.attn = AttentionFusionSimple(vit_ch, ch * 4, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch * 2),
            nn.GELU(),
            nn.ConvTranspose2d(
                ch * 2,
                int(ch) if mode == "concat" else int(ch / 2),
                4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(int(ch) if mode == "concat" else int(ch / 2)),
            nn.GELU(),
        )

    def forward(self, x, att):
        x = self.attn(x, att)
        x = self.decoder(x)

        return x


class AttentionFusionSimple(nn.Module):
    def __init__(self, vit_ch, out_ch, nhead):
        super(AttentionFusionSimple, self).__init__()
        self.conv_l = nn.Sequential(
            nn.Conv2d(vit_ch + nhead, vit_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(vit_ch),
        )
        self.conv_r = nn.Sequential(
            nn.Conv2d(vit_ch, vit_ch, kernel_size=3, padding=1), nn.BatchNorm2d(vit_ch)
        )
        self.act = Swish()
        self.proj = nn.Conv2d(vit_ch, out_ch, kernel_size=1)

    def forward(self, x, att):
        # x:[B,C,H,W]; att:[B,nh,H,W]
        x1 = self.act(self.conv_l(torch.cat([x, att], dim=1)))
        x2 = self.act(self.conv_r(x * att))
        x = self.proj(x1 * x2)
        return x


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
