import torch
import torch.nn as nn


class AverageFusion3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.fused_corr = None
        self.fused_mask = None

    def reset(self):
        self.fused_corr = None
        self.fused_mask = None

    def forward(self, corrs, masks):
        self.reset()

        if len(corrs) > 1:
            # Compute mask as the average of all input masks
            self.fused_mask = torch.stack(masks, dim=0).mean(0)

            # Compute corr as the average of all input corrs
            corrs = torch.stack(corrs, dim=0)
            self.fused_corr = corrs.mean(0) * self.fused_mask

        else:
            self.fused_corr = corrs[0]
            self.fused_mask = masks[0]

        return self.fused_corr, self.fused_mask
