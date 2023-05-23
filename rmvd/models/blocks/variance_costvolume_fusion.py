import torch
import torch.nn as nn


class VarianceCostvolumeFusion(nn.Module):

    def forward(self, feat_key, corrs, masks):

        steps = corrs[0].shape[1]
        feat_key = feat_key.unsqueeze(1).repeat(1, steps, 1, 1, 1)  # NSCHW
        key_mask = torch.ones_like(masks[0])

        corrs = [feat_key] + corrs
        masks = [key_mask] + masks

        corr_sum = torch.stack(corrs, dim=0).sum(0)
        corr_sq_sum = (torch.stack(corrs, dim=0)**2).sum(0)

        mask_sum = torch.stack(masks, dim=0).sum(0)  # >=1 because key_mask is all ones
        fused_mask = (mask_sum > 1).float()  # NSHW

        mask_sum = mask_sum.unsqueeze(2)
        fused_corr = (corr_sq_sum / mask_sum) - ((corr_sum / mask_sum)**2)  # 0 when all source views are masked ; NSCHW
        # TODO: try setting 0 values to -1 or so

        return fused_corr, fused_mask
