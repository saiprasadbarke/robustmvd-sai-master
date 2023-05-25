import torch
import torch.nn as nn


class VarianceCostvolumeFusion(nn.Module):

    def forward(self, feat_key, corrs, masks):

        steps = corrs[0].shape[1]
        feat_key = feat_key.unsqueeze(1).repeat(1, steps, 1, 1, 1)  # NSCHW
        key_mask = torch.ones_like(masks[0])
        
        corr_sum = feat_key
        corr_sq_sum = feat_key**2
        mask_sum = key_mask
        
        for corr, mask in zip(corrs, masks):
            corr_sum = corr_sum + corr
            corr_sq_sum = corr_sq_sum + corr**2
            mask_sum = mask_sum + mask
            
        fused_mask = (mask_sum > 1).float()  # NSHW
        mask_sum = mask_sum.unsqueeze(2)
        fused_corr = (corr_sq_sum.div_(mask_sum)).sub_((corr_sum.div_(mask_sum).pow_(2)))  # 0 when all source views are masked ; NSCHW

        return fused_corr, fused_mask
