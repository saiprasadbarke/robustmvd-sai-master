import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .blocks.gwcnet_hourglass_costvolumeregularization import hourglass, convbn_3d
from .blocks.gwcnet_pyramid_feature_extractor import feature_extraction, convbn
from .blocks.planesweep_corr import PlanesweepCorrelation
from .blocks.variance_costvolume_fusion import VarianceCostvolumeFusion


class GwcNet(nn.Module):
    def __init__(self, num_sampling_points, use_concat_volume=False):
        super(GwcNet, self).__init__()
        self.num_sampling_points = num_sampling_points
        self.use_concat_volume = use_concat_volume

        self.num_groups = 40

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(
                concat_feature=True, concat_feature_channel=self.concat_channels
            )
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(
            convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.dres1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
        )

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif0 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )

        self.classif1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )

        self.classif2 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )

        self.classif3 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.out_channels
                )
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        gwc_volume = build_gwc_volume(
            features_left["gwc_feature"],
            features_right["gwc_feature"],
            self.num_sampling_points // 4,
            self.num_groups,
        )
        if self.use_concat_volume:
            concat_volume = build_concat_volume(
                features_left["concat_feature"],
                features_right["concat_feature"],
                self.num_sampling_points // 4,
            )
            volume = torch.cat((gwc_volume, concat_volume), 1)
        else:
            volume = gwc_volume

        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)

        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)

            cost0 = F.upsample(
                cost0,
                [self.num_sampling_points, left.size()[2], left.size()[3]],
                mode="trilinear",
            )
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.num_sampling_points)

            cost1 = F.upsample(
                cost1,
                [self.num_sampling_points, left.size()[2], left.size()[3]],
                mode="trilinear",
            )
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.num_sampling_points)

            cost2 = F.upsample(
                cost2,
                [self.num_sampling_points, left.size()[2], left.size()[3]],
                mode="trilinear",
            )
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.num_sampling_points)

            cost3 = F.upsample(
                cost3,
                [self.num_sampling_points, left.size()[2], left.size()[3]],
                mode="trilinear",
            )
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.num_sampling_points)
            return [pred0, pred1, pred2, pred3]

        else:
            cost3 = self.classif3(out3)
            cost3 = F.upsample(
                cost3,
                [self.num_sampling_points, left.size()[2], left.size()[3]],
                mode="trilinear",
            )
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.num_sampling_points)
            return [pred3]
