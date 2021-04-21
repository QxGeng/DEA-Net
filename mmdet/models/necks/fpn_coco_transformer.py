import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, normal_init

from ..registry import NECKS
from ..utils import ConvModule
from mmdet.core import multi_apply

import torch



@NECKS.register_module
class FPNTransformer_COCO(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(FPNTransformer_COCO, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        channels_ = 256
        self.r_conv = nn.Conv2d(channels_ * 4, channels_, 3, padding=1)

        self.batch_ = nn.BatchNorm2d(5)
        self.batch_11 = nn.BatchNorm2d(channels_ * 2)
        self.batch_0 = nn.BatchNorm2d(channels_)
        self.batch_1 = nn.BatchNorm2d(channels_)
        self.batch_2 = nn.BatchNorm2d(channels_)
        self.batch_3 = nn.BatchNorm2d(channels_)
        self.batch_4 = nn.BatchNorm2d(channels_)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.conv_p2 = nn.Conv2d(256 * 2, 256, 3, stride=1, padding=1)
        self.conv_p3 = nn.Conv2d(256 * 2, 256, 3, stride=1, padding=1)
        self.conv_p4 = nn.Conv2d(256 * 2, 256, 3, stride=1, padding=1)
        self.conv_p5 = nn.Conv2d(256 * 2, 256, 3, stride=1, padding=1)
        self.conv_p6 = nn.Conv2d(256 * 2, 256, 3, stride=1, padding=1)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

        normal_init(self.batch_, std=0.01)
        normal_init(self.batch_11, std=0.01)
        normal_init(self.batch_0, std=0.01)
        normal_init(self.batch_1, std=0.01)
        normal_init(self.batch_2, std=0.01)
        normal_init(self.batch_3, std=0.01)
        normal_init(self.batch_4, std=0.01)

        normal_init(self.conv_p2, std=0.01)
        normal_init(self.conv_p3, std=0.01)
        normal_init(self.conv_p4, std=0.01)
        normal_init(self.conv_p5, std=0.01)
        normal_init(self.conv_p6, std=0.01)

    def forward_single1(self, x):

        torch.cuda.empty_cache()

        batch, C, height, width = x.size()
        query = x.view(batch, C, -1)
        key = x.view(batch, C, -1).permute(0, 2, 1)
        energy = torch.bmm(query, key)
        value = x.view(batch, C, -1)
        atten = F.softmax(energy)

        out = torch.bmm(atten, value)
        out = out.view(batch, C, height, width)
        out = self.batch_11(out)
        x = self.r_conv(torch.cat([x,out],dim=1))

        x = F.relu(x, inplace=True)
        return x

    def forward(self, inputs):

        torch.cuda.empty_cache()

        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        #--------------------------------------------------
        feats = tuple(outs)

        p2_out = feats[0]

        p3_out = feats[1]
        p3_out_interpolate = F.interpolate(p3_out, scale_factor=2, mode='nearest')

        p4_out = feats[2]
        p4_out_interpolate = F.interpolate(p4_out, scale_factor=4, mode='nearest')

        p5_out = feats[3]
        p5_out_interpolate = F.interpolate(p5_out, scale_factor=8, mode='nearest')
        p5_out_interpolate = p5_out_interpolate[:,:,2:102,:]


        p6_out = feats[4]
        p6_out_interpolate = F.interpolate(p6_out, scale_factor=16, mode='nearest')
        p6_out_interpolate = p6_out_interpolate[:,:,12:196,4:172]

        p_merge = torch.cat([p2_out, p3_out_interpolate, p4_out_interpolate, p5_out_interpolate, p6_out_interpolate],
                            dim=1)

        m_batchsize, C, height, width = p_merge.size()
        proj_query = p_merge.view(m_batchsize, C, -1)
        proj_key = p_merge.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new)
        proj_value = p_merge.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        # out = self.batch_(out)
        out = out.view(m_batchsize, C, -1)
        out_mean = torch.mean(out, dim=2)

        atten_p2 = out_mean[:, 0].unsqueeze(1).expand(m_batchsize, 256) \
            .unsqueeze(2).expand(m_batchsize, 256, 100) \
            .unsqueeze(3).expand(m_batchsize, 256, 100, p2_out.shape[3])
        atten_p3 = out_mean[:, 1].unsqueeze(1).expand(m_batchsize, 256) \
            .unsqueeze(2).expand(m_batchsize, 256, 50) \
            .unsqueeze(3).expand(m_batchsize, 256, 50, p3_out.shape[3])
        atten_p4 = out_mean[:, 2].unsqueeze(1).expand(m_batchsize, 256) \
            .unsqueeze(2).expand(m_batchsize, 256, 25) \
            .unsqueeze(3).expand(m_batchsize, 256, 25, p4_out.shape[3])
        atten_p5 = out_mean[:, 3].unsqueeze(1).expand(m_batchsize, 256) \
            .unsqueeze(2).expand(m_batchsize, 256, 13) \
            .unsqueeze(3).expand(m_batchsize, 256, 13, p5_out.shape[3])
        atten_p6 = out_mean[:, 4].unsqueeze(1).expand(m_batchsize, 256) \
            .unsqueeze(2).expand(m_batchsize, 256, 7) \
            .unsqueeze(3).expand(m_batchsize, 256, 7, p6_out.shape[3])

        p2_new = torch.cat([atten_p2 * p2_out, p2_out], dim=1)
        p3_new = torch.cat([atten_p3 * p3_out, p3_out], dim=1)
        p4_new = torch.cat([atten_p4 * p4_out, p4_out], dim=1)
        p5_new = torch.cat([atten_p5 * p5_out, p5_out], dim=1)
        p6_new = torch.cat([atten_p6 * p6_out, p6_out], dim=1)

        feats_new = (p2_new, p3_new, p4_new, p5_new, p6_new)
        outs = multi_apply(self.forward_single1, feats_new)

        # print(len(outs))
        # print(len(outs[0]),len(outs[0][0]))
        # print(len(outs[1]),len(outs[1][0]))
        # print(outs[2].shape)

        # batch_size == 3

        if len(outs) == 3:
            k00 = torch.unsqueeze(outs[0][0],0)
            k10 = torch.unsqueeze(outs[1][0],0)
            k20 = torch.unsqueeze(outs[2][0],0)
            kk0 = self.batch_0(torch.cat([k00,k10,k20],0))

            k01 = torch.unsqueeze(outs[0][1], 0)
            k11 = torch.unsqueeze(outs[1][1], 0)
            k21 = torch.unsqueeze(outs[2][1], 0)
            kk1 = self.batch_1(torch.cat([k01, k11,k21], 0))

            k02 = torch.unsqueeze(outs[0][2], 0)
            k12 = torch.unsqueeze(outs[1][2], 0)
            k22 = torch.unsqueeze(outs[2][2], 0)
            kk2 = self.batch_2(torch.cat([k02, k12,k22], 0))

            k03 = torch.unsqueeze(outs[0][3], 0)
            k13 = torch.unsqueeze(outs[1][3], 0)
            k23 = torch.unsqueeze(outs[2][3], 0)
            kk3 = self.batch_3(torch.cat([k03, k13,k23], 0))

            k04 = torch.unsqueeze(outs[0][4], 0)
            k14 = torch.unsqueeze(outs[1][4], 0)
            k24 = torch.unsqueeze(outs[2][4], 0)
            kk4 = self.batch_4(torch.cat([k04, k14,k24], 0))

            # print(kk0.shape,feats[0].shape,'0000000000000000000000')

            kk0_ = self.conv_p2(torch.cat([kk0, feats[0]], dim=1))
            kk1_ = self.conv_p3(torch.cat([kk1, feats[1]], dim=1))
            kk2_ = self.conv_p4(torch.cat([kk2, feats[2]], dim=1))
            kk3_ = self.conv_p5(torch.cat([kk3, feats[3]], dim=1))
            kk4_ = self.conv_p6(torch.cat([kk4, feats[4]], dim=1))

            # print(kk0_.shape, '---')
            # print(kk1_.shape, '+++')
            # print(kk2_.shape, '+++')
            # print(kk3_.shape, '+++')
            # print(kk4_.shape, '+++')
            outs = [kk0_, kk1_, kk2_, kk3_, kk4_]

        if len(outs) == 2:
            k00 = torch.unsqueeze(outs[0][0],0)
            k10 = torch.unsqueeze(outs[1][0],0)
            kk0 = self.batch_0(torch.cat([k00,k10],0))

            k01 = torch.unsqueeze(outs[0][1], 0)
            k11 = torch.unsqueeze(outs[1][1], 0)
            kk1 = self.batch_1(torch.cat([k01, k11], 0))

            k02 = torch.unsqueeze(outs[0][2], 0)
            k12 = torch.unsqueeze(outs[1][2], 0)
            kk2 = self.batch_2(torch.cat([k02, k12], 0))

            k03 = torch.unsqueeze(outs[0][3], 0)
            k13 = torch.unsqueeze(outs[1][3], 0)
            kk3 = self.batch_3(torch.cat([k03, k13], 0))

            k04 = torch.unsqueeze(outs[0][4], 0)
            k14 = torch.unsqueeze(outs[1][4], 0)
            kk4 = self.batch_4(torch.cat([k04, k14], 0))

            # print(kk0.shape,feats[0].shape,'0000000000000000000000')

            kk0_ = self.conv_p2(torch.cat([kk0, feats[0]], dim=1))
            kk1_ = self.conv_p3(torch.cat([kk1, feats[1]], dim=1))
            kk2_ = self.conv_p4(torch.cat([kk2, feats[2]], dim=1))
            kk3_ = self.conv_p5(torch.cat([kk3, feats[3]], dim=1))
            kk4_ = self.conv_p6(torch.cat([kk4, feats[4]], dim=1))

            # print(kk0_.shape, '---')
            # print(kk1_.shape, '+++')
            # print(kk2_.shape, '+++')
            # print(kk3_.shape, '+++')
            # print(kk4_.shape, '+++')
            outs = [kk0_, kk1_, kk2_, kk3_, kk4_]

        elif len(outs) == 1:
            k00 = torch.unsqueeze(outs[0][0], 0)

            k01 = torch.unsqueeze(outs[0][1], 0)

            k02 = torch.unsqueeze(outs[0][2], 0)

            k03 = torch.unsqueeze(outs[0][3], 0)

            k04 = torch.unsqueeze(outs[0][4], 0)

            outs = [k00, k01, k02, k03, k04]


        # print('0000000000000000000003')

        return tuple(outs)
