import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import math
import copy
import numpy as np
import torch.nn.functional as F


class Plane_Feature_extraction(nn.Module):
    def __init__(self, args):
        super(Plane_Feature_extraction, self).__init__()

        self.args = args

        # nonlineraity
        self.relu = nn.ReLU()
        self.act_f = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        # encoding layers
        self.encoding = ResNet(self.args.inplanes, self.args.depth, self.args.d_f, self.args, bottleneck=False)
        self.encoding_sag = ResNet(self.args.inplanes, self.args.depth, self.args.d_f, self.args, bottleneck=False)
        self.encoding_cor = ResNet(self.args.inplanes, self.args.depth, self.args.d_f, self.args, bottleneck=False)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, plane):
        # encode
        # x [B, 193, 229, 193]
        # axial
        x = x.unsqueeze(1)
        # x_axial = x.clone()  # [B, 1, 193, 229, 193]

        if plane == 'axial':
            encoding = self.encoding(x)  # [1, 256, 193]
            # encoding, _, _, _ = self.encoding(x)  # [1, 256, 193]
        elif plane == 'sag':
            # x_sagittal = x.clone().permute(0, 1, 4, 3, 2)  # [B, 1, 193, 229, 193]
            encoding = self.encoding_sag(x)  # [1, 256, 193]
            # encoding, _, _, _ = self.encoding_sag(x)  # [1, 256, 193]
        elif plane == 'cor':
            # x_coronal = x.clone().permute(0, 1, 2, 4, 3)
            encoding = self.encoding_cor(x)
            # encoding, _, _, _ = self.encoding_cor(x)
        else:
            pass
        return encoding

# class Feature_extraction(nn.Module):
#     def __init__(self, args):
#         super(Feature_extraction, self).__init__()
#
#         self.args = args
#
#         # nonlineraity
#         self.relu = nn.ReLU()
#         self.act_f = nn.LeakyReLU()
#         self.tanh = nn.Tanh()
#
#         # avg pooling
#         # self.avgpool1d = nn.AvgPool1d(self.args.axial_slicelen * self.args.coronal_slicelen * self.args.axial_slicelen)
#
#         # encoding layers
#         self.encoding = ResNet(self.args.inplanes, self.args.depth, self.args.d_f, self.args, bottleneck=False)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         x = x.unsqueeze(1)  # [B, 1, 193, 229, 193]
#         encoding = self.encoding(x)  # [4, 16, 229]
#
#         avg_encoding = torch.mean(encoding, dim=-1)
#
#         return avg_encoding


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 1), stride=(stride, stride, 1),
                     padding=(1, 1, 0), bias=False)
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.InstanceNorm3d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1), bias=False)
        self.bn1 = nn.InstanceNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 1), stride=(stride, stride, 1), padding=(1, 1, 0), bias=False)
        self.bn2 = nn.InstanceNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * Bottleneck.expansion, kernel_size=(1, 1, 1), bias=False)
        self.bn3 = nn.InstanceNorm3d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, inplanes, depth, num_classes, args, bottleneck=False):
        super(ResNet, self).__init__()
        self.args = args

        blocks = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
        assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'
        self.inplanes = inplanes
        self.f_out = [inplanes, inplanes * 2, inplanes * 4, inplanes * 8]
        # self.inplanes = 64
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=(7, 7, 1), stride=(2, 2, 1), padding=(3, 3, 0), bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = nn.InstanceNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))
        self.layer1 = self._make_layer(blocks[depth], self.f_out[0], layers[depth][0])
        self.layer2 = self._make_layer(blocks[depth], self.f_out[1], layers[depth][1], stride=2)
        self.layer3 = self._make_layer(blocks[depth], self.f_out[2], layers[depth][2], stride=2)
        self.layer4 = self._make_layer(blocks[depth], self.f_out[3], layers[depth][3], stride=2)

        # self.deconv2 = UpsampleConvLayer(self.ch * 2, self.ch, kernel_size=3, stride=1, size=(193, 229))
        # self.upsample = nn.Upsample(size=(160, 192), mode='nearest')

        # self.interp1_fc = nn.Linear(inplanes, self.args.intp_ch)
        # self.interp2_fc = nn.Linear(inplanes * 2, self.args.intp_ch)
        # self.interp3_fc = nn.Linear(inplanes * 4, self.args.intp_ch)

        self.fc = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            #elif isinstance(m, nn.InstanceNorm3d):
            #    m.weight.data.fill_(1)
            #    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # nn.Conv2d(self.inplanes, planes * block.expansion,
                #           kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion),
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=(1, 1, 1), stride=(stride, stride, 1), bias=False),
                nn.InstanceNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # [1, 4, 160, 192, 128]
        b, ch, h, w, d = x.size()
        x = self.conv1(x)  # [1, 16, 80, 96, 128]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)  # [2, 16, 49, 49, 229]
        # x1_interp = F.upsample(x, size=(h, w, d), mode='trilinear')
        # x1_interp = self.interp1_fc(x1_interp.permute(0, 4, 2, 3, 1)).permute(0, 4, 2, 3, 1)  # [2, 32, 193, 193, 229]
        # print('shape of x: ', x.shape, 'shape of x1_interp: ', x1_interp.shape)

        x = self.layer2(x)  # [2, 32, 25, 25, 229]
        # x2_interp = F.upsample(x, size=(h, w, d), mode='trilinear')
        # x2_interp = self.interp2_fc(x2_interp.permute(0, 4, 2, 3, 1)).permute(0, 4, 2, 3, 1)  # [2, 32, 193, 193, 229]
        # print('shape of x: ', x.shape, 'shape of x2_interp: ', x2_interp.shape)

        x = self.layer3(x)  # [2, 64, 13, 13, 229]
        # x3_interp = F.upsample(x, size=(h, w, d), mode='trilinear')
        # x3_interp = self.interp3_fc(x3_interp.permute(0, 4, 2, 3, 1)).permute(0, 4, 2, 3, 1)  # [2, 32, 193, 193, 229]
        # print('shape of x: ', x.shape, 'shape of x3_interp: ', x3_interp.shape)

        x = self.layer4(x)  # [2, 128, 7, 7, 229]
        # print('shape of x: ', x.shape)

        x = nn.AvgPool3d(kernel_size=(x.size(2), x.size(3), 1), stride=1)(x)  # [2, 128, 1, 1, 229]
        # print('shape of x: ', x.shape)

        x = x.view(x.size(0), -1, x.size(-1))  # [2, 128, 229]
        # print('shape of x: ', x.shape)

        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)  # [2, 64, 229]
        # print('shape of x: ', x.shape)

        # return x, x1_interp, x2_interp, x3_interp
        return x
