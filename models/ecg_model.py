from __future__ import print_function
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import sys
import os
from models.model_utils import edge_preserve_sampling, gen_grid_up, get_graph_feature, symmetric_sample, three_nn_upsampling
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
import pointnet2_utils as pn2


class PCN_encoder(nn.Module):
    def __init__(self, output_size=1024):
        super(PCN_encoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, output_size, 1)

    def forward(self, x):
        batch_size, _, num_points = x.size()
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        global_feature, _ = torch.max(x, 2)
        x = torch.cat((x, global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        global_feature, _ = torch.max(x, 2)
        return global_feature.view(batch_size, -1)


class PCN_decoder(nn.Module):
    def __init__(self, num_coarse, num_fine, scale, cat_feature_num):
        super(PCN_decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_coarse * 3)

        self.scale = scale
        self.grid = gen_grid_up(2 ** (int(math.log2(scale))), 0.05).cuda().contiguous()

        self.conv1 = nn.Conv1d(cat_feature_num, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)

    def forward(self, x):
        batch_size = x.size()[0]
        coarse = F.relu(self.fc1(x))
        coarse = F.relu(self.fc2(coarse))
        coarse = self.fc3(coarse).view(-1, 3, self.num_coarse)

        grid = self.grid.clone().detach()
        grid_feat = grid.unsqueeze(0).repeat(batch_size, 1, self.num_coarse).contiguous().cuda()

        # point_feat = coarse.unsqueeze(3).repeat(1, 1, 1, self.scale).view(batch_size, 3, self.num_fine).contiguous()
        point_feat = (
            (coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine,
                                                                                                3)).transpose(1,
                                                                                                              2).contiguous()

        global_feat = x.unsqueeze(2).repeat(1, 1, self.num_fine)

        feat = torch.cat((grid_feat, point_feat, global_feat), 1)

        center = ((coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine,
                                                                                                      3)).transpose(1,
                                                                                                                    2).contiguous()

        fine = self.conv3(F.relu(self.conv2(F.relu(self.conv1(feat))))) + center
        return coarse, fine


class Stack_conv(nn.Module):
    def __init__(self, input_size, output_size, act=None):
        super(Stack_conv, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('conv', nn.Conv2d(input_size, output_size, 1))

        if act is not None:
            self.model.add_module('act', act)

    def forward(self, x):
        y = self.model(x)
        y = torch.cat((x, y), 1)
        return y


class Dense_conv(nn.Module):
    def __init__(self, input_size, growth_rate=64, dense_n=3, k=16):
        super(Dense_conv, self).__init__()
        self.growth_rate = growth_rate
        self.dense_n = dense_n
        self.k = k
        self.comp = growth_rate * 2
        self.input_size = input_size

        self.first_conv = nn.Conv2d(self.input_size * 2, growth_rate, 1)

        self.input_size += self.growth_rate

        self.model = nn.Sequential()
        for i in range(dense_n - 1):
            if i == dense_n - 2:
                self.model.add_module('stack_conv_%d' % (i + 1), Stack_conv(self.input_size, self.growth_rate, None))
            else:
                self.model.add_module('stack_conv_%d' % (i + 1),
                                      Stack_conv(self.input_size, self.growth_rate, nn.ReLU()))
                self.input_size += growth_rate

    def forward(self, x):
        y = get_graph_feature(x, k=self.k)
        y = F.relu(self.first_conv(y))
        y = torch.cat((y, x.unsqueeze(3).repeat(1, 1, 1, self.k)), 1)

        y = self.model(y)
        y, _ = torch.max(y, 3)

        return y


class EF_encoder(nn.Module):
    def __init__(self, growth_rate=24, dense_n=3, k=16, hierarchy=[1024, 256, 64], input_size=3, output_size=256):
        super(EF_encoder, self).__init__()
        self.growth_rate = growth_rate
        self.comp = growth_rate * 2
        self.dense_n = dense_n
        self.k = k
        self.hierarchy = hierarchy

        self.init_channel = 24

        self.conv1 = nn.Conv1d(input_size, self.init_channel, 1)
        self.dense_conv1 = Dense_conv(self.init_channel, self.growth_rate, self.dense_n, self.k)

        out_channel_size_1 = (self.init_channel * 2 + self.growth_rate * self.dense_n)  # 24*2 + 24*3 = 120
        self.conv2 = nn.Conv1d(out_channel_size_1 * 2, self.comp, 1)
        self.dense_conv2 = Dense_conv(self.comp, self.growth_rate, self.dense_n, self.k)

        out_channel_size_2 = (
                    out_channel_size_1 * 2 + self.comp + self.growth_rate * self.dense_n)  # 120*2 + 48 + 24*3 = 360
        self.conv3 = nn.Conv1d(out_channel_size_2 * 2, self.comp, 1)
        self.dense_conv3 = Dense_conv(self.comp, self.growth_rate, self.dense_n, self.k)

        out_channel_size_3 = (
                    out_channel_size_2 * 2 + self.comp + self.growth_rate * self.dense_n)  # 360*2 + 48 + 24*3 = 840
        self.conv4 = nn.Conv1d(out_channel_size_3 * 2, self.comp, 1)
        self.dense_conv4 = Dense_conv(self.comp, self.growth_rate, self.dense_n, self.k)

        out_channel_size_4 = out_channel_size_3 * 2 + self.comp + self.growth_rate * self.dense_n  # 840*2 + 48 + 24*3 = 1800
        self.gf_conv = nn.Conv1d(out_channel_size_4, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1024)

        out_channel_size = out_channel_size_4 + 1024
        self.conv5 = nn.Conv1d(out_channel_size, 1024, 1)

        out_channel_size = out_channel_size_3 + 1024
        self.conv6 = nn.Conv1d(out_channel_size, 768, 1)

        out_channel_size = out_channel_size_2 + 768
        self.conv7 = nn.Conv1d(out_channel_size, 512, 1)

        out_channel_size = out_channel_size_1 + 512
        self.conv8 = nn.Conv1d(out_channel_size, output_size, 1)

    def forward(self, x):
        point_cloud1 = x[:, 0:3, :]
        point_cloud1 = point_cloud1.transpose(1, 2).contiguous()

        x0 = F.relu(self.conv1(x))  # 24
        x1 = F.relu(self.dense_conv1(x0))  # 24 + 24 * 3 = 96
        x1 = torch.cat((x1, x0), 1)  # 120
        x1d, _, _, point_cloud2 = edge_preserve_sampling(x1, point_cloud1, self.hierarchy[0], self.k)  # 240

        x2 = F.relu(self.conv2(x1d))  # 48
        x2 = F.relu(self.dense_conv2(x2))  # 48 + 24 * 3 = 120
        x2 = torch.cat((x2, x1d), 1)  # 120 + 240 = 360
        x2d, _, _, point_cloud3 = edge_preserve_sampling(x2, point_cloud2, self.hierarchy[1], self.k)  # 720

        x3 = F.relu(self.conv3(x2d))
        x3 = F.relu(self.dense_conv3(x3))
        x3 = torch.cat((x3, x2d), 1)
        x3d, _, _, point_cloud4 = edge_preserve_sampling(x3, point_cloud3, self.hierarchy[2], self.k)

        x4 = F.relu(self.conv4(x3d))
        x4 = F.relu(self.dense_conv4(x4))
        x4 = torch.cat((x4, x3d), 1)

        global_feat = self.gf_conv(x4)
        global_feat, _ = torch.max(global_feat, -1)
        global_feat = F.relu(self.fc1(global_feat))
        global_feat = F.relu(self.fc2(global_feat)).unsqueeze(2).repeat(1, 1, self.hierarchy[2])

        x4 = torch.cat((global_feat, x4), 1)
        x4 = F.relu(self.conv5(x4))
        idx, weight = three_nn_upsampling(point_cloud3, point_cloud4)
        x4 = pn2.three_interpolate(x4, idx, weight)

        x3 = torch.cat((x3, x4), 1)
        x3 = F.relu(self.conv6(x3))
        idx, weight = three_nn_upsampling(point_cloud2, point_cloud3)
        x3 = pn2.three_interpolate(x3, idx, weight)

        x2 = torch.cat((x2, x3), 1)
        x2 = F.relu(self.conv7(x2))
        idx, weight = three_nn_upsampling(point_cloud1, point_cloud2)
        x2 = pn2.three_interpolate(x2, idx, weight)

        x1 = torch.cat((x1, x2), 1)
        x1 = self.conv8(x1)
        return x1


class EF_expansion(nn.Module):
    def __init__(self, input_size, output_size=64, step_ratio=2, k=4):
        super(EF_expansion, self).__init__()
        self.step_ratio = step_ratio
        self.k = k
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(input_size * 2, output_size, 1)
        self.conv2 = nn.Conv2d(input_size * 2 + output_size, output_size * step_ratio, 1)
        self.conv3 = nn.Conv2d(output_size, output_size, 1)

    def forward(self, x):
        batch_size, _, num_points = x.size()

        input_edge_feature = get_graph_feature(x, self.k, minus_center=False).permute(0, 1, 3,
                                                                                      2).contiguous()  # B C K N
        edge_feature = self.conv1(input_edge_feature)
        edge_feature = F.relu(torch.cat((edge_feature, input_edge_feature), 1))

        edge_feature = F.relu(self.conv2(edge_feature))  # B C K N
        edge_feature = edge_feature.permute(0, 2, 3, 1).contiguous().view(batch_size, self.k,
                                                                          num_points * self.step_ratio,
                                                                          self.output_size).permute(0, 3, 1,
                                                                                                    2)  # B C K N

        edge_feature = self.conv3(edge_feature)
        edge_feature, _ = torch.max(edge_feature, 2)

        return edge_feature


class ECG_decoder(nn.Module):
    def __init__(self, num_coarse, num_fine, num_input, downsample_im=False, mirror_im=False, points_label=False):
        super(ECG_decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine

        if not downsample_im:
            self.scale = int(np.ceil(num_fine / (num_coarse + num_input)))
        else:
            self.scale = int(np.ceil(num_fine / 2048))

        self.downsample_im = downsample_im
        self.mirror_im = mirror_im
        self.points_label = points_label

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_coarse * 3)

        self.dense_feature_size = 256
        self.expand_feature_size = 64

        if points_label:
            self.input_size = 4
        else:
            self.input_size = 3

        self.encoder = EF_encoder(growth_rate=24, dense_n=3, k=16, hierarchy=[1024, 256, 64],
                                  input_size=self.input_size, output_size=self.dense_feature_size)

        if self.scale >= 2:
            self.expansion = EF_expansion(input_size=self.dense_feature_size, output_size=self.expand_feature_size,
                                          step_ratio=self.scale, k=4)
            self.conv1 = nn.Conv1d(self.expand_feature_size, self.expand_feature_size, 1)
        else:
            self.expansion = None
            self.conv1 = nn.Conv1d(self.dense_feature_size, self.expand_feature_size, 1)
        self.conv2 = nn.Conv1d(self.expand_feature_size, 3, 1)

    def forward(self, global_feat, point_input):
        batch_size = global_feat.size()[0]
        coarse = F.relu(self.fc1(global_feat))
        coarse = F.relu(self.fc2(coarse))
        coarse = self.fc3(coarse).view(batch_size, 3, self.num_coarse)

        if self.downsample_im:
            if self.mirror_im:
                org_points_input = symmetric_sample(point_input.transpose(1, 2).contiguous(),
                                                    int((2048 - self.num_coarse) / 2))
                org_points_input = org_points_input.transpose(1, 2).contiguous()
            else:
                org_points_input = pn2.gather_operation(point_input,
                                                        pn2.furthest_point_sample(
                                                            point_input.transpose(1, 2).contiguous(),
                                                            int(2048 - self.num_coarse)))
        else:
            org_points_input = point_input

        if self.points_label:
            id0 = torch.zeros(coarse.shape[0], 1, coarse.shape[2]).cuda().contiguous()
            coarse_input = torch.cat((coarse, id0), 1)
            id1 = torch.ones(org_points_input.shape[0], 1, org_points_input.shape[2]).cuda().contiguous()
            org_points_input = torch.cat((org_points_input, id1), 1)
            points = torch.cat((coarse_input, org_points_input), 2)
        else:
            points = torch.cat((coarse, org_points_input), 2)

        dense_feat = self.encoder(points)

        if self.scale >= 2:
            dense_feat = self.expansion(dense_feat)

        point_feat = F.relu(self.conv1(dense_feat))
        fine = self.conv2(point_feat)

        num_out = fine.size()[2]
        if num_out > self.num_fine:
            fine = pn2.gather_operation(fine,
                                        pn2.furthest_point_sample(fine.transpose(1, 2).contiguous(), self.num_fine))

        return coarse, fine


class ECG(nn.Module):
    def __init__(self, num_coarse=1024, num_fine=2048, num_input=2048, downsample_im=False, mirror_im=False,
                 points_label=False):
        super(ECG, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        # self.scale = num_fine // num_coarse
        self.downsample_im = downsample_im
        self.mirror_im = mirror_im

        self.encoder = PCN_encoder()
        self.decoder = ECG_decoder(num_coarse, num_fine, num_input, downsample_im, mirror_im, points_label)

    def forward(self, x):
        feat = self.encoder(x)
        coarse, fine = self.decoder(feat, x)
        coarse = coarse.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()
        return coarse, fine
