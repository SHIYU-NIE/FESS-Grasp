import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import pointnet2.pytorch_utils as pt_utils
from pointnet2.pointnet2_utils import CylinderQueryAndGroup
from loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix
import numpy as np



class ResBlock(nn.Module):
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        # self.conv_graspable = nn.Conv1d(self.in_dim, 3, 1)
        self.seed_feature=None

        self.net1 = nn.Sequential(
            
            nn.Conv1d(in_channels=self.in_dim, out_channels=self.in_dim,kernel_size=1),
            nn.BatchNorm1d(self.in_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.5)
        )
        self.net2 = nn.Sequential(

            nn.Conv1d(in_channels=self.in_dim, out_channels=self.in_dim,kernel_size=1),
            nn.BatchNorm1d(self.in_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.in_dim, out_channels=self.in_dim,kernel_size=1),
        
            nn.BatchNorm1d(self.in_dim),

            )
        self.relu=nn.ReLU()

    def forward(self,seed_features):
        res_features=self.net1(seed_features)
        res_features=self.net2(res_features)
        res_features=res_features+seed_features
        res_features=self.relu(res_features)
        return res_features

class GraspableNet(nn.Module):       #用1×1卷积卷积实现MLP 得到 objectness_score graspness_score
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv_graspable = nn.Conv1d(self.in_dim, 3, 1)
        # self.seed_feature=None

        # self.net1 = nn.Sequential(
            
        #     nn.Conv1d(in_channels=self.in_dim, out_channels=self.in_dim,kernel_size=1),
        #     nn.BatchNorm1d(self.in_dim),
        #     nn.ReLU(),
        #     # nn.Dropout(p=0.5)
        # )
        # self.net2 = nn.Sequential(

        #     nn.Conv1d(in_channels=self.in_dim, out_channels=self.in_dim,kernel_size=1),
        #     nn.BatchNorm1d(self.in_dim),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=self.in_dim, out_channels=self.in_dim,kernel_size=1),
        
        #     nn.BatchNorm1d(self.in_dim),

        #     )
        # self.net3=nn.Sequential(
        #     nn.ReLU(),
        #     # nn.Dropout(p=0.5),
        #     nn.Conv1d(in_channels=self.in_dim,out_channels=3,kernel_size=1)

        # )
        
    

    def forward(self, seed_features, end_points):
            
        # graspable_score=self.net1(seed_features)
        # graspable_score=self.net2(graspable_score)

        # graspable_score=graspable_score+seed_features

        # graspable_score=self.net3(seed_features)
        graspable_score = self.conv_graspable(seed_features)  # (B, 3, num_seed)     2,3,15000
        end_points['objectness_score'] = graspable_score[:, :2]     #取所有维度的第 0，1个数据
        end_points['graspness_score'] = graspable_score[:, 2]       #取所有维度的第2和数据
        return end_points


class ApproachNet(nn.Module):
    def __init__(self, num_view, seed_feature_dim, is_training=True):      # 300，512
        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.is_training = is_training
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view, 1)

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()
        #print(seed_features.size())       ################   (2,512,1024)


        res_features = F.relu(self.conv1(seed_features), inplace=True)
        features = self.conv2(res_features)                   #(2,300,1024)
        # print("feature=",features.shape,features)

        view_score = features.transpose(1, 2).contiguous() # (B, num_seed, num_view)   (2,1024,300) 转置
        # print("view_score=",view_score.shape,view_score)

        end_points['view_score'] = view_score

        if self.is_training:

            # normalize view graspness score to 0~1    归一化
            view_score_ = view_score.clone().detach()     #复制一份数据
            view_score_max, _ = torch.max(view_score_, dim=2)    #(2,1024)
            view_score_min, _ = torch.min(view_score_, dim=2)    # 取300个向量中得分最高的



            view_score_max = view_score_max.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_min = view_score_min.unsqueeze(-1).expand(-1, -1, self.num_view)

            view_score_ = (view_score_ - view_score_min) / (view_score_max - view_score_min + 1e-8)    #归一化# (2,512,1024)  归一化得分


            
            top_view_inds = []
            for i in range(B):
                top_view_inds_batch = torch.multinomial(view_score_[i], 1, replacement=False)    # 1024,1     按照view_score_分数的权重每组采样一个score，每个点采样一个score
                top_view_inds.append(top_view_inds_batch)
            top_view_inds = torch.stack(top_view_inds, dim=0).squeeze(-1)  # B, num_seed     2,1024
            #print("top_view_inds",top_view_inds.shape,top_view_inds)
        else:
            _, top_view_inds = torch.max(view_score, dim=2)  # (B, num_seed)

            top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
            template_views = generate_grasp_views(self.num_view).to(features.device)  # (num_view, 3)
            template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous()
            vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (B, num_seed, 3)
            vp_xyz_ = vp_xyz.view(-1, 3)
            batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
            vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
            end_points['grasp_top_view_xyz'] = vp_xyz
            end_points['grasp_top_view_rot'] = vp_rot

        end_points['grasp_top_view_inds'] = top_view_inds
        return end_points, res_features


class CloudCrop_multi_scale(nn.Module):
    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=0.04):
        super().__init__()
        self.crop1 = CloudCrop(nsample, seed_feature_dim, cylinder_radius * 0.25, hmin, hmax)
        self.crop2 = CloudCrop(nsample, seed_feature_dim, cylinder_radius * 0.5, hmin, hmax)
        self.crop3 = CloudCrop(nsample, seed_feature_dim, cylinder_radius * 0.75, hmin, hmax)
        self.crop4 = CloudCrop(nsample, seed_feature_dim, cylinder_radius, hmin, hmax)
        self.fuse_multi_scale = nn.Conv1d(256 * 4, 256, 1)  # 输入1024 输出256的MLP层
        self.gate_fusion = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.Sigmoid()
        )
        self.seed_features_transform = nn.Conv1d(512, 256, 1)

    def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):
        # 2 1024 3 #2 512 1024 #2 1024 3 3
        vp_features1 = self.crop1(seed_xyz_graspable, seed_features_graspable, vp_rot)  # 2 256 1024
        vp_features2 = self.crop2(seed_xyz_graspable, seed_features_graspable, vp_rot)
        vp_features3 = self.crop3(seed_xyz_graspable, seed_features_graspable, vp_rot)
        vp_features4 = self.crop4(seed_xyz_graspable, seed_features_graspable, vp_rot)

        B, _, num_seed = vp_features1.size()
        vp_features_concat = torch.cat([vp_features1, vp_features2, vp_features3, vp_features4], dim=1)  # 2 1024 1024
        vp_features_concat = vp_features_concat.view(B, -1, num_seed)  # 2 1024 1024
        vp_features_concat = self.fuse_multi_scale(vp_features_concat)  # 2 256 1024
        vp_features_concat = vp_features_concat.view(B, -1, num_seed)  # 2 256 1024

        seed_features_graspable = self.seed_features_transform(seed_features_graspable)  # 512 to 256 可改进点

        #  seed_features_graspable 2 512 1024
        seed_features_gate = self.gate_fusion(seed_features_graspable) * seed_features_graspable  # 2 256 1024
        # print("seed_features_gate:", seed_features_gate.size())
        # seed_features_gate = seed_features_gate.unsqueeze(3).repeat(1, 1, 1, 4)  # 2 512 1024 4
        vp_features = vp_features_concat + seed_features_gate
        return vp_features



class CloudCrop(nn.Module):
    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=0.04):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [3 + self.in_dim, 256, 256]   # use xyz, so plus 3

        self.grouper = CylinderQueryAndGroup(radius=cylinder_radius, hmin=hmin, hmax=hmax, nsample=nsample,
                                             use_xyz=True, normalize_xyz=True)
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):
        grouped_feature = self.grouper(seed_xyz_graspable, seed_xyz_graspable, vp_rot,
                                       seed_features_graspable)  # B*3 + feat_dim*M*K
        new_features = self.mlps(grouped_feature)  # (batch_size, mlps[-1], M, K)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (batch_size, mlps[-1], M, 1)
        new_features = new_features.squeeze(-1)   # (batch_size, mlps[-1], M)
        return new_features


class SWADNet(nn.Module):
    def __init__(self, num_angle, num_depth):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth

        self.conv1 = nn.Conv1d(256, 256, 1)  # input feat dim need to be consistent with CloudCrop module
        self.conv_swad = nn.Conv1d(256, 2*num_angle*num_depth, 1)

    def forward(self, vp_features, end_points):
        B, _, num_seed = vp_features.size()
        vp_features = F.relu(self.conv1(vp_features), inplace=True)
        vp_features = self.conv_swad(vp_features)
        vp_features = vp_features.view(B, 2, self.num_angle, self.num_depth, num_seed)
        vp_features = vp_features.permute(0, 1, 4, 2, 3)

        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0]  # B * num_seed * num angle * num_depth
        end_points['grasp_width_pred'] = vp_features[:, 1]
        return end_points
