""" GraspNet baseline model definition.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import json
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from models.backbone_resunet14 import MinkUNet14D
from models.modules import ApproachNet, GraspableNet, CloudCrop, SWADNet, CloudCrop_multi_scale,ResBlock
from loss_utils import GRASP_MAX_WIDTH, NUM_VIEW, NUM_ANGLE, NUM_DEPTH, GRASPNESS_THRESHOLD, M_POINT
from label_generation import process_grasp_labels, match_grasp_view_and_label, batch_viewpoint_params_to_matrix
from pointnet2.pointnet2_utils import furthest_point_sample, gather_operation,three_interpolate,three_nn


class GraspNet(nn.Module):

    def __init__(self, cylinder_radius=0.05, seed_feat_dim=512, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.num_depth = NUM_DEPTH
        self.num_angle = NUM_ANGLE
        self.M_points = M_POINT
        self.num_view = NUM_VIEW

        self.backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)


        self.resblock=ResBlock(seed_feature_dim=seed_feat_dim)

        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)                       #用1×1卷积卷积实现MLP 得到 objectness_score graspness_score
        self.rotation = ApproachNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        self.crop = CloudCrop_multi_scale(nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim)
        self.swad = SWADNet(num_angle=self.num_angle, num_depth=self.num_depth)

    def forward(self, end_points):     
        ###### 训练时运行一下代码
        seed_xyz = end_points['point_clouds']  # use all sampled point cloud, B*Ns*3
        # print("seed_xyz",seed_xyz.size(),type(seed_xyz))   tensor
        B, point_num, _ = seed_xyz.shape  # batch _size     2,15000,3

        # point-wise features
        coordinates_batch = end_points['coors']
        features_batch = end_points['feats']      #逐点特征
        # print("xxxxxx",coordinates_batch.size(),coordinates_batch)
        # print("yyyy",features_batch.size(),features_batch)
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)     #得到一个稀疏张量
        



        # #########   debug时运行以下代码
        # mink_input=end_points
        # print("x.shape",mink_input.shape,mink_input.type)





        # print("mink_input",type(mink_input),mink_input.shape)
        #送入ResUNet特征提取    
        seed_features = self.backbone(mink_input).F               #骨干网络输出的特征（是否包含坐标）
        seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)   #view改变维度，改变第1维度和第2维度
        # B,C,N    2*512*15000

        res_feature=self.resblock(seed_features)


        #送入GraspableNet       得到单物体抓取得分和多物体抓取得分
        end_points = self.graspable(res_feature, end_points)              #用1×1卷积卷积实现MLP 得到 objectness_score graspness_score

        seed_features_flipped = seed_features.transpose(1, 2)  # B*Ns*feat_dim     2*15000*512   转置


        # (B, 3, num_seed)     2,3,15000
        objectness_score = end_points['objectness_score']                   #前两个数据    2,2,15000   对象分类得分
        graspness_score = end_points['graspness_score'].squeeze(1)          #第三个数据    2,1,15000   squeeze降维，结果为  2，15000

        objectness_pred = torch.argmax(objectness_score, 1)  # 2,15000  一组二进制数     #取第一维上最大值索引   dim=1 比较行最大值返回列标，也就取出两个得分维度中的最大值列标

        objectness_mask = (objectness_pred == 1)   # 2,15000  1为ture 0为false                #判断返回的列标是否等于1


        graspness_mask = graspness_score > GRASPNESS_THRESHOLD        #选择逐点抓取得分较大的点生成逐向量景观  返回true或者false
        graspable_mask = objectness_mask & graspness_mask            # 2,15000    true或者false的list    # & 转化为二进制按位运算，两个都为true时返回true    


        seed_features_graspable = []
        seed_xyz_graspable = []
        graspable_num_batch = 0.                    #计算True的个数
        for i in range(B):
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()           #计算cur_mask中true的个数

            cur_feat = seed_features_flipped[i][cur_mask]  # Ns*feat_dim    N*512   N不固定  选出ture的点的特征

            cur_seed_xyz = seed_xyz[i][cur_mask]  # Ns*3    选出对应的坐标   N根据ture的个数确定


            cur_seed_xyz = cur_seed_xyz.unsqueeze(0) # 1*Ns*3    增加一维
            fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)    # 根据满足条件的坐标采样1024个点     (1,1024) 
            # print("fps_idxs",fps_idxs.shape,fps_idxs)

            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()  # 1*3*Ns    转置


            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous() # Ns*3



            cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()  # 1*feat_dim*Ns
            cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous() # feat_dim*Ns

            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)
        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)  # B*Ns*3     可抓取点的坐标
        seed_features_graspable = torch.stack(seed_features_graspable)  # B*feat_dim*Ns   可抓取点的特征
        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['graspable_count_stage1'] = graspable_num_batch / B
        
   



        # 后续特征插值思路
        # 1、将seed_features改成res_features
        # 2、将特征插值加入训练
        # if not self.is_training:
        #     seed_features_graspable=nearest_neighbor_interpolate(seed_xyz_graspable.contiguous(),seed_xyz.contiguous(),res_feature) # 
        seed_features_graspable=nearest_neighbor_interpolate(seed_xyz_graspable.contiguous(),seed_xyz.contiguous(),seed_features)






        ###########送入Approach网络
        end_points, res_feat = self.rotation(seed_features_graspable, end_points)

        seed_features_graspable = seed_features_graspable + res_feat

        if self.is_training:
            end_points = process_grasp_labels(end_points)
            grasp_top_views_rot, end_points = match_grasp_view_and_label(end_points)
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']


        # seed_features_graspable=self.channelattention(seed_features_graspable)
        # seed_features_graspable=self.spatialattention(seed_features_graspable)
        # print("seed_features",seed_features_graspable.shape,seed_features_graspable)   (2, 512,1024)
        group_features = self.crop(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        end_points = self.swad(group_features, end_points)

        return end_points


def pred_decode(end_points):
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        grasp_center = end_points['xyz_graspable'][i].float()

        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_score = grasp_score.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_score, grasp_score_inds = torch.max(grasp_score, -1)  # [M_POINT]
        grasp_score = grasp_score.view(-1, 1)
        grasp_angle = (grasp_score_inds // NUM_DEPTH) * np.pi / 12
        grasp_depth = (grasp_score_inds % NUM_DEPTH + 1) * 0.01
        grasp_depth = grasp_depth.view(-1, 1)
        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.
        grasp_width = grasp_width.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))
        grasp_width = torch.clamp(grasp_width, min=0., max=GRASP_MAX_WIDTH)

        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
        grasp_rot = grasp_rot.view(M_POINT, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1))
    return grasp_preds

# known 表示已知点的位置信息 [m,4]
# known_feats 表示已知点的特征信息 [m,C]
# unknown 表示需要插值点的位置信息 [n,4]，一般来所，n>m
# interpolated_feats 表示需要插值点的特征信息 [n,C]，这是返回结果
def nearest_neighbor_interpolate(unknown, known, known_feats):
    """
    :param pts: (n, 4) tensor of the bxyz positions of the unknown features
    :param ctr: (m, 4) tensor of the bxyz positions of the known features
    :param ctr_feats: (m, C) tensor of features to be propigated
    :return:
        new_features: (n, C) tensor of the features of the unknown features
    """
    # 获取 unknown 和 known 之间的近邻关系和距离信息
    dist, idx = three_nn(unknown, known)
    # 权值是距离的倒数
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm
    # 根据近邻关系以及距离信息，直接插值特征信息
    interpolated_feats = three_interpolate(known_feats.contiguous(), idx, weight)

    return interpolated_feats

