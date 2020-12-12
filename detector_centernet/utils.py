# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import torch
import numpy as np
import torch.nn as nn


def get_affine_transform(center, scale, rot, output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inverse=0):
    """
    根据设置的中心点和尺寸等参数，计算仿射矩阵
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = _get_direction([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inverse:
        trans_matrix = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans_matrix = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans_matrix


def ctdet_decode(heat, wh, reg=None, k=100):
    """
    模型输出结果解码，转化为框、置信度等
    """
    batch, category, height, width = heat.size()
    heat = _nms(heat)
    # ys、xs是inds转化在热图上的列、行
    scores, indexs, classes, ys, xs = _topk(heat, k=k)
    # 增加偏移结果
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, indexs)
        reg = reg.view(batch, k, 2)
        xs = xs.view(batch, k, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, k, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, k, 1) + 0.5
        ys = ys.view(batch, k, 1) + 0.5
    # 取宽高中对应结果
    wh = _tranpose_and_gather_feat(wh, indexs)
    wh = wh.view(batch, k, 2)
    classes = classes.view(batch, k, 1).float()
    scores = scores.view(batch, k, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, classes], dim=2)
    return detections


def ctdet_post_process(dets, center, scale, h, w, num_classes):
    """
    后处理过程
    """
    result = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = _transform_preds(
            dets[i, :, 0:2], center[i], scale[i], (w, h))
        dets[i, :, 2:4] = _transform_preds(
            dets[i, :, 2:4], center[i], scale[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
        result.append(top_preds)
    return result


def _get_3rd_point(a, b):
    """
    仿射变换-获取第三个参照点坐标
    """
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_direction(src_point, rot_rad):
    """
    仿射变换-计算变换方向
    """
    sin, cos = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cos - src_point[1] * sin
    src_result[1] = src_point[0] * sin + src_point[1] * cos
    return src_result


def _nms(heatmap, kernel=3):
    """
    非极大抑制处理，筛选出极大值点为原值，其余为0
    """
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def _topk(scores, k=40):
    """
    筛选检测结果中热图置信度最高的前K个结果
    返回前K个结果对应的分数、索引、类别、列、行
    """
    batch, category, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, category, -1), k)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), k)
    topk_clses = (topk_ind / k).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, k)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, k)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, k)
    # 前K个结果对应的分数、索引、类别、列、行
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _gather_feat(feat, ind):
    """
    取出对应索引的结果
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat


def _tranpose_and_gather_feat(feat, index):
    """
    进行维度转换，转换为batch*(W*H)*C
    并在特征中取出对应索引的结果
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, index)
    return feat


def _affine_transform(coords, affine_matrix):
    """
    根据变换矩阵对坐标进行仿射变换
    """
    new_pt = np.array([coords[0], coords[1], 1.], dtype=np.float32).T
    new_pt = np.dot(affine_matrix, new_pt)
    return new_pt[:2]


def _transform_preds(coords, center, scale, output_size):
    """
    将原始坐标根据仿射变换矩阵转为目标坐标
    """
    target_coords = np.zeros(coords.shape)
    trans_matrix = get_affine_transform(center, scale,
                                        0, output_size, inverse=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = _affine_transform(coords[p, 0:2], trans_matrix)
    return target_coords
