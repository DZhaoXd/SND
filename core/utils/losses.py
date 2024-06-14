# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import errno
import os
import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.init as initer

import torch.nn.functional as F


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict
    
def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))
    
class BinaryCrossEntropy(torch.nn.Module):

    def __init__(self, size_average=True, ignore_index=255):
        super(BinaryCrossEntropy, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_index

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.binary_cross_entropy_with_logits(predict, target.unsqueeze(-1), pos_weight=weight, size_average=self.size_average)
        return loss
        
# refer to https://github.com/visinf/da-sac
def pseudo_labels_probs(probs, running_conf, THRESHOLD_BETA, RUN_CONF_UPPER=0.80, ignore_augm=None, discount = True):
    """Consider top % pixel w.r.t. each image"""
    
    RUN_CONF_UPPER = RUN_CONF_UPPER
    RUN_CONF_LOWER = 0.20
    
    B,C,H,W = probs.size()
    max_conf, max_idx = probs.max(1, keepdim=True) # B,1,H,W

    probs_peaks = torch.zeros_like(probs)
    probs_peaks.scatter_(1, max_idx, max_conf) # B,C,H,W
    top_peaks, _ = probs_peaks.view(B,C,-1).max(-1) # B,C
    
    # top_peaks 是一张图上每个类的最大置信度
    top_peaks *= RUN_CONF_UPPER

    if discount:
        # discount threshold for long-tail classes
        top_peaks *= (1. - torch.exp(- running_conf / THRESHOLD_BETA)).view(1, C)

    top_peaks.clamp_(RUN_CONF_LOWER) # in-place
    probs_peaks.gt_(top_peaks.view(B,C,1,1))

    # ignore if lower than the discounted peaks
    ignore = probs_peaks.sum(1, keepdim=True) != 1

    # thresholding the most confident pixels
    pseudo_labels = max_idx.clone()
    pseudo_labels[ignore] = 255

    pseudo_labels = pseudo_labels.squeeze(1)
    #pseudo_labels[ignore_augm] = 255

    return pseudo_labels, max_conf, max_idx

# refer to https://github.com/visinf/da-sac
def update_running_conf(probs, running_conf, THRESHOLD_BETA, tolerance=1e-8):
    """Maintain the moving class prior"""
    STAT_MOMENTUM = 0.9
    
    B,C,H,W = probs.size()
    probs_avg = probs.mean(0).view(C,-1).mean(-1)

    # updating the new records: copy the value
    update_index = probs_avg > tolerance
    new_index = update_index & (running_conf == THRESHOLD_BETA)
    running_conf[new_index] = probs_avg[new_index]

    # use the moving average for the rest (Eq. 2)
    running_conf *= STAT_MOMENTUM
    running_conf += (1 - STAT_MOMENTUM) * probs_avg
    return running_conf

 
def full2weak(feat, target_weak_params, down_ratio=1, nearest=False):
    tmp = []
    for i in range(feat.shape[0]):
        #### rescale
        h, w = target_weak_params['rescale_size'][0][i], target_weak_params['rescale_size'][1][i]
        if nearest:
            feat_ = F.interpolate(feat[i:i+1], size=[int(h/down_ratio), int(w/down_ratio)])
        else:
            feat_ = F.interpolate(feat[i:i+1], size=[int(h/down_ratio), int(w/down_ratio)], mode='bilinear', align_corners=True)
        #### then crop
        y1, y2, x1, x2 = target_weak_params['random_crop_axis'][0][i], target_weak_params['random_crop_axis'][1][i], target_weak_params['random_crop_axis'][2][i], target_weak_params['random_crop_axis'][3][i]
        y1, th, x1, tw = int(y1/down_ratio), int((y2-y1)/down_ratio), int(x1/down_ratio), int((x2-x1)/down_ratio)
        feat_ = feat_[:, :, y1:y1+th, x1:x1+tw]
        if target_weak_params['RandomHorizontalFlip'][i]:
            inv_idx = torch.arange(feat_.size(3)-1,-1,-1).long().to(feat_.device)
            feat_ = feat_.index_select(3,inv_idx)
        tmp.append(feat_)
    feat = torch.cat(tmp, 0)
    return feat

def entropy(p, prob=True, mean=True):
    if prob:
        p = F.softmax(p, dim=1)
    en = -torch.sum(p * torch.log(p + 1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en
        
