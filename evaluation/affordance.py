import torch
import torch.nn as nn
from dataset.LVDataset import LVDataset
# from dataset.Dataset import Config
from model.model import Net
from model.modelGlobal import NetG
import logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from model.utils import plot, save_loss
from model.data_utils import get_dataloader, get_dataloader_special, ramdom_sample_pos
import argparse
import time
import os
import pickle
import trimesh
import json
import numpy as np

mp.set_start_method('spawn', force=True)
VISION_LOCAL_THRESHOLD = 1000.0



# KL Divergence
# kld(map2||map1) -- map2 is gt 
def KLD(map1, map2, eps = 1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    kld = np.sum(map2*np.log( map2/(map1+eps) + eps))
    return kld

# historgram intersection
def SIM(map1, map2, eps=1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)

def AUC_Judd(saliency_map, fixation_map, jitter=True):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape)
    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        saliency_map += np.random.rand(*saliency_map.shape) * 1e-7
    # Normalize saliency map to have values between [0,1]
    saliency_map =  (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map) + 1e-12)

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds)+2)
    fp = np.zeros(len(thresholds)+2)
    tp[0] = 0; tp[-1] = 1
    fp[0] = 0; fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh) # Total number of saliency map values above threshold
        tp[k+1] = (k + 1) / float(n_fix) # Ratio saliency map values at fixation locations above threshold
        fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix) # Ratio other saliency map values above threshold
    return np.trapz(tp, fp) # y, x
    
def main(params):
    kld_list = []
    sim_list = []
    auc_list = []
    kld_list_vgn = []
    sim_list_vgn = []
    auc_list_vgn = []
    vgn_path = 'data_vgn_map.pickle'
    with open(vgn_path, 'rb') as f:
        vgn_vol_map = pickle.load(f)
        print(len(vgn_vol_map))

    for i in range(303):
        maps_path = f'./checkpoints/maps_{params["model_id"]}_{i}.pkl'
        if os.path.exists(maps_path) == False:
            print(f'File {maps_path} does not exist')
            continue
        with open(maps_path, 'rb') as f:
            maps = pickle.load(f)
            gt = maps['grasp_map'] # (batch_size, 2048), 100
            pred = maps['prediction']
            pc = maps['point_cloud']
            entry_paths = maps['entry_path']
            pred[pred < 0.] = 0.
            pred[pred > 1.] = 1.
            assert (pred >= 0.).all(), 'Negative value in prediction'
            assert (pred <= 1.).all(), 'Value greater than 1 in prediction'
            for k in range(gt.shape[0]):
                kld_list.append(KLD(pred[k], gt[k]))
                sim_list.append(SIM(pred[k], gt[k]))
                auc_list.append(AUC_Judd(pred[k], gt[k]))
                object_id = entry_paths[k].split('/')[-2]
                vgn_vol = vgn_vol_map[object_id]
                vgn_pred = map_convert(vgn_vol, pc[k])
                kld_list_vgn.append(KLD(vgn_pred, gt[k]))
                sim_list_vgn.append(SIM(vgn_pred, gt[k]))
                auc_list_vgn.append(AUC_Judd(vgn_pred, gt[k]))
    print(f'KL Divergence: {np.mean(kld_list)}')
    print(f'Histogram Intersection: {np.mean(sim_list)}')
    print(f'AUC Judd: {np.mean(auc_list)}')

    print(f'KL Divergence VGN: {np.mean(kld_list_vgn)}')
    print(f'Histogram Intersection VGN: {np.mean(sim_list_vgn)}')
    print(f'AUC Judd VGN: {np.mean(auc_list_vgn)}')
                
def map_convert(vgn_vol, point_cloud):
    if len(vgn_vol) == 0:
        return np.ones(point_cloud.shape[0]) / point_cloud.shape[0]
    # normalize point cloud
    point_cloud = point_cloud - point_cloud.min(0)
    point_cloud = point_cloud / point_cloud.max(0)
    # voxel grid: 40x40x40
    heatmap = np.zeros(point_cloud.shape[0])
    point_cloud = point_cloud * 39
    for i in range(point_cloud.shape[0]):
        x, y, z = point_cloud[i].astype(int)
        heatmap[i] = vgn_vol[x, y, z]
    # heatmap[heatmap < 0.] = 0.
    # heatmap[heatmap > 1.] = 1.
    if heatmap.sum() == 0:
        return np.ones(point_cloud.shape[0]) / point_cloud.shape[0]
    heatmap = heatmap / heatmap.sum()
    return heatmap

if __name__ == '__main__':
    # vgn_path = 'data_vgn_map.pickle'
    # with open(vgn_path, 'rb') as f:
    #     vgn = pickle.load(f)
    #     print(len(vgn))
        # for k, v in vgn.items():
        #     print(len(v))
    # exit()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset_dir', type=str, default='./data/objects/')
    argparser.add_argument('--epoch_id', type=int, default=16)
    argparser.add_argument('--model_id', type=str, default='')
    argparser.add_argument('--global', default=False, action='store_true')
    argparser.add_argument('--gt', default=False, action='store_true')

    args = argparser.parse_args()
    params = vars(args)
    main(params)