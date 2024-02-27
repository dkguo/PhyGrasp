import torch
import torch.nn as nn
from dataset.LVDataset import LVDataset
# from dataset.Dataset import Config
from model.model_ab_vg import NetABVG
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

def evaluate(model: NetABVG, test_loader, device, params):
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            language, point_cloud, vision_global, vision_local, grasp_map, pos_index, neg_index, pos_neg_num, entry_paths = data
            language = language.to(device).float()
            point_cloud = point_cloud.to(device).float()
            vision_global = vision_global.to(device).float()
            vision_local = vision_local.to(device).float()
            grasp_map = grasp_map.to(device).float()
            pos_index = pos_index.to(device).long()
            neg_index = neg_index.to(device).long()
            pos_neg_num = pos_neg_num.to(device).long()
            point_cloud_copy = point_cloud.clone().detach()

            vision_local_mean = torch.mean(vision_local, dim=(1,2))
            if torch.any(vision_local_mean > VISION_LOCAL_THRESHOLD):
                index = torch.nonzero(vision_local_mean > VISION_LOCAL_THRESHOLD)
                for i in index:
                    print("invalid data entry path", entry_paths[i])
                    print("vision_local mean", vision_local_mean[i])
                    print("vision_local", vision_local[i])
                continue
            
            if params['global']:
                output_global = model(language, point_cloud, vision_global, vision_local)
            else:
                output_global, output_local = model(language, point_cloud, vision_global, vision_local)
                loss_global, loss_local_pos, loss_local_neg, _, _ = model.get_loss(output_global, output_local, grasp_map,
                                                                             pos_index, neg_index, pos_neg_num,
                                                                             params['delta_v'], params['delta_d'])
                index1 = torch.topk(output_global.squeeze(), k=params['kp1']).indices # (batch_size, kp1)
                score = model.get_score(output_local, index1) # (batch_size, kp1, 2048, 1)
                pos = model.get_pos(output_local, index1, point_cloud_copy) # (batch_size, kp1, 6)
                pos = ramdom_sample_pos(pos) # (batch_size, 5, 6)
                
                print(f'Evaluating Batch {i}, Loss Global {loss_global.item()}, Loss Local Pos {loss_local_pos.item()}, Loss Local Neg {loss_local_neg.item()}')

            if params['maps_save']:
                maps = {
                    # 'point_cloud': point_cloud.squeeze().cpu().numpy(), # (batch_size, 2048, 3)
                    'point_cloud': point_cloud_copy.squeeze().cpu().numpy(), # (batch_size, 2048, 3)
                    'grasp_map': grasp_map.squeeze().cpu().numpy(), # (batch_size, 2048)
                    'prediction': output_global.squeeze().cpu().numpy(), # (batch_size, 2048)
                    'embeddings': output_local.squeeze().cpu().numpy(), # (batch_size, 2048, 32)
                    'entry_path': entry_paths,
                    'index20': index1.squeeze().cpu().numpy(), # (batch_size, kp1)
                    # 'index1': index_1.squeeze().cpu().numpy(), # (batch_size,)
                    'score': score.squeeze().cpu().numpy(),
                    'pos': pos.squeeze().cpu().numpy(), # (batch_size, 5, 6)
                }
                pickle.dump(maps, open('./checkpoints/maps_{}_{}.pkl'.format(params['model_id'], i), 'wb'))
                # print('Saved maps_{}_{}.pkl'.format(params['model_id'], i))
    
def main(params):
    _, _, test_loader = get_dataloader(params)
    # test_loader = get_dataloader_special(params)
    if params['global']:
        checkpoint_path = './checkpoints/modelGlobal_{}_{}.pth'.format(params['model_id'], params['epoch_id'])
        model = NetG()
    else:   
        checkpoint_path = './checkpoints/model_{}_{}.pth'.format(params['model_id'], params['epoch_id']) 
        model = NetABVG()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    # torch.set_printoptions(threshold=1000000)
    device = torch.device('cuda:0')
    evaluate(model, test_loader, device, params)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset_dir', type=str, default='./data/objects/')
    argparser.add_argument('--shuffle', type=bool, default=False)
    argparser.add_argument('--epoch_id', type=int, default=16)
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--num_workers', type=int, default=16)
    argparser.add_argument('--lr', type=float, default=2e-3)
    # argparser.add_argument('--object_id', type=str, default='42')
    # argparser.add_argument('--config_id', type=str, default='1704368644.959488')

    argparser.add_argument('--model_id', type=str, default='')
    argparser.add_argument('--global', default=False, action='store_true')
    argparser.add_argument('--gt', default=False, action='store_true')
    argparser.add_argument('--maps_save', default=False, action='store_true')
    argparser.add_argument('--delta_v', type=float, default=0.5)
    argparser.add_argument('--delta_d', type=float, default=3.0)
    argparser.add_argument('--kp1', type=int, default=20)
    argparser.add_argument('--kp2', type=int, default=10)
    # 1000_1704368410.6270053

    args = argparser.parse_args()
    params = vars(args)
    # params['model_id'] = time.time()
    main(params)