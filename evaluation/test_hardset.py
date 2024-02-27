import pickle
import random
import time
import logging
import os
import sys
from multiprocessing import Pool

import numpy as np
import trimesh
from tqdm import tqdm
import pybullet as p
import pandas as pd
import argparse

from dataset.Dataset import Dataset
from dataset.generate_data import show_grasp_heatmap
from dataset.utils import compute_part_ids
from evaluation.hardset_list import entry_paths


if __name__ == '__main__':
    entry_paths = pickle.load(open('./data/dataset/hard_entry_paths_q2.pkl', 'rb'))
    # entry_paths = pickle.load(open('./evaluation/easy_entry_paths_t.pkl', 'rb'))
    print('len(entry_paths)', len(entry_paths))
    obj_configs = []
    for path in entry_paths:
        object_id = path.split('/')[-2]
        config_id = path.split('/')[-1].split('_')[1]
        obj_configs.append((object_id, config_id))

    modes = ['analytical_pos', 'ours_1706605305.8925593', 
             'ours_1706613034.3918543', 'ours_1706620680.2738318', 'ours_1706628441.2965753', 
             'graspnet', 'vgn']

    top_n = 5

    dfs = []
    save_paths = []
    for mode in modes:
        save_path = f'/home/gdk/Repositories/DualArmManipulation/evaluation/results/{mode}_v2.pkl'
        save_paths.append(save_path)
        load_path = save_path
        print(f'Loading {load_path}')
        df = pd.read_pickle(load_path)
        dfs.append(df)

    test_indices = []

    for object_id, config_id in obj_configs:
        df = dfs[0]
        ii = df[(df['object_id'] == object_id) & (df['config_id'] == config_id)].index.to_list()
        test_indices.extend(ii)


    for m, mode in enumerate(modes):
        print(mode)
        df = dfs[m].loc[test_indices]
        n_success = len(df[(df['success'] == 1) & (df['top_n'] == 1)])
        n_total = len(df[(df['success'] > -1) & (df['top_n'] == 1)])
        print(f'Top_1 Success rate: {n_success / n_total} ({n_success}/{n_total})')

        if top_n > 1:
            success_mask = df[df['top_n'] == 1]['success'] == 2  # all False
            success_mask = success_mask.to_numpy()
            for i in range(top_n):
                success_mask_i = df[df['top_n'] == i + 1]['success'] == 1
                success_mask_i = success_mask_i.to_numpy()
                success_mask = success_mask | success_mask_i
            n_success = success_mask.sum()
            print(f'Top_{top_n} Success rate: {n_success / n_total} ({n_success}/{n_total})')
