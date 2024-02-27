import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import pickle

from dataset.Dataset import Dataset


if __name__ == '__main__':
    # modes = ['ours_1706605305.8925593', 'ours_1706613034.3918543',
    #          'ours_1706620680.2738318',	'ours_1706628441.2965753']
    modes = ['ours_1706605305.8925593wol']
    # modes = ['vgn']
    save_names = []
    grasp_datas = []
    for mode in modes:
        if mode == 'analytical_pos':
            save_name = 'analytical_pos_not_tested'
            grasp_data = {}
        elif mode == 'analytical_neg':
            save_name = 'analytical_neg_not_tested'
            grasp_data = {}
        elif mode == 'graspnet':
            save_name = 'graspnet_not_tested'
            grasp_data = pickle.load(open('./evaluation/results/graspnet.pickle', 'rb'))
        elif mode == 'vgn':
            save_name = 'vgn_not_tested'
            grasp_data = pickle.load(open('./data_vgn_grasp.pickle', 'rb'))
        elif 'ours' in mode:
            model_id = mode.split('_')[1]
            save_name = f'ours_{model_id}_not_tested'
            grasp_data = {}
            for i in range(303):
                if not os.path.exists(f'./checkpoints/maps_{model_id}_{i}.pkl'):
                    continue
                test_dataset = pickle.load(open(f'./checkpoints/maps_{model_id}_{i}.pkl', 'rb'))
                if 'pos' not in test_dataset:
                    print('no pos in map', i)
                    continue
                paths = test_dataset['entry_path']
                positions = test_dataset['pos']
                assert len(paths) == len(positions)
                for path, position in zip(paths, positions):
                    object_id = path.split('/')[-2]
                    config_id = path.split('/')[-1].split('_')[1]
                    if 'pkl' in config_id:
                        continue
                    grasp_data[(object_id, config_id)] = position   #(n, 6)
        save_names.append(save_name)
        grasp_datas.append(grasp_data)
    
        
    source_df = pd.read_csv('./data/dataset/test_object_config_ids.csv', dtype = str).sort_values(['object_id'], ignore_index=True)#[:10]

    dfs = []
    for _ in range(len(modes)):
        df = pd.DataFrame(
            index=range(len(source_df) * 10),
            columns=['object_id', 'config_id', 'f1', 'p1', 'n1', 'f2', 'p2', 'n2', 'obj_mass', 'top_n', 'success']
        )
        df['top_n'] = -1
        df['success'] = -1
        df = df.astype({'object_id': str, 'config_id': str, 'f1': float, 'p1': object, 'n1': object, 'f2': float, 'p2': object, 'n2': object, 'obj_mass': float, 'top_n': int, 'success': int})
        dfs.append(df)

    # print(df)

    dataset = Dataset('/home/gdk/Repositories/DualArmManipulation/data/objects')

    prev_object_id = -1
    for i in tqdm(range(len(source_df))):
        object_id, config_id = source_df.loc[i, ['object_id', 'config_id']]

        if prev_object_id != object_id:
            if prev_object_id != -1:
                dataset[prev_object_id].unload()
            dataset[object_id].load('_v1')
            prev_object_id = object_id
        
        data_entry = dataset[object_id].data[config_id]
        frictions = data_entry.config.frictions

        mass_center = data_entry.mass_center
        mass_center_shift = np.zeros(14)
        mass_center_shift[1:4] = mass_center
        mass_center_shift[8:11] = mass_center

        for m, mode in enumerate(modes):
            if mode == 'analytical_pos':
                grasps = data_entry.pos_grasps[:10] + mass_center_shift
            elif mode == 'analytical_neg':
                grasps = data_entry.neg_grasps[:10] + mass_center_shift
            elif mode == 'graspnet':
                grasps = grasp_datas[m][object_id]
                if len(grasps) == 0:
                    print(object_id, 'has no grasps')
            elif mode == 'vgn':
                grasps = grasp_datas[m][object_id]
                if len(grasps) == 0:
                    print(object_id, 'has no grasps')

            if 'ours' not in mode:            
                for j in range(min(10, len(grasps))):
                    grasp = grasps[j]
                    dfs[m].at[i * 10 + j, 'object_id'] = object_id
                    dfs[m].at[i * 10 + j, 'config_id'] = config_id
                    dfs[m].at[i * 10 + j, 'f1'] = frictions[int(grasp[0])]
                    dfs[m].at[i * 10 + j, 'p1'] = grasp[1:4]
                    dfs[m].at[i * 10 + j, 'n1'] = grasp[4:7]
                    dfs[m].at[i * 10 + j, 'f2'] = frictions[int(grasp[7])]
                    dfs[m].at[i * 10 + j, 'p2'] = grasp[8:11]
                    dfs[m].at[i * 10 + j, 'n2'] = grasp[11:14]
                    dfs[m].at[i * 10 + j, 'obj_mass'] = sum(data_entry.config.masses)
            elif 'ours' in mode:
                if (object_id, config_id) in grasp_datas[m]:
                    grasp_positions = grasp_datas[m][(object_id, config_id)]    # (n, 6)
                    for j in range(min(10, len(grasp_positions))):
                        dfs[m].at[i * 10 + j, 'object_id'] = object_id
                        dfs[m].at[i * 10 + j, 'config_id'] = config_id
                        dfs[m].at[i * 10 + j, 'p1'] = grasp_positions[j, 0:3]
                        dfs[m].at[i * 10 + j, 'p2'] = grasp_positions[j, 3:6]
                        dfs[m].at[i * 10 + j, 'obj_mass'] = sum(data_entry.config.masses)
            
            for j in range(10):
                dfs[m].at[i * 10 + j, 'top_n'] = j + 1
    
    for m, mode in enumerate(modes):
        dfs[m].to_pickle(f'./evaluation/results/{save_names[m]}.pkl')
