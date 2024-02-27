import pickle
import os
import numpy as np
import json
import csv

# import LVDataset


if __name__ == '__main__':
    paths = []
    for i in range(303):
        if os.path.exists('./checkpoints/maps_{}.pkl'.format(i)) == False:
            continue
        test_dataset = pickle.load(open('./checkpoints/maps_{}.pkl'.format(i), 'rb'))
        paths += test_dataset['entry_path']
    
    # object_ids = [path.split('/')[-2] for path in paths]
    # object_ids = np.array(object_ids)
    # object_ids = np.unique(object_ids)
    # print(object_ids)
    # print(len(object_ids))
    # # save json
    # json.dump(object_ids.tolist(), open('./data/dataset/test_object_ids.json', 'w'))

    # config_ids = [path.split('/')[-1].split('_')[1] for path in paths]
    # print(config_ids[0])
    # print(len(config_ids))
    # config_ids = np.unique(config_ids)
    # print(len(config_ids))
        
    object_config_ids = [[path.split('/')[-2], path.split('/')[-1].split('_')[1]] for path in paths]

    i = 0
    filtered_object_config_ids = []
    for object_id, config_id in object_config_ids:
        if 'pkl' in config_id:
            i += 1
            continue
        filtered_object_config_ids.append([object_id, config_id])
    print('not _v1: ', i)
    
    # with open('./data/dataset/test_object_config_ids.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['object_id', 'config_id'])
    #     writer.writerows(filtered_object_config_ids)
    print(len(filtered_object_config_ids))
    

    # dataset_set = pickle.load(open('./data/dataset/test_dataset_v1.pkl', 'rb'))
    # object_ids = [entry[-1].split('/')[-2] for entry in dataset_set]
    # object_ids = np.array(object_ids)
    # object_ids = np.unique(object_ids)
    # print(object_ids)
    # print(len(object_ids))

