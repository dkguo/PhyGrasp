import pickle
import torch
from torch.utils.data import DataLoader
from dataset.LVDataset import LVDataset
import numpy as np
import random

def get_dataloader(params):
    batch_size = params['batch_size']
    shuffle = params['shuffle']
    num_workers = params['num_workers']
    with open('./data/dataset/train_dataset_v2.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open('./data/dataset/val_dataset_v2.pkl', 'rb') as f:
        val_dataset = pickle.load(f)
    with open('./data/dataset/test_dataset_v2.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader, val_loader, test_loader

def get_dataloader_special(params):
    batch_size = params['batch_size']
    shuffle = params['shuffle']
    num_workers = params['num_workers']
    dataset_path = f'./checkpoints/test_dataset_{params["model_id"]}.pkl'
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    dataset.version = ""
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return test_loader

def ramdom_sample_pos(pos):
    '''
    pos: tensor of shape (batch_size, kp1, 6)
    select best 1 in kp1 for each batch, then select the second with max distance with the first one, select the third with max distance with the first two, and so on until 5
    '''

    batch_size, kp1, _ = pos.shape
    pos_sample = torch.zeros((batch_size, 5, 6)).to(pos.device)
    for i in range(batch_size):
        pos_sample[i, 0] = pos[i, 0]
        indexes = [0]
        for j in range(1, 5):
            pos_i = pos[i].unsqueeze(1).repeat(1, j, 1) # (kp1, j, 6)
            pos_sample_i = pos_sample[i, :j].unsqueeze(0) # (1, j, 6)
            dis = torch.norm(pos_i - pos_sample_i, dim=(1,2)) # (kp1)
            ids = torch.topk(dis, k=5).indices
            for id in ids:
                if id not in indexes:
                    break
            pos_sample[i, j] = pos[i, id]
            indexes.append(id)
            # check if the index is repeated
        assert len(indexes) == len(set(indexes)), "index is repeated"
    return pos_sample

def split_data(params):
    dataset_dir = params['dataset_dir']
    # assert False, "split_data is deprecated"

    dataset = LVDataset()
    dataset.load(dataset_dir, version="_v1")
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.70)
    val_size = int(dataset_size * 0.10)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    with open('./data/dataset/train_dataset_v1.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open('./data/dataset/val_dataset_v1.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)
    with open('./data/dataset/test_dataset_v1.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)

def place_data():
    with open('./data/dataset/train_dataset_v1.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
        train_entry_paths = [entry for entry in train_dataset]
        pickle.dump(train_entry_paths, open('./data/dataset/train_entry_paths_v1.pkl', 'wb'))
        print(len(train_entry_paths))
    with open('./data/dataset/val_dataset_v1.pkl', 'rb') as f:
        val_dataset = pickle.load(f)
        val_entry_paths = [entry for entry in val_dataset]
        pickle.dump(val_entry_paths, open('./data/dataset/val_entry_paths_v1.pkl', 'wb'))
        print(len(val_entry_paths))
    with open('./data/dataset/test_dataset_v1.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
        test_entry_paths = [entry for entry in test_dataset]
        pickle.dump(test_entry_paths, open('./data/dataset/test_entry_paths_v1.pkl', 'wb'))
        print(len(test_entry_paths))
        
def move_data():
    with open('./data/dataset/train_entry_paths_v1.pkl', 'rb') as f:
        train_entry_paths = pickle.load(f)
    with open('./data/dataset/val_entry_paths_v1.pkl', 'rb') as f:
        val_entry_paths = pickle.load(f)
        random.shuffle(val_entry_paths)
        val_entry_paths_move = val_entry_paths[0:9385]
        val_entry_paths = val_entry_paths[9385:]
        train_entry_paths += val_entry_paths_move
    with open('./data/dataset/test_entry_paths_v1.pkl', 'rb') as f:
        test_entry_paths = pickle.load(f)

    with open('./data/dataset/test_entry_paths_v2.pkl', 'rb') as f:
        test_entry_paths_remain = pickle.load(f)
    test_entry_paths_move = set(test_entry_paths) - set(test_entry_paths_remain)
    train_entry_paths += list(test_entry_paths_move)
        
    train_dataset = LVDataset(version="_v1")
    train_dataset.entry_paths = train_entry_paths
    val_dataset = LVDataset(version="_v1")
    val_dataset.entry_paths = val_entry_paths
    test_dataset = LVDataset(version="_v1")
    test_dataset.entry_paths = test_entry_paths_remain
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    # with open('./data/dataset/train_dataset_v2.pkl', 'wb') as f:
    #     pickle.dump(train_dataset, f)
    # with open('./data/dataset/val_dataset_v2.pkl', 'wb') as f:
    #     pickle.dump(val_dataset, f)
    # with open('./data/dataset/test_dataset_v2.pkl', 'wb') as f:
    #     pickle.dump(test_dataset, f)

if __name__ == "__main__":
    move_data()