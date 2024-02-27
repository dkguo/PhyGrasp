from dataset.LVDataset import LVDataset
from model.data_utils import get_dataloader
import pickle
import argparse

def load_entry(entry_path):
    with open(entry_path, 'rb') as f:
        entry = pickle.load(f)
        language = entry.language
        language_feature = entry.language_feature
        point_cloud = entry.point_cloud
        vision_global = entry.global_feature
        vision_local = entry.local_feature.T
        grasp_map = entry.grasp_map
        pos_index = entry.pos_index
        neg_index = entry.neg_index
        lanugalge_feature_15 = entry.language_feature_15
        # print(point_cloud.shape, vision_global.shape, vision_local.shape, grasp_map.shape, pos_index.shape, neg_index.shape)
        # print(type(pos_index), pos_index.shape, type(neg_index), neg_index.shape, type(vision_local), vision_local.shape)
        # print(pos_index)
        # print(lanugalge_feature_15.shape)
        print(language)
            
        print(entry_path)
        return language
    

    
# language = load_entry("./data/objects/40855/40855_1704580390.4439733_v1.pkl")
if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--dataset_dir', type=str, default='./data/objects/')
    argparse.add_argument('--shuffle', type=bool, default=False)
    argparse.add_argument('--batch_size', type=int, default=128)
    argparse.add_argument('--num_workers', type=int, default=16)
    args = argparse.parse_args()
    params = vars(args)
    _, _, test_loader = get_dataloader(params)
    for i, data in enumerate(test_loader):
        language, point_cloud, vision_global, vision_local, grasp_map, pos_index, neg_index, pos_neg_num, entry_paths = data
        for j, entry_path in enumerate(entry_paths):
            language = load_entry(entry_path)
            if 'display' in language:
                print(i, j)
                exit()

    

