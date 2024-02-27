import torch
import os
import pickle
import random
import numpy as np

class LVDataset(torch.utils.data.Dataset):
    def __init__(self, version="", language_layer=20) -> None:
        self.entry_paths = []
        self.version = version
        self.language_layer = language_layer
    
    def __len__(self):
        return len(self.entry_paths)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return self.get_entry(self.entry_paths[index])
    
    def get_entry(self, entry_path):
        # return entry_path
        # print(dir(self))
        with open(entry_path, 'rb') as f:
            entry = pickle.load(f)
            assert self.language_layer == 20
            if hasattr(self, 'language_layer') == False:
                assert False, "self.language_layer is not defined"
                self.language_layer = 0
            if self.language_layer == 15:
                language = entry.language_feature_15
            elif self.language_layer == 20:
                language = entry.language_feature_20
            elif self.language_layer == 25:
                language = entry.language_feature_25
            else:
                language = entry.language_feature.squeeze(0).cpu().numpy()
            point_cloud = entry.point_cloud
            vision_global = entry.global_feature
            vision_local = entry.local_feature.T
            grasp_map = entry.grasp_map
            pos_index = entry.pos_index
            neg_index = entry.neg_index
            pos_neg_num = np.array([pos_index.shape[0], neg_index.shape[0]])
            pos_index = np.pad(pos_index, ((0, 200 - pos_index.shape[0]), (0, 0)), mode='constant', constant_values=0)
            neg_index = np.pad(neg_index, ((0, 200 - neg_index.shape[0]), (0, 0)), mode='constant', constant_values=0)

            # print(type(pos_index), pos_index.shape, type(neg_index), neg_index.shape, type(vision_local), vision_local.shape)
            # print(language.shape)
        return language, point_cloud, vision_global, vision_local, grasp_map, pos_index, neg_index, pos_neg_num, entry_path
    
    def test_attributes(self):
        cnt = 0
        random.shuffle(self.entry_paths)
        
        for id, entry_path in enumerate(self.entry_paths):
            with open(entry_path, 'rb') as f:
                entry = pickle.load(f)
                if hasattr(entry, 'global_feature') == False:
                    print(dir(entry))
                    print(entry_path)
                    cnt += 1
            if id % 100 == 0:
                print("{}/{}".format(cnt, id))

    def test_data(self):
        print("test data")
        print(len(self.entry_paths))
        random.shuffle(self.entry_paths)
        # self.entry_paths = self.entry_paths[0:2]

        pos = []
        neg = []
        for entry_path in self.entry_paths:
            with open(entry_path, 'rb') as f:
                entry = pickle.load(f)
                pos_grasps = entry.pos_grasps
                neg_grasps = entry.neg_grasps
                pos.append(pos_grasps.shape[0])
                neg.append(neg_grasps.shape[0])
        pos = np.array(pos)
        neg = np.array(neg)
        print("pos mean, std", np.mean(pos, axis=0), np.std(pos, axis=0))
        print("neg mean, std", np.mean(neg, axis=0), np.std(neg, axis=0))

    def small_dataset(self):
        random.shuffle(self.entry_paths)
        self.entry_paths = self.entry_paths[0:10000]
        
    def load(self, dataset_dir, version=""):
        for object_id in os.listdir(dataset_dir):
            object_dir = os.path.join(dataset_dir, object_id)
            for entry_name in os.listdir(object_dir):
                suffix = f'{version}.pkl'
                if entry_name.endswith(suffix):
                    config_id = entry_name[:-len(suffix)].split('_')[1]
                    entry_path = os.path.join(object_dir, entry_name)
                    self.entry_paths.append(entry_path)

        # self.entry_paths = self.entry_paths[0:1000]
    
if __name__ == '__main__':
    lvdataset = LVDataset()
    lvdataset.load('./data/objects/', version="_v1")
    entry_paths = lvdataset.entry_paths
    print(len(entry_paths))
    random.shuffle(entry_paths)
    pickle.dump(entry_paths, open('./data/dataset/v1_random_1000.pkl', 'wb'))