import json
import os
import time

import numpy as np
import trimesh
import pickle
import dataset.helpers as helpers

class Dataset:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.data_entries = {}

    def __getitem__(self, object_id):
        if object_id not in self.data_entries:
            self.data_entries[object_id] = ObjectEntry(object_id, self.dataset_dir)
        return self.data_entries[object_id]

    def get_object_ids(self):
        return os.listdir(self.dataset_dir)

    def load(self):
        for object_id in self.get_object_ids():
            self[object_id].load()
            # for entry_name in os.listdir(os.path.join(self.dataset_dir, object_id)):
            #     if entry_name.endswith('.pkl'):
            #         config_id = entry_name.split('_')[1][0:-4]
            #         self[object_id].load(config_id)

    def count_category(self):
        return helpers.count_category(self)
    
    def get_languages(self):
        return helpers.get_languages(self)

class ObjectEntry:
    def __init__(self, object_id, dataset_dir):
        self.object_id = object_id
        self.dataset_dir = dataset_dir
        self.object_dir = os.path.join(self.dataset_dir, self.object_id)
        self.data = {}
        self.load_metadata()

    def __getitem__(self, config_id):
        if config_id not in self.data:
            self.data[config_id] = self.load(config_id)
        return self.data[config_id]

    def __setitem__(self, config_id, data_entry):
        self.data[config_id] = data_entry

    def load_metadata(self):
        parts_json = json.load(open(os.path.join(self.object_dir, 'parts.json'), 'r'))
        self.name = parts_json['name']
    
    def load_meshes(self):
        parts_json = json.load(open(os.path.join(self.object_dir, 'parts.json'), 'r'))
        if 'name' not in parts_json:
            raise Exception('parts.json does not have name field')
            meshes_dir = os.path.join(self.object_dir, 'meshes')
            meshes = []
            names = []
            for part in parts_json.values():
                name = part['name']
                mesh = trimesh.load(os.path.join(meshes_dir, part['mesh_name']))
                mesh.units = 'm'
                meshes.append(mesh)
                names.append(name)
            # print(names)
            return meshes
        else:
            self.name = parts_json['name']
            meshes_dir = os.path.join(self.object_dir, 'objs')
            meshes = []
            names = []
            for part in parts_json['parts'].values():
                names.append(part['name'])
                obj_names = part['objs']
                obj_meshes = []
                for obj_name in obj_names:
                    mesh = trimesh.load(f'{meshes_dir}/{obj_name}.obj')
                    mesh.units = 'm'
                    obj_meshes.append(mesh)
                mesh = trimesh.util.concatenate(obj_meshes)
                meshes.append(mesh)
            # print(names)
            return meshes

    def save(self, version=""):
        for config_id, data_entry in self.data.items():
            assert hasattr(data_entry, 'language_feature'), 'data_entry does not have language_feature'
            entry_name = f'{self.object_id}_{config_id}{version}.pkl'
            entry_path = os.path.join(self.object_dir, entry_name)
            with open(entry_path, 'wb') as f:
                pickle.dump(data_entry, f)

    def load_config(self, config_id, version=""):
        entry_name = f'{self.object_id}_{config_id}{version}.pkl'
        entry_path = os.path.join(self.object_dir, entry_name)
        if os.path.exists(entry_path):
            with open(entry_path, 'rb') as f:
                data_entry = pickle.load(f)
                self.__setitem__(config_id, data_entry)
    
    def load(self, version=""):
        for entry_name in os.listdir(self.object_dir):
            suffix = f'{version}.pkl'
            if entry_name.endswith(suffix):
                config_id = entry_name[:-len(suffix)].split('_')[1]
                self.load_config(config_id, version=version)
    
    def unload(self):
        for config_id in list(self.data.keys()):
            del self.data[config_id]
    

class DataEntry:
    def __init__(self, config, pos_grasps, neg_grasps, grasp_map, language):
        self.config = config
        self.pos_grasps = pos_grasps
        self.neg_grasps = neg_grasps
        self.grasp_map = grasp_map
        self.language = language


class Config:
    def __init__(self, config_id, materials, frictions, densities, grasp_likelihoods, fragilities,
                 sample_probs, max_normal_forces, masses):
        self.id = config_id
        # self.meshes = meshes
        assert len(materials) == len(frictions) == len(densities) == len(fragilities)
        self.num_parts = len(materials)

        # defined config
        self.materials = materials
        self.frictions = frictions
        self.densities = densities
        self.grasp_likelihoods = grasp_likelihoods
        self.fragilities = fragilities

        # calculated config
        # self.sample_probs = self.grasp_likelihoods / sum(self.grasp_likelihoods)
        # self.max_normal_forces = np.power(10, self.fragilities)
        self.sample_probs = sample_probs
        self.max_normal_forces = max_normal_forces
        self.masses = masses


if __name__ == '__main__':
    dataset = Dataset('./data/objects/')
    meshes = dataset['10314'].load_meshes()
    print(len(meshes))   
