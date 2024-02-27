import json
import os
import time

import numpy as np
import random
import trimesh
import traceback
from multiprocessing import Pool
import multiprocessing

from dataset.Dataset import Dataset, Config, DataEntry
from dataset.utils import shift_mass_center
from grasp.force_optimization import filter_contact_points_by_force
from grasp.generate_grasp import find_contact_points_multi, vis_grasp
import dataset.GPTsummary as GPTsummary
import dataset.category as category
from tqdm import tqdm
import pickle

# start_time = time.time()
MIN_POS_GRASP = 10
updated_objects = pickle.load(open('./data/updated_objects.pkl', 'rb'))

def min_distances(query_points, reference_points):
    squared_diff = np.sum((query_points[:, np.newaxis] - reference_points) ** 2, axis=-1)
    return np.sqrt(np.min(squared_diff, axis=1))


def gaussian(x, mean, variance):
    sigma = np.sqrt(variance)
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(- (x - mean) ** 2 / (2 * variance))


def generate_grasp_map(pos_grasps, mesh):
    contact_pairs = pos_grasps[:, np.r_[1:4, 8:11]].reshape(-1, 2, 3)
    contact_locs = contact_pairs.reshape(-1, 3)
    dists = np.linalg.norm(mesh.vertices[:, np.newaxis] - contact_locs, axis=-1, ord=2)
    guassian_dists = gaussian(dists, 0, 0.01)
    guassian_map = np.sum(guassian_dists, axis=1)
    guassian_map = guassian_map / np.max(guassian_map)
    # global start_time
    # print(f'generate grasp map time: {time.time() - start_time} s')
    # is_show = input('show grasp map? (y/n)')
    # if is_show == 'y':
    #     show_grasp_heatmap(mesh, guassian_map, contact_pairs)
    return guassian_map

def show_grasp_heatmap(mesh, grasp_map, contact_pairs=[]):
    mesh.visual.vertex_colors = trimesh.visual.color.interpolate(grasp_map, color_map='hot')
    mesh.visual.vertex_colors[:, 3] = 0.8 * 255
    scene_list = [mesh]
    for contact_point_1, contact_point_2 in contact_pairs:
        grasp_axis = trimesh.creation.cylinder(0.005, sections=6,
                                               segment=np.vstack([contact_point_1, contact_point_2]))
        grasp_axis.visual.vertex_colors = [0, 0., 1.]
        scene_list += [grasp_axis]
    trimesh.Scene(scene_list).show()


def generate_grasp_data(meshes, frictions, sample_probs, max_normal_forces, weight, num_sample=100):
    pos_grasps, neg_grasps = find_contact_points_multi(meshes, frictions, (sample_probs * num_sample).astype('int'))
    pos_grasps, force_neg_grasps = filter_contact_points_by_force(pos_grasps, frictions, max_normal_forces, weight)
    grasp_map = generate_grasp_map(pos_grasps, trimesh.util.concatenate(meshes))
    return grasp_map, pos_grasps, np.append(neg_grasps, force_neg_grasps, axis=0)


                
def test(object_id):
    objects = [object_id]
    label_grasp(objects, config_id_t=1704583341.9222496)
    exit()
    a = pickle.load(open('./data/objects/4573/4573_1704605092.5541441_v1.pkl', 'rb'))
    print(a.language)
    print(a.pos_grasps.shape)
    print(a.neg_grasps.shape)
    print(a.grasp_map.shape)
    print(a.config.id)
    exit()

    

def label_grasp(objects, config_id_t=None):
    dataset = Dataset('./data/objects')
    available_materials = json.load(open('./dataset/materials.json', 'r'))
    n_total_entries = 0
    n_valid_entries = 0
    num_sample = 1000
    current_objects = []
    for i, object_id in enumerate(objects):
        meshes = dataset[object_id].load_meshes()
        name = dataset[object_id].name
        dataset[object_id].load()
        entries = dataset[object_id].data
        n_total_entries += len(entries)
        invalid_entries = []
        for entry in entries.values():
            # print(f'Object {object_id} config {entry.config.id} language: {entry.language}')
            try:
                config = entry.config
                # if config_id_t is not None and config.id != config_id_t:
                #     continue
                # else:
                #     print(f'Object {object_id} config {entry.config.id} start')
                shift_mass_center(meshes, config.densities)
                grasp_map, pos_grasps, neg_grasps = generate_grasp_data(meshes, config.frictions, config.sample_probs,
                                                                    config.max_normal_forces, sum(config.masses) * 9.81,
                                                                    num_sample)
                if pos_grasps.shape[0] < MIN_POS_GRASP:
                    invalid_entries.append(str(entry.config.id))
                    continue
                entry.pos_grasps = pos_grasps
                entry.neg_grasps = neg_grasps
                entry.language = GPTsummary.summary(config, dataset[object_id], available_materials)
            except Exception as e:
                traceback.print_exc()
                print(f'Object {object_id} config {entry.config.id} error: {e}')
                with open('./dataset/error.txt', 'a') as f:
                    f.write(f'Object {object_id} config {entry.config.id} error: {e}\n')
                invalid_entries.append(str(entry.config.id))
                continue

        # print("finish object {}".format(object_id))
        for config_id in invalid_entries:
            del entries[config_id]
        n_valid_entries += len(dataset[object_id].data)

        # if config_id_t is not None:
        dataset[object_id].save(version='_v1')
        dataset[object_id].unload()
        current = multiprocessing.current_process()
        print("{}/{} entries are valid {}/{} objects done by process {}".format(n_valid_entries, n_total_entries, i + 1, len(objects), current.name))
        current_objects.append(object_id)
        if i % 10 == 0 or i == len(objects) - 1:
            updated_objects = pickle.load(open('./data/updated_objects.pkl', 'rb'))
            updated_objects += current_objects
            updated_objects = list(set(updated_objects))
            pickle.dump(updated_objects, open('./data/updated_objects.pkl', 'wb'))

if __name__ == '__main__':
    # test('8001')
    dataset0 = Dataset('./data/objects')
    objs = dataset0.get_object_ids()
    objs = [obj for obj in objs if obj not in updated_objects]
    NUM_PROCESS = 24

    tasks = []
    random.shuffle(objs)
    for i in range(NUM_PROCESS):
        tasks.append(objs[i::NUM_PROCESS])
    
    pool = Pool(NUM_PROCESS)
    pool.map(label_grasp, tasks)
    pool.close()
        
    # dataset.load()