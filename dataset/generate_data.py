import json
import os
import time

import numpy as np
import random
import trimesh
import traceback
from multiprocessing import Pool

from dataset.Dataset import Dataset, Config, DataEntry
from dataset.utils import shift_mass_center
from grasp.force_optimization import filter_contact_points_by_force
from grasp.generate_grasp import find_contact_points_multi, vis_grasp
import dataset.GPTsummary as GPTsummary
import dataset.category as category


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
    # show_grasp_heatmap(mesh, guassian_map, contact_pairs)
    return guassian_map

def show_grasp_heatmap(mesh, grasp_map, contact_pairs=[]):
    mesh.visual.vertex_colors = trimesh.visual.color.interpolate(grasp_map, color_map='hot')
    mesh.visual.vertex_colors[:, 3] = 0.8 * 255
    scene_list = [mesh]
    for contact_point_1, contact_point_2 in contact_pairs:
        # c1 = trimesh.creation.uv_sphere(radius=0.005)
        # c2 = trimesh.creation.uv_sphere(radius=0.005)
        # c1.vertices += contact_point
        # c2.vertices += another_contact_point
        grasp_axis = trimesh.creation.cylinder(0.005, sections=6,
                                               segment=np.vstack([contact_point_1, contact_point_2]))
        grasp_axis.visual.vertex_colors = [0, 0., 1.]
        # c1.visual.vertex_colors = [1., 0, 0]
        # c2.visual.vertex_colors = [1., 0, 0]
        # scene_list += [c1, c2, grasp_axis]
        scene_list += [grasp_axis]
    trimesh.Scene(scene_list).show()


def generate_grasp_data(meshes, frictions, sample_probs, max_normal_forces, weight, num_sample=100):
    pos_grasps, neg_grasps = find_contact_points_multi(meshes, frictions, (sample_probs * num_sample).astype('int'))
    pos_grasps, force_neg_grasps = filter_contact_points_by_force(pos_grasps, frictions, max_normal_forces, weight)
    grasp_map = generate_grasp_map(pos_grasps, trimesh.util.concatenate(meshes))
    return grasp_map, pos_grasps, np.append(neg_grasps, force_neg_grasps, axis=0)


def generate_random_config(meshes, available_materials):
    config_id = time.time()
    num_parts = len(meshes)
    material_ids = random.sample(range(0, len(available_materials)), num_parts)

    materials = [available_materials[material_id]['Material'] for material_id in material_ids]
    frictions = [available_materials[material_id]['Friction'] for material_id in material_ids]
    densities = [available_materials[material_id]['Density'] for material_id in material_ids]
    grasp_likelihoods = random.choice([None, np.random.random(num_parts)])

    fragilities = [available_materials[material_id]['Fragility'] for material_id in material_ids]

    sample_probs = grasp_likelihoods / sum(grasp_likelihoods) if grasp_likelihoods is not None else np.ones(num_parts) / num_parts

    max_normal_forces = np.power(10, np.array(fragilities))
    masses = shift_mass_center(meshes, densities)

    return Config(config_id, materials, frictions, densities, grasp_likelihoods, fragilities,
                 sample_probs, max_normal_forces, masses)
                
def test(args):
    print(len(args))
    time.sleep(1)

def generate_data(objects):
    dataset = Dataset('./data/objects')
    available_materials = json.load(open('./dataset/materials.json', 'r'))
    num_sample = 200
    for object_id in objects:
        meshes = dataset[object_id].load_meshes()
        name = dataset[object_id].name
        num_config = category.SAMPLES[name]
        for i in range(num_config):
            try:
                config = generate_random_config(meshes, available_materials)
                grasp_map, pos_grasps, neg_grasps = generate_grasp_data(meshes, config.frictions, config.sample_probs,
                                                                    config.max_normal_forces, sum(config.masses) * 9.81,
                                                                    num_sample)
                language = GPTsummary.summary(config, dataset[object_id], available_materials)
                dataset[object_id][config.id] = DataEntry(config, pos_grasps, neg_grasps, grasp_map, language)
            except Exception as e:
                # traceback.print_exc()
                # print(f'Object {object_id} config {i} error: {e}')
                with open('./dataset/error.txt', 'a') as f:
                    f.write(f'Object {object_id} config {i} error: {e}\n')
                continue
        dataset[object_id].save()

if __name__ == '__main__':
    dataset0 = Dataset('./data/objects')
    objs = dataset0.get_object_ids()
    NUM_PROCESS = 16
    tasks = []
    random.shuffle(objs)
    for i in range(NUM_PROCESS):
        tasks.append(objs[i::NUM_PROCESS])
    
    pool = Pool(NUM_PROCESS)
    pool.map(generate_data, tasks)
    pool.close()
        
    # dataset.load()