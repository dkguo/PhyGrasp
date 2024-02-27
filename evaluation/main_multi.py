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


def test_grasp(grasps, vertices, indices, obj_mass=None):
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)

    obj_mass = obj_mass if obj_mass else default_obj_mass

    if visualize:
        p.addUserDebugLine([0, 0, 0], [1, 0, 0], [1, 0, 0])
        p.addUserDebugLine([0, 0, 0], [0, 1, 0], [0, 1, 0])
        p.addUserDebugLine([0, 0, 0], [0, 0, 1], [0, 0, 1])

    fingers = []
    for g in grasps:
        pos = g['pos'] - g['normal'] * sphere_radius
        f = g['friction']
        sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius)
        sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=color)
        finger = p.createMultiBody(sphere_mass, sphere_collision, sphere_visual, basePosition=pos)
        p.changeDynamics(finger, -1, lateralFriction=f, spinningFriction=f, rollingFriction=f)
        fingers.append(finger)

    # Load object
    # mesh = trimesh.load(obj_urdf_path)
    # vertices = mesh.vertices
    # faces = mesh.faces
    # indices = faces.reshape(-1)

    objId = p.createCollisionShape(p.GEOM_MESH, vertices=vertices, indices=indices)
    obj = p.createMultiBody(baseMass=obj_mass, baseCollisionShapeIndex=objId, basePosition=[0, 0, 0],
                            baseOrientation=[0, 0, 0, 1])
    p.changeDynamics(obj, -1, lateralFriction=objfriction, spinningFriction=objfriction, rollingFriction=objfriction)

    # force_magnitude = obj_mass * 9.81 * 10
    force_magnitude = sphere_mass * 9.81 * 10

    for i in range(100):
        for finger, grasp in zip(fingers, grasps):
            # Gravity compensation force
            force = -sphere_mass * np.array([0, 0, -9.81])
            # Apply force in contact point in the direction of the contact normal
            finger_pos, finger_quat = p.getBasePositionAndOrientation(finger)
            p.resetBasePositionAndOrientation(finger, finger_pos, [0, 0, 0, 1])

            force += np.array(force_magnitude * grasp['normal'] / np.linalg.norm(grasp['normal']))
            force += np.array(sphere_mass * 9.81 * (grasp['pos'] - finger_pos))
            p.applyExternalForce(finger, -1, force, grasp['pos'], p.WORLD_FRAME)
            # p.applyExternalTorque(finger, -1, [0, 0, 0])

        p.stepSimulation()
        if visualize:
            time.sleep(0.02)

    distanceThreshold = 0.01  # Large number to ensure all closest points are found
    closestPoints_1 = p.getClosestPoints(fingers[0], obj, distanceThreshold)
    closestPoints_2 = p.getClosestPoints(fingers[1], obj, distanceThreshold)
    success = True if closestPoints_1 and closestPoints_2 else False

    return success


def print_stats(df):
    df = df[:n_test_config * 10]
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


if __name__ == '__main__':
    # mode = 'graspnet'   # 0.402
    # mode = 'analytical_pos'   # 0.873
    # mode = 'analytical_neg'   # 0.518
    # mode = 'ours_1706493574.6243873'   # 0.673
    # mode = 'random'

    # don't put graspnet in the first place
    # modes = ['analytical_neg', 'ours_1706493574.6243873', 'ours_1706351155.818242', 'graspnet']
    # modes = ['analytical_pos',
    #          'ours_1706605305.8925593', 'ours_1706613034.3918543',
    #          'ours_1706620680.2738318', 'ours_1706628441.2965753']
    # modes = ['vgn']
    modes = ['ours_1706605305.8925593wol']

    argparser = argparse.ArgumentParser()
    # argparser.add_argument('--mode', '-m', type=str, default=mode)
    argparser.add_argument('--top_n', type=int, default=1)
    argparser.add_argument('--n_test_config', '-n', type=int, default=1000)
    args = argparser.parse_args()

    # mode = args.mode
    top_n = args.top_n
    n_test_config = args.n_test_config
    n_test = n_test_config * top_n
    useMass = True
    skip_tested = True

    visualize = False
    sphere_radius = 0.05
    sphere_mass = 1000 if useMass else 1
    default_obj_mass = 1
    color = [1, 0, 0, 1]
    downsample = True
    objfriction = 0.5

    if visualize:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)
    start_time = time.time()
    dataset = Dataset('/home/gdk/Repositories/DualArmManipulation/data/objects')
    dfs = []
    save_paths = []
    for mode in modes:
        source_path = f'/home/gdk/Repositories/DualArmManipulation/evaluation/results/{mode}_not_tested.pkl'
        save_path = f'/home/gdk/Repositories/DualArmManipulation/evaluation/results/{mode}_v2.pkl'
        save_paths.append(save_path)
        load_path = save_path if os.path.exists(save_path) else source_path
        print(f'Loading {load_path}')
        df = pd.read_pickle(load_path)
        # print(len(df))
        if 'v2' not in load_path:
            v2_index = pickle.load(
                open('/home/gdk/Repositories/DualArmManipulation/data/dataset/test_indices.pkl', 'rb'))
            v2_index = np.concatenate(np.array([[i + j for j in range(10)] for i in v2_index]))
            # print(len(v2_index), min(v2_index), max(v2_index))
            df = df.loc[v2_index]
            # print(len(df))
        dfs.append(df)
    # exit()

    # print(df[:100])

    prev_object_id = -1
    mesh = None

    index = []
    for df in dfs:
        to_test = df[df['top_n'] <= top_n][:n_test]
        not_tested = to_test[to_test['success'] == -1]
        effective_test_index = not_tested.index if skip_tested else to_test.index
        index.append(effective_test_index)
    index = np.concatenate(index)
    index = np.unique(index)

    num_tested = 0
    data_entry_loaded = False
    to_save = False
    for i in tqdm(index):
        object_id = dfs[0].loc[i, 'object_id']

        if object_id == 'nan':
            print(f'{i} has no grasp at top', dfs[0].loc[i, 'top_n'])
            continue

        if prev_object_id != object_id:
            meshes = dataset[object_id].load_meshes()
            mesh = trimesh.util.concatenate(meshes)
            if downsample:
                mesh = mesh.simplify_quadric_decimation(5000)
            if data_entry_loaded:
                dataset[prev_object_id].unload()
                data_entry_loaded = False
            prev_object_id = object_id

        for m, mode in enumerate(modes):
            if dfs[m].loc[i, 'success'] != -1 and skip_tested:
                continue
            f1, p1, n1, f2, p2, n2 = dfs[m].loc[i, ['f1', 'p1', 'n1', 'f2', 'p2', 'n2']]
            obj_mass = dfs[m].loc[i, 'obj_mass'] if useMass and 'obj_mass' in dfs[m].columns else None

            to_save = True

            if 'ours' in mode and np.isnan(f1):
                if not data_entry_loaded:
                    dataset[object_id].load('_v1')
                    data_entry_loaded = True
                data_entry = dataset[object_id].data[dfs[m].loc[i, 'config_id']]
                frictions = data_entry.config.frictions
                obj_mass = sum(data_entry.config.masses)
                part_ids = compute_part_ids(np.array([p1, p2]), meshes)
                norm = (p1 - p2) / np.linalg.norm(p1 - p2)
                dfs[m].at[i, 'f1'] = f1 = frictions[part_ids[0]]
                dfs[m].at[i, 'f2'] = f2 = frictions[part_ids[1]]
                dfs[m].at[i, 'n1'] = n1 = -norm
                dfs[m].at[i, 'n2'] = n2 = norm

            if np.isnan(f1):
                dfs[m].at[i, 'success'] = 0
                continue

            grasps = [
                {
                    'pos': p1,
                    'normal': n1,
                    'friction': f1 ** 2 / objfriction,
                },
                {
                    'pos': p2,
                    'normal': n2,
                    'friction': f2 ** 2 / objfriction,
                }
            ]

            try:
                vertices = mesh.vertices
                faces = mesh.faces
                indices = faces.reshape(-1)
                if i == 345110:
                    dfs[m].at[i, 'success'] = -2
                else:
                    success = test_grasp(grasps, vertices, indices, obj_mass)
                    dfs[m].at[i, 'success'] = 1 if success else 0
            except Exception as e:
                print(i, object_id, 'pybullet error:', e)
                dfs[m].at[i, 'success'] = -2

            if visualize:
                print(mode, i, object_id, dfs[m].loc[i, 'config_id'], dfs[m].loc[i, 'top_n'], dfs[m].at[i, 'success'])

        if num_tested > 0 and num_tested % 2000 == 0 and to_save:
            for m, mode in enumerate(modes):
                tt = time.time()
                dfs[m].to_pickle(save_paths[m])
                print(f'Saved {mode} in {time.time() - tt} seconds')
            to_save = False

        if num_tested > 0 and num_tested % 100 == 0:
            for m, mode in enumerate(modes):
                print(mode)
                print_stats(dfs[m])

        num_tested += 1

    for m, mode in enumerate(modes):
        print(mode)
        if to_save:
            tt = time.time()
            dfs[m].to_pickle(save_paths[m])
            print(f'Saved in {time.time() - tt} seconds')
        print_stats(dfs[m])
