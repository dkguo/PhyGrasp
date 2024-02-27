import random
import time

import os
import sys
import numpy as np
import pybullet as p
import pybullet_planning as pp
import trimesh
from tqdm import tqdm
import pandas as pd
import argparse

from dataset.Dataset import Dataset
from dataset.generate_data import show_grasp_heatmap
from dataset.utils import compute_part_ids



def test_grasp(grasps, mesh, obj_mass=None):
    pp.reset_simulation()
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
    vertices = mesh.vertices
    faces = mesh.faces
    indices = faces.reshape(-1)
    objId = p.createCollisionShape(p.GEOM_MESH, vertices=vertices, indices=indices)
    obj = p.createMultiBody(baseMass=obj_mass, baseCollisionShapeIndex=objId, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1])
    p.changeDynamics(obj, -1, lateralFriction=objfriction, spinningFriction=objfriction, rollingFriction=objfriction)


    # p.loadURDF(obj_urdf_path, [0, -0.5, 0], globalScaling=10)

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

        pp.step_simulation()
        if visualize:
            time.sleep(0.02)

    distanceThreshold = 0.01  # Large number to ensure all closest points are found
    closestPoints_1 = p.getClosestPoints(fingers[0], obj, distanceThreshold)
    closestPoints_2 = p.getClosestPoints(fingers[1], obj, distanceThreshold)
    success = True if closestPoints_1 and closestPoints_2 else False

    return success


def print_stats(df):
    n_success = len(df[(df['success'] == 1) & (df['top_n'] == 1)])
    n_total = len(df[(df['success'] != -1) & (df['top_n'] == 1)])
    print(f'Top_1 Success rate: {n_success / n_total} ({n_success}/{n_total})')

    if top_n > 1:
        success_mask = df[df['top_n'] == 1]['success'] == 2     # all False
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
    mode = 'ours_1706493574.6243873'   # 0.673
    # mode = 'random'

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', '-m', type=str, default=mode)
    argparser.add_argument('--top_n', type=int, default=1)
    argparser.add_argument('--n_test_config', '-n', type=int, default=1000)
    args = argparser.parse_args()

    mode = args.mode
    top_n = args.top_n
    n_test_config = args.n_test_config
    n_test = n_test_config * top_n
    useMass = True
    skip_tested = True
    source_path = f'./evaluation/results/{mode}_not_tested.pkl'
    save_path = f'./evaluation/results/{mode}.pkl'
    load_path = save_path if os.path.exists(save_path) else source_path

    visualize = False
    sphere_radius = 0.05
    sphere_mass = 1000 if useMass else 1
    default_obj_mass = 1
    color = [1, 0, 0, 1]
    downsample = True
    objfriction = 0.5

    pp.connect(use_gui=visualize)
    start_time = time.time()
    dataset = Dataset('/home/gdk/Repositories/DualArmManipulation/data/objects')
    df = pd.read_pickle(load_path)

    # print(df[:100])

    prev_object_id = -1
    mesh = None
    to_test = df[df['top_n'] <= top_n][:n_test] 
    not_tested = to_test[to_test['success'] == -1]
    effective_test_index = not_tested.index if skip_tested else to_test.index
    num_tested = 0
    for i in tqdm(effective_test_index):
        object_id = df.loc[i, 'object_id']
        if object_id == 'nan':
            print(f'object {object_id} has no grasp at top', df.loc[i, 'top_n'])
            df.at[i, 'success'] = 0
            continue
        if prev_object_id != object_id:
            meshes = dataset[object_id].load_meshes()
            mesh = trimesh.util.concatenate(dataset[object_id].load_meshes())
            if downsample:
                mesh = mesh.simplify_quadric_decimation(5000)
            prev_object_id = object_id

        f1, p1, n1, f2, p2, n2 = df.loc[i, ['f1', 'p1', 'n1', 'f2', 'p2', 'n2']]
        obj_mass = df.loc[i, 'obj_mass'] if useMass and 'obj_mass' in df.columns else None

        if 'ours' in mode and np.isnan(f1):
            dataset[object_id].load('_v1')
            data_entry = dataset[object_id].data[df.loc[i, 'config_id']]
            frictions = data_entry.config.frictions
            obj_mass = sum(data_entry.config.masses)
            part_ids = compute_part_ids(np.array([p1, p2]), meshes)
            norm = (p1- p2) / np.linalg.norm(p1 - p2)
            df.at[i, 'f1'] = f1 = frictions[part_ids[0]] 
            df.at[i, 'f2'] = f2 = frictions[part_ids[1]]
            df.at[i, 'n1'] = n1 = norm
            df.at[i, 'n2'] = n2 = -norm
            dataset[object_id].unload()
        
        if mode == 'random':
            samples, _ = trimesh.sample.sample_surface(mesh, 2)
            df.at[i, 'p1'] = p1 = samples[0] * 1.1
            df.at[i, 'p2'] = p2 = samples[1] * 1.1
            df.at[i, 'n1'] = n1 = (p1 - p2) / np.linalg.norm(p1 - p2)
            df.at[i, 'n2'] = n2 = -n1
            part_ids = compute_part_ids(np.array([p1, p2]), meshes)
            dataset[object_id].load('_v1')
            data_entry = dataset[object_id].data[df.loc[i, 'config_id']]
            frictions = data_entry.config.frictions
            df.at[i, 'f1'] = f1 = frictions[part_ids[0]]
            df.at[i, 'f2'] = f2 = frictions[part_ids[1]]
            dataset[object_id].unload()

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
            success = test_grasp(grasps, mesh, obj_mass)
            df.at[i, 'success'] = 1 if success else 0
        except Exception as e:
            print(i, object_id)
            print(e)
            df.at[i, 'success'] = -2

        if visualize:
            print(i, object_id, df.loc[i, 'config_id'], df.loc[i, 'top_n'], df.at[i, 'success'])

        num_tested += 1

        if num_tested % 1000 == 0:
            tt = time.time()
            df.to_pickle(save_path)
            print(f'Saved in {time.time() - tt} seconds')
            print_stats(df)
        
    
    tt = time.time()
    df.to_pickle(save_path)
    print(f'Saved in {time.time() - tt} seconds')

    tt = time.time()
    print_stats(df)
