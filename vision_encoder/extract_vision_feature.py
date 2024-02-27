import logging
import os
import sys

import gdown
import numpy as np
import torch
import trimesh
from tqdm import tqdm

sys.path.append('..')
from dataset.Dataset import Dataset
from openpoints.models import build_model_from_cfg
from openpoints.utils import load_checkpoint, EasyConfig


def gaussian(x, mean, variance):
    sigma = np.sqrt(variance)
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(- (x - mean) ** 2 / (2 * variance))


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # cfg = EasyConfig()
    # cfg_path = './shapenetpart_pointnext-s.yaml'
    # pretrain_path = './shapenetpart-train-pointnext-s-ngpus4-seed5011-20220821-170334-J6Ez964eYwHHPZP4xNGcT9_ckpt_best.pth'
    # if not os.path.exists(pretrain_path):
    #     url = 'https://drive.usercontent.google.com/download?id=1tJR8hrPUPfQ5jEzDXELfJGsOpFGh-UH6&export=download&confirm=t&uuid=5ebda4bb-0381-49fe-a481-3ed05eafa199'
    #     gdown.download(url, pretrain_path, quiet=False)
    # cfg.load(cfg_path, recursive=True)
    # seg_model = build_model_from_cfg(cfg.model).cuda()
    # load_checkpoint(seg_model, pretrained_path=pretrain_path)
    # seg_model.eval()

    cfg = EasyConfig()
    cfg_path = './shapenetpart_pointnext-s_c64.yaml'
    pretrain_path = './shapenetpart-train-pointnext-s_c64-ngpus4-seed7798-20220822-024210-ZcJ8JwCgc7yysEBWzkyAaE_ckpt_best.pth'
    if not os.path.exists(pretrain_path):
        url = 'https://drive.usercontent.google.com/download?id=1OCi0XmOCBqGgGKmTf_7FrRHjS55fsFQ8&export=download&authuser=1&confirm=t&uuid=3c4f7b0f-7cc1-4769-b7cf-762eee42bfa1'
        gdown.download(url, pretrain_path, quiet=False)
    cfg.load(cfg_path, recursive=True)
    seg_model = build_model_from_cfg(cfg.model).cuda()
    load_checkpoint(seg_model, pretrained_path=pretrain_path)
    seg_model.eval()

    # cfg = EasyConfig()
    # cfg_path = './shapenetpart_pointnext-s_c160.yaml'
    # pretrain_path = './shapenetpart-train-pointnext-s_c160-ngpus4-seed6692-20220821-200729-kZfzCKh7kJ7n3chYLBSXrr_ckpt_best.pth'
    # if not os.path.exists(pretrain_path):
    #     url = 'https://drive.usercontent.google.com/download?id=1VLZQ1iuOnjjrWvvGb76H5L1rdAfDKYm3&export=download&confirm=t&uuid=2917d493-7075-4b8c-895e-a26c60598792'
    #     gdown.download(url, pretrain_path, quiet=False)
    # cfg.load(cfg_path, recursive=True)
    # seg_model = build_model_from_cfg(cfg.model).cuda()
    # load_checkpoint(seg_model, pretrained_path=pretrain_path)
    # seg_model.eval()d

    cfg = EasyConfig()
    cfg_path = './modelnet40_pointnext-s.yaml'
    pretrain_path = './modelnet40ply2048-train-pointnext-s-ngpus1-seed6848-model.encoder_args.width=64-20220525-145053-7tGhBV9xR9yQEBtN4GPcSc_ckpt_best.pth'
    if not os.path.exists(pretrain_path):
        url = 'https://drive.usercontent.google.com/download?id=1TUOR3j3xqOoLNzEneiXccR5eRmeZxKq9&export=download&authuser=0&confirm=t&uuid=9f0ee3bd-8976-4396-bd67-eced0425ffc3'
        gdown.download(url, pretrain_path, quiet=False)
    cfg.load(cfg_path, recursive=True)
    clf_model = build_model_from_cfg(cfg.model).cuda()
    load_checkpoint(clf_model, pretrained_path=pretrain_path)
    clf_model.eval()

    dataset = Dataset('../data/objects')

    if not os.path.exists('../data/updated_object_ids_v1.txt'):
        updated_ids = []
    else:
        with open('../data/updated_object_ids_v1.txt', 'r') as f:
            updated_ids = f.read().split('\n')

    with torch.no_grad():
        for object_id in tqdm(dataset.get_object_ids()):
            if object_id in updated_ids:
                print(f'{object_id} already updated')
                continue

            dataset[object_id].load('_v1')
            meshes = dataset[object_id].load_meshes()
            mesh = trimesh.util.concatenate(meshes)

            pos = []
            x = []

            config_ids = dataset[object_id].data.keys()

            if len(config_ids) == 0:
                print(f'{object_id} has no config')
                continue

            for config_id in config_ids:
                config = dataset[object_id][config_id]
                densities = config.config.densities
                masses = config.config.masses

                # find mass center
                weighted_mass_center = np.zeros(3)
                for m, density, mass in zip(meshes, densities, masses):
                    center_mass = m.centroid if m.volume < 1e-6 else m.center_mass
                    weighted_mass_center += center_mass * mass
                mass_center = weighted_mass_center / sum(masses)
                config.mass_center = mass_center

                # sample points
                points, idx = trimesh.sample.sample_surface(mesh, 2048)
                normals = mesh.face_normals[idx]

                # sample index for positive and negative grasps
                pos_grasps = config.pos_grasps
                neg_grasps = config.neg_grasps
                len_pos = min(len(pos_grasps), 200)
                len_neg = min(len(neg_grasps), 200)
                random_index = np.random.choice(2048, (len_pos + len_neg, 2), replace=False)
                config.pos_index = random_index[:len_pos]
                config.neg_index = random_index[len_pos:]

                '''for debugging visualize
                pos_contact_pairs = pos_grasps[:, np.r_[1:4, 8:11]].reshape(-1, 2, 3)
                pos_contact_locs = pos_contact_pairs.reshape(-1, 3)
                pos_contact_locs += mass_center
                neg_contact_pairs = neg_grasps[:, np.r_[1:4, 8:11]].reshape(-1, 2, 3)
                neg_contact_locs = neg_contact_pairs.reshape(-1, 3)
                neg_contact_locs += mass_center

                scene_list = [mesh]
                pcd = trimesh.PointCloud(points)
                pcd.visual.vertex_colors = [0, 0., 1.]
                scene_list += [pcd]

                pcd = trimesh.PointCloud(pos_contact_locs)
                pcd.visual.vertex_colors = [1., 0., 0]
                scene_list += [pcd]

                pcd = trimesh.PointCloud(neg_contact_locs)
                pcd.visual.vertex_colors = [0, 1., 0]
                scene_list += [pcd]

                trimesh.Scene(scene_list).show()
                '''
                try:
                    # replace points and normals with known grasps
                    points[config.pos_index[:, 0]] = pos_grasps[:len_pos, 1:4] + mass_center
                    points[config.pos_index[:, 1]] = pos_grasps[:len_pos, 8:11] + mass_center
                    points[config.neg_index[:, 0]] = neg_grasps[:len_neg, 1:4] + mass_center
                    points[config.neg_index[:, 1]] = neg_grasps[:len_neg, 8:11] + mass_center

                    normals[config.pos_index[:, 0]] = pos_grasps[:len_pos, 4:7]
                    normals[config.pos_index[:, 1]] = pos_grasps[:len_pos, 11:14]
                    normals[config.neg_index[:, 0]] = neg_grasps[:len_neg, 4:7]
                    normals[config.neg_index[:, 1]] = neg_grasps[:len_neg, 11:14]
                except:
                    print(f'{object_id} {config_id} has no grasp')
                    continue

                heights = points[:, 2] - points[:, 2].min()

                pos.append(points)
                config.point_cloud = points
                x.append(np.concatenate([points, normals, heights[:, np.newaxis]], axis=1).T)

                # gererate grasp map
                pos_contact_pairs = pos_grasps[:, np.r_[1:4, 8:11]].reshape(-1, 2, 3)
                pos_contact_locs = pos_contact_pairs.reshape(-1, 3)
                pos_contact_locs += mass_center
                dists = np.linalg.norm(points[:, np.newaxis] - pos_contact_locs, axis=-1, ord=2)
                guassian_dists = gaussian(dists, 0, 0.01)
                guassian_map = np.sum(guassian_dists, axis=1)
                config.grasp_map = guassian_map / np.max(guassian_map)

                '''for debugging visualize
                pcd = trimesh.PointCloud(points)
                pcd.visual.vertex_colors = trimesh.visual.color.interpolate(config.grasp_map, color_map='hot')
                pcd.visual.vertex_colors[:, 3] = 0.8 * 255
                scene_list = [pcd]
                for contact_point_1, contact_point_2 in pos_contact_pairs:
                    contact_point_1 += mass_center
                    contact_point_2 += mass_center
                    grasp_axis = trimesh.creation.cylinder(0.005, sections=6,
                                                           segment=np.vstack([contact_point_1, contact_point_2]))
                    grasp_axis.visual.vertex_colors = [0, 0., 1.]
                    scene_list += [grasp_axis]
                trimesh.Scene(scene_list).show()
                '''

            pos = torch.Tensor(np.array(pos)).cuda().contiguous()
            x = torch.Tensor(np.array(x)).cuda().contiguous()

            for batch in range(0, len(pos), 64):
                start = batch
                end = min(batch + 64, len(pos))
                inp = {'pos': pos[start:end],
                       'x': x[start:end],
                       'cls': torch.zeros(1, 16).long().cuda(),
                       }

                local_features = seg_model(inp).cpu().numpy()
                global_features = clf_model(inp['pos']).cpu().numpy()

                for i, config_id in enumerate(list(config_ids)[start:end]):
                    config = dataset[object_id][config_id]
                    config.local_feature = local_features[i]
                    config.global_feature = global_features[i]

                    '''for debug
                    for k, v in config.__dict__.items():
                        print(k, v.shape if isinstance(v, np.ndarray) else v)
                    '''


            dataset[object_id].save('_v1')
            dataset[object_id].unload()

            with open('../data/updated_object_ids_v1.txt', 'a') as f:
                f.write(f'{object_id}\n')
