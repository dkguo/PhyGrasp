import os
import pickle

import trimesh
import numpy as np

from dataset.Dataset import Dataset
from evaluation.affordance import map_convert


def show_heatmap(points, grasp_map, mesh=None):
    if mesh is not None:
        dists = np.linalg.norm(mesh.vertices[:, np.newaxis] - points, axis=-1, ord=2)
        closest_points = np.argmin(dists, axis=1)
        colors = grasp_map[closest_points]
        mesh.visual.vertex_colors = trimesh.visual.color.interpolate(colors, color_map='hot')
        mesh.show()
    else:
        pcd = trimesh.PointCloud(points)
        pcd.visual.vertex_colors = trimesh.visual.color.interpolate(grasp_map, color_map='hot')
        pcd.visual.vertex_colors[:, 3] = 0.8 * 255
        scene_list = [pcd]
        trimesh.Scene(scene_list).show()


def show_score_map(pid, points, score, mesh=None):
    pcd = trimesh.PointCloud(points)
    pid2 = np.argmax(score)
    score_max = np.max(score)
    score_min = np.min(score)
    score = (score - score_min) / (score_max - score_min)
    score = np.square(score)

    ball = trimesh.creation.uv_sphere(radius=0.05)
    ball.visual.vertex_colors = [255., 0., 0., 0.8 * 255]
    ball.apply_translation(points[pid])
    ball2 = trimesh.creation.uv_sphere(radius=0.05)
    ball2.visual.vertex_colors = [255., 255., 0., 0.8 * 255]
    ball2.apply_translation(points[pid2])

    if mesh is None:
        pcd.visual.vertex_colors = trimesh.visual.color.interpolate(score, color_map='cividis')
        pcd.visual.vertex_colors[:, 3] = 1.0 * 255
        scene_list = [pcd, ball, ball2]
    else:
        dists = np.linalg.norm(mesh.vertices[:, np.newaxis] - points, axis=-1, ord=2)
        closest_points = np.argmin(dists, axis=1)
        second_closest_points = np.argsort(dists, axis=1)[:, 1]
        colors = (score[closest_points] + score[second_closest_points]) / 2
        mesh.visual.vertex_colors = trimesh.visual.color.interpolate(colors, color_map='cividis')
        mesh.visual.vertex_colors[:, 3] = 0.8 * 255
        scene_list = [mesh, ball, ball2]

    trimesh.Scene(scene_list).show()

def find_predition(entry_path):
    for map in maps:
        entry_paths = map['entry_path']
        if entry_path in entry_paths:
            id_in_map = entry_paths.index(entry_path)
            points = map['point_cloud'][id_in_map]
            grasp_map = map['prediction'][id_in_map]
            score = map['score'][id_in_map]
            return points, grasp_map, score

def find_config(object_id):
    found_entry_paths = []
    for map in maps:
        entry_paths = map['entry_path']
        for entry_path in entry_paths:
            object_id_in_entry_path = entry_path.split('/')[-2]
            if object_id == object_id_in_entry_path:
                found_entry_paths.append(entry_path)
    return found_entry_paths


if __name__ == '__main__':
    # entry_paths = ['./data/objects/4931/4931_1704412294.0766087_v1.pkl']
    # entry_paths = ['./data/objects/1168/1168_1704368664.9076674_v1.pkl']

    model_id = '1706605305.8925593'
    # model_id = '1706613034.3918543'
    maps = []
    all_entry_paths = []
    for batch_id in range(80):
        filename = './checkpoints/maps_{}_{}.pkl'.format(model_id, batch_id)
        if not os.path.exists(filename):
            continue
        maps.append(pickle.load(open(filename, 'rb')))
        all_entry_paths += maps[-1]['entry_path']

    print(len(all_entry_paths))

    dataset = Dataset('./data/objects')

    # vgn_maps = pickle.load(open('./data_vgn_map.pickle', 'rb'))

    # object_id = '4935'
    # meshes = dataset[object_id].load_meshes()
    # for mesh in meshes:
    #     mesh.visual.vertex_colors = np.random.randint(0, 255, size=4, dtype=np.uint8)
    #     mesh.visual.vertex_colors[:, 3] = 255
    # mesh = trimesh.util.concatenate(meshes)
    #
    # # find other config
    # found_entry_paths = find_config(object_id)
    # for entry_path in found_entry_paths:
    #     print(entry_path)
    #     points, grasp_map, score = find_predition(entry_path)
    #     with open(entry_path, 'rb') as f:
    #         entry = pickle.load(f)
    #         show_heatmap(points, grasp_map, mesh)
    #         show_score_map(0, points, score[0], mesh)
    #         print(entry.language)
    #
    # exit()

    found_entry_paths = find_config('10782')

    for entry_path in found_entry_paths:
    # for entry_path in all_entry_paths[60:]:
    # for entry_path in entry_paths:
        object_id = entry_path.split('/')[-2]
        print(object_id)

        meshes = dataset[object_id].load_meshes()
        for mesh in meshes:
            mesh.visual.vertex_colors = np.random.randint(0, 255, size=4, dtype=np.uint8)
            mesh.visual.vertex_colors[:, 3] = 255
        mesh = trimesh.util.concatenate(meshes)
        # mesh.show()

        points, grasp_map, score = find_predition(entry_path)
        entry = pickle.load(open(entry_path, 'rb'))
        print(entry.language)

        gt_map = entry.grasp_map
        show_heatmap(points, gt_map, mesh)

        print(gt_map - grasp_map)

        # pcd = trimesh.PointCloud(points)
        # pcd.show()

        # if object_id in vgn_maps:
        #     print('VGN map found')
        #     grasp_map = map_convert(vgn_maps[object_id], points)
        #     show_heatmap(points, grasp_map, mesh)
        #     continue


        # show_heatmap(points, grasp_map, mesh)
        for k in range(20):
            show_score_map(k, points, score[k], mesh)
        # show_score_map(10, points, score[10], mesh)
    #
    # # find other config
    # found_entry_paths = find_config(object_id)
    # for entry_path in found_entry_paths:
    #     print(entry_path)
    #     points, grasp_map, score = find_predition(entry_path)
    #     with open(entry_path, 'rb') as f:
    #         entry = pickle.load(f)
    #         show_heatmap(points, grasp_map, mesh)
    #         show_score_map(0, points, score[0], mesh)



