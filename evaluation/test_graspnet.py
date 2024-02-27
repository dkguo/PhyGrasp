""" Demo to show prediction results.
    Author: chenxi-wang
"""
import random
import os
import sys
sys.path.append('/home/gdk/Repositories/DualArmManipulation')
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
from dataset.Dataset import Dataset
from tqdm import tqdm
import trimesh
from dataset.utils import shift_mass_center
import pandas as pd
import pickle
import time
import json

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet_baseline/models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet_baseline/dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet_baseline/utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(data_dir):
    # load data
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]
    # save_numpy_pic(cloud_sampled, color_sampled)
    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_and_process_our_data(data, device):
    # object_id = '155'
    # object_meshes = []
    # object_meshes_seperate = []
    # object_clouds = []
    
    # object_order = []
    # object_ids = dataset.get_object_ids()
    # for object_id in random.sample(object_ids, min(len(object_ids), 5)):
    # for object_id in object_ids:
    # object_order.append(object_id)
    meshes = data.load_meshes()        
    # object_meshes_seperate.append(meshes)
    mesh = trimesh.util.concatenate(meshes)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array(mesh.vertices).astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(np.zeros_like(np.array(mesh.vertices)).astype(np.float32))
    mesh_sample = torch.from_numpy(np.array(mesh.vertices)[np.newaxis].astype(np.float32))
    mesh_sample = mesh_sample.to(device)
    mesh_dict = dict()
    mesh_dict['point_clouds'] = mesh_sample
    # object_meshes.append(mesh_dict)
    # object_clouds.append(cloud)
    return mesh_dict, cloud, meshes

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])
        #     vis_p = o3d.geometry.PointCloud()
        # vis_p.points = o3d.utility.Vector3dVector(np.array([p1[0],p2[0]]))
        # vis_p.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (2, 1)))
        # grippers = gg.to_open3d_geometry_list()
        # o3d.visualization.draw_geometries([clouds[i], vis_p, *grippers])

def save_numpy_pic(points, colors):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible = True) #works for me with False, on some systems needs to be true
    mesh = o3d.geometry.PointCloud()
    mesh.points = o3d.utility.Vector3dVector(points)
    mesh.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(mesh)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image('/home/gdk/Repositories/DualArmManipulation/evaluation/graspnet_baseline/doc/test_pic/1.jpg', do_render=True)
    vis.destroy_window()    

def demo(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    # end_points, cloud = get_and_process_our_data(data_dir)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    vis_grasps(gg, cloud)

def test(data_dir):
    net = get_net()
    count = 0
    count_hun = 0
    pickle_file_path = 'data.pickle'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = Dataset(data_dir)
    json_file_path = '/home/gdk/Repositories/DualArmManipulation/data/dataset/test_object_ids.json'
    with open(json_file_path, 'r') as file:
        object_ids = json.load(file)
    # object_ids = ['155','42','4070']
    data_all = dict()
    for object_id in object_ids:
        count = count + 1
        end_points, clouds, object_meshes = get_and_process_our_data(dataset[object_id], device)
        data = []
        gg = get_grasps(net, end_points)
        if cfgs.collision_thresh > 0:
            gg = collision_detection(gg, np.array(clouds.points))
        gg.nms()
        gg.sort_by_score()
        gg = gg[:10]
        if len(gg) != 0:
            id1, id2, p1, p2 = compute_pose_id(gg.translations, gg.rotation_matrices, gg.widths, gg.depths, object_meshes)
            for j in range(len(p1)):
                tmp = (p1[j] - p2[j])/np.linalg.norm(p1[j] - p2[j])
                data_gg = np.hstack((np.array(id1[j]),np.array(p1[j]),-tmp,np.array(id2[j]),np.array(p2[j]),tmp))
                data.append(data_gg)
        data_all[object_id] = data
        if count%100 == 0:
            count_hun = count_hun + 1
            print("finished processing ", count_hun*100)
            count = 0
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(data_all, file)
        print("finish writing file")

        
def compute_pose_id(p, r, w, d, meshes):
    p1 = []
    p2 = []
    for i in range(len(p)):
        left_offset = np.array([d[i], -w[i]/2, 0])
        right_offset = np.array([d[i], w[i]/2, 0])
        p1.append(p[i] + np.dot(r[i], left_offset))
        p2.append(p[i] + np.dot(r[i], right_offset))
    id1=compute_all_id(p1, meshes)
    id2=compute_all_id(p2, meshes)
    return id1, id2, p1, p2

def compute_all_id(p, meshes):
    distance = np.zeros((len(meshes), len(p))) # (n, m)
    for id in range(len(meshes)):
        _, dis, _ = trimesh.proximity.closest_point(meshes[id], p)
        distance[id] = np.array(dis)
    return np.argmin(distance, axis=0)

def compute_id(contact_p, meshes):
    final_id = 0
    min = 1e8
    for id in range(len(meshes)):
        for p in meshes[id].vertices:
            length = np.linalg.norm(contact_p-p)
            if min > length:
                min = length
                final_id = id
    return final_id  

if __name__=='__main__':   
    data_dir = '/home/gdk/Repositories/DualArmManipulation/data/objects'
    test(data_dir)
    # pickle_file_path = 'data.pickle'
    # with open(pickle_file_path, 'rb') as file:
    #     loaded_dict = pickle.load(file)
    # print(loaded_dict)