import argparse
import pickle

import numpy as np
import torch
import trimesh

from graspnetAPI import GraspGroup

from evaluation.graspnet_baseline.models.graspnet import GraspNet, pred_decode
from evaluation.graspnet_baseline.utils.collision_detector import ModelFreeCollisionDetector


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net


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

def gg_to_positions(p, r, w, d):
    p1 = []
    p2 = []
    for i in range(len(p)):
        left_offset = np.array([d[i], -w[i]/2, 0])
        right_offset = np.array([d[i], w[i]/2, 0])
        p1.append(p[i] + np.dot(r[i], left_offset))
        p2.append(p[i] + np.dot(r[i], right_offset))
    grasp_positions = np.hstack((p1, p2))   # (n, 6)
    return grasp_positions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
    parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
    parser.add_argument('--collision_thresh', type=float, default=0.01,
                        help='Collision Threshold in collision detection [default: 0.01]')
    parser.add_argument('--voxel_size', type=float, default=0.01,
                        help='Voxel Size to process point clouds before collision detection [default: 0.01]')
    cfgs = parser.parse_args()
    device = torch.device("cuda:0")

    objects_path = '/home/gdk/Repositories/DualArmManipulation/demo/demo_objects'

    net = get_net()

    object_name = 'banana'
    mesh = trimesh.load(f'{objects_path}/{object_name}/{object_name}.obj')
    points = trimesh.sample.sample_surface(mesh, 2048)[0]
    print(points.shape)

    end_points = {
        'point_clouds': torch.Tensor(points).unsqueeze(0).to(device),
    }

    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(points))
    gg.nms()
    gg.sort_by_score()
    gg = gg[:10]

    positions = gg_to_positions(gg.translations, gg.rotation_matrices, gg.widths, gg.depths)

    print(positions)

    pickle.dump(positions, open(f'{objects_path}/{object_name}/graspnet_grasps.pkl', 'wb'))

