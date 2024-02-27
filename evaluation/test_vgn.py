import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append('/home/gdk/Repositories/DualArmManipulation')
from vgn.detection import VGN
from vgn.experiments import clutter_removal
from vgn.perception import *
from dataset.Dataset import Dataset
import json
import numpy as np
import trimesh
import pickle
import pyrender
from matplotlib import pyplot as plt
from IPython import embed


def main(args):

    if args.rviz or str(args.model) == "gpd":
        import rospy

        rospy.init_node("sim_grasp", anonymous=True)

    if str(args.model) == "gpd":
        from vgn.baselines import GPD

        grasp_planner = GPD()
    else:
        grasp_planner = VGN(args.model, rviz=args.rviz)

    count = 0
    count_hun = 0
    pickle_file_path = 'data_vgn_grasp.pickle'
    pickle_file_path_2 = 'data_vgn_map.pickle'
    dataset = Dataset('/home/gdk/Repositories/DualArmManipulation/data/objects')
    json_file_path = '/home/gdk/Repositories/DualArmManipulation/data/dataset/test_object_ids.json'
    # image = np.random.rand(480, 640).astype(np.float32)
    # with open(json_file_path, 'r') as file:
    #     object_ids = json.load(file)
    object_ids = ['10519', '9000', '4931', '11405', '13214', '2056', '5174', '10850', '16208']
    data_grasp_all = dict()
    data_vol_all = dict()
    fx, fy, cx, cy = 540.0, 540.0, 320.0, 240.0
    camera_in = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy, znear=0.01, zfar=2)
    intrinsic = CameraIntrinsic(640, 480, fx, fy, cx, cy)
    
    for object_id in object_ids:
        count = count + 1
        meshes = dataset[object_id].load_meshes()
        
        # TODO add transformation of the meshes here with mesh.apply_transform()
        
        grasps, scores, grasp_info, vol = clutter_removal.run_baseline(
            grasp_plan_fn=grasp_planner,
            meshes=meshes,
            intrinsic=intrinsic,
            camera_in =camera_in,
        )
        if len(scores) != 0:
        # best_g_p, best_g_r = grasps[0].pose.translation, grasps[0].pose.rotation.as_matrix()
            # data_all[object_id]['grasps'] = data
            # data_all[object_id]['map'] = scores
            # data_all[object_id]['grasp_info'] = grasp_info
            data_grasp_all[object_id] = grasp_info
            data_vol_all[object_id] = vol
            print(object_id)
            print("grasp_info", np.array(grasp_info).shape)
            print("vol", np.array(vol).shape)
        else:
            data_grasp_all[object_id] = []
            data_vol_all[object_id] = []
        if count%100 == 0:
            count_hun = count_hun + 1
            print("finished processing ", count_hun*100)
            count = 0
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(data_grasp_all, file)
        print("finish writing grasp file")
    with open(pickle_file_path_2, 'wb') as file_2:
        pickle.dump(data_vol_all, file_2)
        print("finish writing map file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default="./evaluation/vgn/data/models/vgn_conv.pth")
    parser.add_argument("--logdir", type=Path, default="data/experiments")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object-set", type=str, default="blocks")
    parser.add_argument("--num-objects", type=int, default=5)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--rviz", action="store_true")
    args = parser.parse_args()
    main(args)
    # pickle_file_path = 'object_count.pkl'
    # with open(pickle_file_path, 'rb') as file:
    #     loaded_dict = pickle.load(file)
    # print(loaded_dict)
    # with open('data2.csv', 'w') as f:
    #     [f.write('{0}\n'.format(key)) for key, value in loaded_dict.items()]
    #     [f.write('{0}\n'.format(value)) for key, value in loaded_dict.items()]
    # import pandas as pd
    # df = pd.DataFrame(loaded_dict)
    # df.to_csv('my_file.csv', index=False, header=True)