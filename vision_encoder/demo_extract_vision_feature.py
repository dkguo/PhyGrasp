import logging
import pickle

import numpy as np
import torch
import trimesh

from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig, load_checkpoint

if __name__ == '__main__':
    # conda activate iae
    # cd vision_encoder
    # CUDA_VISIBLE_DEVICES=0 python demo_extract_vision_feature.py

    logging.basicConfig(level=logging.INFO)

    cfg = EasyConfig()
    cfg_path = './shapenetpart_pointnext-s_c64.yaml'
    pretrain_path = './shapenetpart-train-pointnext-s_c64-ngpus4-seed7798-20220822-024210-ZcJ8JwCgc7yysEBWzkyAaE_ckpt_best.pth'
    cfg.load(cfg_path, recursive=True)
    seg_model = build_model_from_cfg(cfg.model).cuda()
    load_checkpoint(seg_model, pretrained_path=pretrain_path)
    seg_model.eval()

    cfg = EasyConfig()
    cfg_path = './modelnet40_pointnext-s.yaml'
    pretrain_path = './modelnet40ply2048-train-pointnext-s-ngpus1-seed6848-model.encoder_args.width=64-20220525-145053-7tGhBV9xR9yQEBtN4GPcSc_ckpt_best.pth'
    cfg.load(cfg_path, recursive=True)
    clf_model = build_model_from_cfg(cfg.model).cuda()
    load_checkpoint(clf_model, pretrained_path=pretrain_path)
    clf_model.eval()

    objects_path = '/home/gdk/Repositories/DualArmManipulation/demo/demo_objects'

    object_names = ['banana', 'monitor', 'pill_bottle', 'plastic_hammer', 'hammer']

    object_name = object_names[4]

    mesh = trimesh.load(f'{objects_path}/{object_name}/{object_name}.obj')

    # rescale to [-1, 1] box
    rescale = max(mesh.extents) / 2.
    tform = [
        -(mesh.bounds[1][i] + mesh.bounds[0][i]) / 2.
        for i in range(3)
    ]
    matrix = np.eye(4)
    matrix[:3, 3] = tform
    # mesh.apply_transform(matrix)
    transform = matrix
    matrix = np.eye(4)
    matrix[:3, :3] /= rescale
    transform = np.dot(matrix, transform)
    # mesh.apply_transform(matrix)
    mesh.apply_transform(transform)

    points, idx = trimesh.sample.sample_surface(mesh, 2048)
    normals = mesh.face_normals[idx]

    heights = points[:, 2] - points[:, 2].min()
    pos = [points]
    x = [np.concatenate([points, normals, heights[:, np.newaxis]], axis=1).T]

    pos = torch.Tensor(np.array(pos)).cuda().contiguous()
    x = torch.Tensor(np.array(x)).cuda().contiguous()

    print(pos.shape, x.shape)

    inp = {'pos': pos,
           'x': x,
           'cls': torch.zeros(1, 16).long().cuda(),
           }

    local_features = seg_model(inp).detach().cpu().numpy()
    global_features = clf_model(inp['pos']).detach().cpu().numpy()

    print(local_features.shape, global_features.shape)

    print(local_features[0, :, :4])
    print(global_features[0])

    vision_features = {
        'points': points,
        'local_features': local_features,
        'global_features': global_features,
        'transform': transform,
    }

    pickle.dump(vision_features, open(f'{objects_path}/{object_name}/vision_features.pkl', 'wb'))