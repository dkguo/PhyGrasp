import os.path
import pickle

import numpy as np
import torch
import trimesh
from sklearn.cluster import KMeans

from model.data_utils import ramdom_sample_pos
from model.model import Net
from scripts.plot_affordance_map import show_heatmap


def cluster_points(points, ncluster=5):
    """
    Function to cluster a set of points into 5 groups using K-Means algorithm.

    Parameters:
    data (numpy.ndarray): A 2D array of shape (20, 3), where each row represents a point in 3D space.

    Returns:
    list: A list containing the first index of each cluster.
    """

    # Perform K-Means clustering to divide the points into 5 clusters
    kmeans = KMeans(n_clusters=ncluster)
    kmeans.fit(points)

    # Get the cluster labels for each point
    labels = kmeans.labels_

    # Prepare the result: list of first indices for each cluster
    clusters = {i: [] for i in range(5)}
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    print(clusters)


    # Extracting the first index of each cluster
    first_indices = [cluster[0] for cluster in clusters.values()]

    return first_indices




if __name__ == '__main__':
    seed = 32
    np.random.seed(seed)
    torch.manual_seed(seed)

    objects_path = '/home/gdk/Repositories/DualArmManipulation/demo/demo_objects'
    object_name = 'hammer'
    # language_level = 'simple'
    language_level = 'detailed'

    mesh = trimesh.load(f'{objects_path}/{object_name}/{object_name}.obj')

    if False and os.path.exists(f'{objects_path}/{object_name}/{language_level}_grasps.pkl'):
        print('loading grasps')
        pos = pickle.load(open(f'{objects_path}/{object_name}/{language_level}_grasps.pkl', 'rb'))
        k1, k2, _ = pos.shape
        for i in range(k1):
            for j in range(k2):
                print(i, j)
                p1, p2 = pos[i, j, :3], pos[i, j, 3:]
                ball1 = trimesh.primitives.Sphere(radius=0.01, center=p1)
                ball1.visual.face_colors = [255, 0, 0, 255]
                ball2 = trimesh.primitives.Sphere(radius=0.01, center=p2)
                ball2.visual.face_colors = [0, 255, 0, 255]
                scene = [mesh, ball1, ball2]
                trimesh.Scene(scene).show()
        exit()


    language_feature = pickle.load(open(f'{objects_path}/{object_name}/{language_level}_language_feature_20.pkl', 'rb'))
    language_feature = torch.Tensor(language_feature).unsqueeze(0)
    vision_features = pickle.load(open(f'{objects_path}/{object_name}/vision_features.pkl', 'rb'))
    vision_local = torch.Tensor(vision_features['local_features']).permute(0, 2, 1)
    vision_global = torch.Tensor(vision_features['global_features'])
    points = torch.Tensor(vision_features['points']).unsqueeze(0)
    matrix = vision_features['transform']
    matrix = np.linalg.inv(matrix)
    # mesh.apply_transform(matrix)

    print(language_feature.shape, points.shape, vision_local.shape, vision_global.shape)

    model = Net()
    checkpoint_path = '/home/gdk/Repositories/DualArmManipulation/checkpoints/model_1706605305.8925593_39.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    output_global, output_local = model(language_feature, points, vision_global, vision_local)

    # show grasp map
    pcd_points = points.squeeze().cpu().numpy()
    pcd_points = pcd_points @ matrix[:3, :3].T + matrix[:3, 3]
    mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=[255, 0, 0, 255])
    show_heatmap(pcd_points, output_global.squeeze().detach().cpu().numpy(), mesh)

    k1, k2, f1, f2 = 5, 5, 4, 10
    index1 = torch.topk(output_global.squeeze(), k=k1 * f1).indices  # (batch_size, kp1)
    index1 = index1.unsqueeze(0)
    pos = model.get_pos(output_local, index1, points, kp2=k2 * f2)  # (batch_size, kp1, kp2, 6)
    # pos = ramdom_sample_pos(pos)  # (batch_size, 5, 6)
    pos = pos.squeeze(0).cpu().numpy() # (kp1, kp2, 6)

    grasp1_pos = pos[:, 0, 3:]
    grasp1_index = cluster_points(grasp1_pos, ncluster=k1)
    print(grasp1_index)
    pos = pos[grasp1_index] # (k1, kp2, 6)
    new_pos = np.zeros((k1, k2, 6))
    for i in range(k1):
        grasp2_pos = pos[i, :, :3]
        grasp2_index = cluster_points(grasp2_pos, ncluster=k2)
        new_pos[i] = pos[i][grasp2_index]
    pos = new_pos

    for i in range(k1):
        for j in range(k2):
            print(i, j)
            p1, p2 = pos[i, j, :3], pos[i, j, 3:]
            p1 = matrix[:3, :3] @ p1 + matrix[:3, 3]
            p2 = matrix[:3, :3] @ p2 + matrix[:3, 3]
            pos[i, j, :3], pos[i, j, 3:] = p1, p2
            ball1 = trimesh.primitives.Sphere(radius=0.0075, center=p1)
            ball1.visual.face_colors = [0, 255, 0, 255]
            ball2 = trimesh.primitives.Sphere(radius=0.0075, center=p2)
            ball2.visual.face_colors = [0, 255, 0, 255]
            mesh.visual.vertex_colors[:, 3] = 0.8 * 255
            scene = [mesh, ball1, ball2]
            trimesh.Scene(scene).show()

    pickle.dump(pos, open(f'{objects_path}/{object_name}/{language_level}_grasps.pkl', 'wb'))

