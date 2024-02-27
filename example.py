import numpy as np
import trimesh

from dataset.generate_data import min_distances
from grasp.generate_grasp import find_contact_points_multi


def vis_grasp_heatmap(mesh, contact_pairs):
    contact_locs = contact_pairs.reshape(-1, 3)
    dists = min_distances(mesh.vertices, contact_locs)
    # TODO: change to Gaussian
    mesh.visual.vertex_colors = trimesh.visual.color.interpolate(np.sqrt(dists), color_map='hot')
    mesh.visual.vertex_colors[:, 3] = 0.8 * 255
    scene_list = [mesh]

    for contact_point, another_contact_point in contact_pairs:
        # c1 = trimesh.creation.uv_sphere(radius=0.005)
        # c2 = trimesh.creation.uv_sphere(radius=0.005)
        # c1.vertices += contact_point
        # c2.vertices += another_contact_point
        grasp_axis = trimesh.creation.cylinder(0.005, sections=6,
                                               segment=np.vstack([contact_point, another_contact_point]))
        grasp_axis.visual.vertex_colors = [0, 0., 1.]
        # c1.visual.vertex_colors = [1., 0, 0]
        # c2.visual.vertex_colors = [1., 0, 0]
        # scene_list += [c1, c2, grasp_axis]
        scene_list += [grasp_axis]

    trimesh.Scene(scene_list).show()


if __name__ == '__main__':
    meshes = []
    for i in range(3):
        obj_path = f'./demo_objects/knife/meshes/new-{i}.obj'
        meshes.append(trimesh.load(obj_path))

    frictions = [0.5, 0.2, 0.1]
    sample_nums = [100, 20, 10]

    # for i in range(1, 7):
    #     obj_path = f'./demo_objects/table/meshes/original-{i}.obj'
    #     meshes.append(trimesh.load(obj_path))
    #
    # frictions = [0.5, 0.2, 0.1, 0.1, 0.1, 0.1]
    # sample_nums = [100, 20, 10, 10, 10, 10]

    combined_mesh = trimesh.util.concatenate(meshes)

    grasps = find_contact_points_multi(meshes, frictions, sample_nums)

    # vis_grasp(combined_mesh, np.array(grasps, dtype=object)[:, (1, 4)])

    vis_grasp_heatmap(combined_mesh, grasps[:, np.r_[1:4, 8:11]].reshape(-1, 2, 3))