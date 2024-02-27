import numpy as np
import trimesh

import logging

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

"""
Install
pip install numpy trimesh rtree
"""


def sample_from_cone(tx, ty, tz, friction_coef=0.3):
    """ Samples directoins from within the friction cone using uniform sampling.
    
    Parameters
    ----------
    tx : 3x1 normalized :obj:`numpy.ndarray`
        tangent x vector
    ty : 3x1 normalized :obj:`numpy.ndarray`
        tangent y vector
    tz : 3x1 normalized :obj:`numpy.ndarray`
        surface normal
    num_samples : int
        number of directions to sample

    Returns
    -------
    v_samples : :obj:`list` of 3x1 :obj:`numpy.ndarray`
        sampled directions in the friction cone
    """
    theta = 2 * np.pi * np.random.rand()
    r = friction_coef * np.random.rand()
    v = tz + r * np.cos(theta) * tx + r * np.sin(theta) * ty
    return normalize(v)


def normalize(x):
    norm = np.linalg.norm(x)
    norm = 1e-5 if norm == 0 else norm
    return x / norm


def find_contact(mesh, vertices, directions, offset=20.0):
    start_points = vertices + offset * directions
    hit_indices = mesh.ray.intersects_first(ray_origins=start_points, ray_directions=-directions)
    return hit_indices


def vis_grasp(mesh, contact_points):
    mesh.visual.vertex_colors = [0.5, 0.5, 0.5, 0.5]
    scene_list = [mesh]
    # scene_list = []

    for contact_point, another_contact_point in contact_points:
        c1 = trimesh.creation.uv_sphere(radius=0.015)
        c2 = trimesh.creation.uv_sphere(radius=0.015)
        c1.vertices += contact_point
        c2.vertices += another_contact_point
        grasp_axis = trimesh.creation.cylinder(0.005, sections=6,
                                               segment=np.vstack([contact_point, another_contact_point]))
        grasp_axis.visual.vertex_colors = [1.0, 0.0, 0.0]
        c1.visual.vertex_colors = [0.0, 1.0, 0.0]
        c2.visual.vertex_colors = [0.0, 1.0, 0.0]
        scene_list += [c1, c2, grasp_axis]

    trimesh.Scene(scene_list).show()


def generate_contact_rays(surface_vertices, surface_normals, friction):
    rays = np.zeros_like(surface_vertices)
    for i, (vertice, normal) in enumerate(zip(surface_vertices, surface_normals)):
        tz = normalize(normal)
        up = normalize(np.random.rand(3))
        tx = normalize(np.cross(tz, up))
        ty = normalize(np.cross(tz, tx))
        ray = sample_from_cone(tx, ty, tz, friction_coef=friction)
        rays[i] = ray
    return rays


def sample_contact_points(mesh, n_sample_point):
    surface_vertices, face_idx = trimesh.sample.sample_surface_even(mesh, count=n_sample_point)
    surface_vertices = np.asarray(surface_vertices)
    surface_normals = - mesh.face_normals[face_idx]  # flip normals to point inward
    return surface_vertices, surface_normals


def check_collision_points(contact_rad, points_volume, contact_point, another_contact_point, contact_normal,
                           another_contact_normal):
    center_1 = contact_point - contact_rad * contact_normal
    center_2 = another_contact_point - contact_rad * another_contact_normal
    dist_1 = np.linalg.norm(points_volume - center_1, axis=1)
    dist_2 = np.linalg.norm(points_volume - center_2, axis=1)
    c1_not_in_collision = all(dist_1 > contact_rad)
    c2_not_in_collision = all(dist_2 > contact_rad)
    is_collision_free = c1_not_in_collision and c2_not_in_collision
    return is_collision_free


def find_contact_points(mesh, n_sample_point, friction, contact_rad):
    surface_vertices, surface_normals = sample_contact_points(mesh, n_sample_point)
    rays = generate_contact_rays(surface_vertices, surface_normals, friction)
    hit_indices = find_contact(mesh, surface_vertices, rays, offset=np.linalg.norm(mesh.extents) * 4.0)


    alpha = np.arctan(friction)
    grasps = []

    # Sample surface and volume point cloud for collision check
    points_volume = trimesh.sample.volume_mesh(mesh, count=4096)
    points_surface, _ = trimesh.sample.sample_surface_even(mesh, count=2048)
    points_volume = np.vstack([points_volume, points_surface])

    for contact_point, contact_normal, ray, hit_index in zip(surface_vertices, surface_normals, rays, hit_indices):
        if hit_index == -1:
            continue
        another_contact_point, another_contact_normal = mesh.triangles_center[hit_index], -mesh.face_normals[hit_index]

        # Check whether force closure
        is_force_closure = np.arccos(-ray.dot(another_contact_normal)) <= alpha

        # Check whether collision free
        if is_force_closure:
            is_collision_free = check_collision_points(contact_rad, points_volume, contact_point, another_contact_point,
                                                       contact_normal, another_contact_normal)

            if is_collision_free:
                # g = np.hstack([contact_point, another_contact_point, contact_normal, another_contact_normal])
                grasps.append((np.array([contact_point, another_contact_point]),
                               np.array([contact_normal, another_contact_normal])))

    return grasps


def find_contact_points_multi(meshes, frictions, sample_nums, contact_rad=0.05):
    """
    Return (n, 14) vector: part_idx_1, contact_point_1, contact_normal_1, part_idx_2, contact_point_2, contact_normal_2
    """
    part_indices, surface_vertices, surface_normals, rays = [], [], [], []
    for i, (mesh, friction, n_sample) in enumerate(zip(meshes, frictions, sample_nums)):
        v, n = sample_contact_points(mesh, n_sample)
        r = generate_contact_rays(v, n, friction)
        surface_vertices.extend(v)
        surface_normals.extend(n)
        rays.extend(r)
        part_indices.extend([i] * n_sample)

    combined_mesh = trimesh.util.concatenate(meshes)
    hit_indices = find_contact(combined_mesh, surface_vertices, np.array(rays), offset=np.linalg.norm(combined_mesh.extents) * 4.0)
    part_i_thres = np.cumsum([len(mesh.face_normals) for mesh in meshes])
    hit_part_indices = np.sum(np.repeat([hit_indices], len(meshes), axis=0).T >= part_i_thres, axis=1)

    # Sample surface and volume point cloud for collision check
    # points_volume = trimesh.sample.volume_mesh(combined_mesh, count=4096)
    # points_surface, _ = trimesh.sample.sample_surface_even(combined_mesh, count=2048)
    # points_volume = np.vstack([points_volume, points_surface])

    pos_grasps = []
    neg_grasps = []
    for part_idx_1, contact_point_1, contact_normal_1, ray, hit_idx, part_idx_2 in zip(
            part_indices, surface_vertices, surface_normals, rays, hit_indices, hit_part_indices):
        if hit_idx == -1:
            continue
        contact_point_2, contact_normal_2 = combined_mesh.triangles_center[hit_idx], -combined_mesh.face_normals[hit_idx]

        grasp = np.concatenate(([part_idx_1], contact_point_1, contact_normal_1, [part_idx_2], contact_point_2, contact_normal_2))

        if np.arccos(-ray.dot(contact_normal_2)) <= np.arctan(frictions[part_idx_2]):# and \
                # check_collision_points(contact_rad, points_volume, contact_point_1, contact_point_2,
                #                        contact_normal_1, contact_normal_2):
            pos_grasps.append(grasp)
        else:
            neg_grasps.append(grasp)
    return np.array(pos_grasps), np.array(neg_grasps)


def parse_grasp(grasp):
    """
    :param grasp: (14,) numpy array
    :return: part_idx_1, contact_point_1, contact_normal_1, part_idx_2, contact_point_2, contact_normal_2
    """
    part_idx_1, contact_point_1, contact_normal_1, part_idx_2, contact_point_2, contact_normal_2 = \
        int(grasp[0]), grasp[1:4], grasp[4:7], int(grasp[7]), grasp[8:11], grasp[11:14]
    return part_idx_1, contact_point_1, contact_normal_1, part_idx_2, contact_point_2, contact_normal_2


if __name__ == "__main__":
    obj_path = 'stanford_bunny.obj'
    n_contact_point = 100
    friction = 0.2
    contact_rad = 0.05  # used for collision check

    # Load mesh
    mesh = trimesh.load(obj_path)

    # Compute grasps
    grasps = find_contact_points(mesh, n_contact_point, friction, contact_rad)

    # Vis
    vis_grasp(mesh, grasps)
