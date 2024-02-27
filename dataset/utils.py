import numpy as np
import trimesh


def shift_mass_center(meshes, densities):
    weighted_mass_center = np.zeros(3)
    masses = []
    for mesh, density in zip(meshes, densities):
        center_mass = mesh.centroid if mesh.volume < 1e-6 else mesh.center_mass
        volume = mesh.convex_hull.volume if mesh.volume < 1e-6 else mesh.volume
        mass = volume * density
        weighted_mass_center += center_mass * mass
        masses.append(mass)
        # green_ball = trimesh.creation.uv_sphere(radius=0.01)
        # green_ball.visual.vertex_colors = [0.0, 1.0, 0.0]
        #
        # red_ball = trimesh.creation.uv_sphere(radius=0.01)
        # red_ball.visual.vertex_colors = [1.0, 0.0, 0.0]
        # red_ball.vertices += center_mass
        # trimesh.Scene([mesh, green_ball, red_ball]).show()
    mass_center = weighted_mass_center / sum(masses)

    for mesh in meshes:
        mesh.vertices -= mass_center

    # b = trimesh.creation.uv_sphere(radius=0.05)
    # b.visual.vertex_colors = [0.0, 1.0, 0.0]
    # trimesh.Scene([combined_mesh, b]).show()

    return masses


def compute_part_ids(p, meshes): # p: (n, 3)
    distance = np.zeros((len(meshes), len(p))) # (n, m)
    for id in range(len(meshes)):
        _, dis, _ = trimesh.proximity.closest_point(meshes[id], p)
        distance[id] = np.array(dis)
    return np.argmin(distance, axis=0)  # (n,)
