import numpy as np
import trimesh
from tqdm import tqdm

from force_optimization import solve_force
from generate_grasp import find_contact_points, vis_grasp

if __name__ == '__main__':
    """A simple example with stanford bunny."""
    obj_path = 'demo_objects/stanford_bunny.obj'
    n_sample_point = 100
    friction = 1
    contact_rad = 0.05  # used for collision check

    # Load mesh
    mesh = trimesh.load(obj_path)
    # mesh.vertices -= mesh.center_mass
    # mesh.vertices /= np.linalg.norm(mesh.vertices, axis=1).max()

    contacts = find_contact_points(mesh, n_sample_point, friction, contact_rad)

    print(len(contacts), 'contacts found.')

    # vis_grasp
    # vis_grasp(mesh, contacts)

    grasps = []
    for contact_points, contact_normals in tqdm(contacts):
        force = solve_force(contact_points, contact_normals, friction, np.array([0.0, 0.0, 1, 0.0, 0.0, 0.0]), soft_contact=True)
        if force is not None:
            grasps.append((contact_points, contact_normals, force))

    print(grasps)






