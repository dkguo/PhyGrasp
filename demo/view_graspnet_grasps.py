import pickle

import trimesh


if __name__ == '__main__':
    objects_path = '/home/gdk/Repositories/DualArmManipulation/demo/demo_objects'
    object_name = 'hammer'

    mesh = trimesh.load(f'{objects_path}/{object_name}/{object_name}.obj')

    pos = pickle.load(open(f'{objects_path}/{object_name}/graspnet_grasps.pkl', 'rb'))
    for i in range(len(pos)):
        p1, p2 = pos[i, :3], pos[i, 3:]
        ball1 = trimesh.primitives.Sphere(radius=0.01, center=p1)
        ball1.visual.face_colors = [255, 0, 0, 255]
        ball2 = trimesh.primitives.Sphere(radius=0.01, center=p2)
        ball2.visual.face_colors = [0, 255, 0, 255]
        scene = [mesh, ball1, ball2]
        trimesh.Scene(scene).show()
