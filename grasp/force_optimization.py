import numpy as np
import cvxpy as cp

from grasp.generate_grasp import parse_grasp


def normalize(x):
    mag = np.linalg.norm(x)
    if mag == 0:
        mag = mag + 1e-10
    return x / mag


def hat(v):
    if v.shape == (3, 1) or v.shape == (3,):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    else:
        raise ValueError


def generate_contact_frame(pos, normal):
    """Generate contact frame, whose z-axis aligns with the normal direction (inward to the object)
    """
    up = normalize(np.random.rand(3))
    z = normalize(normal)
    x = normalize(np.cross(up, z))
    y = normalize(np.cross(z, x))

    result = np.eye(4)
    result[0:3, 0] = x
    result[0:3, 1] = y
    result[0:3, 2] = z
    result[0:3, 3] = pos
    return result


def adj_T(frame):
    """Compute the adjoint matrix for the contact frame
    """
    assert frame.shape[0] == frame.shape[1] == 4, 'Frame needs to be 4x4'

    R = frame[0:3, 0:3]
    p = frame[0:3, 3]
    result = np.zeros((6, 6))
    result[0:3, 0:3] = R
    result[3:6, 0:3] = hat(p) @ R
    result[3:6, 3:6] = R
    return result


def compute_grasp_map(contact_pos, contact_normal, soft_contact=False):
    """ Computes the grasp map for all contact points.
    Check chapter 5 of http://www.cse.lehigh.edu/~trink/Courses/RoboticsII/reading/murray-li-sastry-94-complete.pdf for details.
    Args:
        contact_pos: location of contact in the object frame
        contact_normal: surface normals at the contact location, point inward !!!, N x 3, in the object frame
        soft_contact: whether use soft contact model. Defaults to False.
    Returns:
        G: grasp map for the contacts
    """
    n_point = len(contact_pos)

    # Compute the contact basis B
    if soft_contact:
        B = np.zeros((6, 4))
        B[0:3, 0:3] = np.eye(3)
        B[5, 3] = 1
    else:  # use point contact w/ friction
        B = np.zeros((6, 3))
        B[0:3, 0:3] = np.eye(3)

    # Compute the contact frames, adjoint matrix, and grasp map
    contact_frames = []
    grasp_maps = []
    for pos, normal in zip(contact_pos, contact_normal):
        contact_frame = generate_contact_frame(pos, normal)
        contact_frames.append(contact_frame)

        adj_matrix = adj_T(contact_frame)
        grasp_map = adj_matrix @ B
        grasp_maps.append(grasp_map)

    G = np.hstack(grasp_maps)
    assert G.shape == (6, n_point * B.shape[1]), 'Grasp map shape does not match'

    return G


def solve_force(contact_positions, contact_normals, frictions, weight, soft_contact=False):
    w_ext = np.array([0.0, 0.0, weight, 0.0, 0.0, 0.0])

    num_contact = len(contact_positions)
    f = cp.Variable(4 * num_contact) if soft_contact else cp.Variable(3 * num_contact)
    s = cp.Variable(1)

    G = compute_grasp_map(contact_pos=contact_positions, contact_normal=contact_normals, soft_contact=soft_contact)

    constraints = [
        G @ f == - w_ext,
        s >= -1
    ]

    # cp.SOC(t, x) creates the SOC constraint ||x||_2 <= t.
    for i in range(num_contact):
        constraints += [
            cp.SOC(frictions[i] * (f[3 * i + 2] + s),
                   f[3 * i: 3 * i + 2])
        ]

    prob = cp.Problem(cp.Minimize(s), constraints)
    prob.solve()

    if f.value is None:
        # print("Cannot find a feasible solution")
        return None

    # print("The optimal value for s is", prob.value)
    # print("The optimal value for f is", f.value.reshape(num_contact, -1))
    return f.value.reshape(num_contact, -1)


def filter_contact_points_by_force(grasps, part_frictions, part_max_normal_forces, weight):
    neg_grasps, pos_grasps = [], []
    for grasp in grasps:
        part_idx_1, contact_point_1, contact_normal_1, part_idx_2, contact_point_2, contact_normal_2 = parse_grasp(grasp)
        contact_postions = [contact_point_1, contact_point_2]
        contact_normals = [contact_normal_1, contact_normal_2]
        frictions = [part_frictions[part_idx_1], part_frictions[part_idx_2]]
        try:
            forces = solve_force(contact_postions, contact_normals, frictions, weight, soft_contact=True)
        except:
            forces = None
        if forces is None \
                or abs(forces[0, 2]) > part_max_normal_forces[part_idx_1] \
                or abs(forces[1, 2]) > part_max_normal_forces[part_idx_2]:
            neg_grasps.append(grasp)
        else:
            pos_grasps.append(grasp)
    return np.array(pos_grasps), np.array(neg_grasps)


if __name__ == '__main__':
    contact_pos = np.array([[0, 1, 0], [0, -1, 0]])
    contact_normal = np.array([[0, -1, 0], [0, 1, 0]])

    print(solve_force(contact_pos, contact_normal, [0.5, 0.5], soft_contact=True))

    grasps = np.array([[0, 0, 1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 1, 0],
                       [0, 0, 1, 0, 0, -1, 0, 1, 0, -1, 0, 0, -1, 0]])
    part_frictions = [0.3, 0.5]
    max_forces = [0.5, 0.5]
    pos_grasps, neg_grasps = filter_contact_points_by_force(grasps, part_frictions, max_forces, 1.0)
    print(pos_grasps, neg_grasps)
