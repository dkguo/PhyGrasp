import numpy as np
import trimesh



def show_heatmap(points, grasp_map):
    pcd = trimesh.PointCloud(points)
    pcd.visual.vertex_colors = trimesh.visual.color.interpolate(grasp_map, color_map='hot')
    pcd.visual.vertex_colors[:, 3] = 0.8 * 255
    scene_list = [pcd]
    trimesh.Scene(scene_list).show()

'''
def show_score_map(pid, points, score,):
    pcd = trimesh.PointCloud(points)
    pid2 = np.argmax(score)
    score_max = np.max(score)
    score_min = np.min(score)
    print("score_min {}, score_max {}".format(score_min, score_max))
    print("score {}".format(score))
    score = (score - score_min) / (score_max - score_min)

    pcd.visual.vertex_colors = trimesh.visual.color.interpolate(score, color_map='hot')
    pcd.visual.vertex_colors[:, 3] = 0.8 * 255
    # highlight the point pid with larger size and different color
    pcd.visual.vertex_colors[pid] = [0, 255., 0., .8 * 255]
    pcd.visual.vertex_colors[pid2] = [0, 0., 255., .8 * 255]
    scene_list = [pcd]
    trimesh.Scene(scene_list).show()
'''
    
def show_score_map(pid, points, score,):
    pcd = trimesh.PointCloud(points)
    pid2 = np.argmax(score)
    score_max = np.max(score)
    score_min = np.min(score)
    print("score_min {}, score_max {}".format(score_min, score_max))
    print("score {}".format(score))
    score = (score - score_min) / (score_max - score_min)
    score = np.square(score)

    pcd.visual.vertex_colors = trimesh.visual.color.interpolate(score, color_map='cividis')
    pcd.visual.vertex_colors[:, 3] = 1.0 * 255
    # highlight the point pid with larger size and different color
    # pcd.visual.vertex_colors[pid] = [0, 255., 0., .8 * 255]
    # pcd.visual.vertex_colors[pid2] = [0, 0., 255., .8 * 255]
    ball = trimesh.creation.uv_sphere(radius=0.025)
    ball.visual.vertex_colors = [255., 0., 0., .8 * 255]
    ball.apply_translation(points[pid])
    ball2 = trimesh.creation.uv_sphere(radius=0.025)
    ball2.visual.vertex_colors = [255., 255., 0., .8 * 255]
    ball2.apply_translation(points[pid2])
    
    scene_list = [pcd, ball, ball2]
    trimesh.Scene(scene_list).show()

def show_embedding_map(pid, points, embeddings):
    embed_dists = embeddings_map(pid, embeddings)
    pcd = trimesh.PointCloud(points)
    pcd.visual.vertex_colors = trimesh.visual.color.interpolate(embed_dists, color_map='hot')

    pcd.visual.vertex_colors[:, 3] = 0.8 * 255
    # highlight the point pid with larger size and different color

    pcd.visual.vertex_colors[pid] = [0, 255., 0., .8 * 255]

    scene_list = [pcd]
    trimesh.Scene(scene_list).show()

def embeddings_map(pid, embeddings):
    assert embeddings.shape == (2048, 32), "embeddings shape is {}".format(embeddings.shape)
    embed_dists = np.linalg.norm(embeddings - embeddings[pid], axis=-1, ord=2)
    embed_dist_copy = embed_dists.copy()
    embed_min = np.min(embed_dist_copy)
    embed_max = np.max(embed_dist_copy)
    print("embed_min {}, embed_max {}".format(embed_min, embed_max))
    embed_dist_copy = np.delete(embed_dist_copy, pid)
    embed_min = np.min(embed_dist_copy)
    embed_max = np.max(embed_dist_copy)
    print("embed_min {}, embed_max {}".format(embed_min, embed_max))
    print("embed_dists {}".format(embed_dists))
    embed_dists = (embed_dists - embed_min) / (embed_max - embed_min)
    embed_dists[pid] = 0.
    # print("embed_dists {}".format(embed_dists))

    return embed_dists