import trimesh
import pickle
import numpy as np
import argparse
import sys
sys.path.append("/home/gdk/Repositories/DualArmManipulation")
from model.grasp_utils import show_heatmap, show_embedding_map, show_score_map

def main(params):
    filename = './checkpoints/maps_{}_{}.pkl'.format(params['model_id'], params['batch_id'])
    
    maps = pickle.load(open(filename, 'rb'))
    points = maps['point_cloud']
    grasp_map = maps['grasp_map']
    prediction = maps['prediction']
    entry_path = maps['entry_path']
    embeddings = maps['embeddings']
    # best_index = maps['index1']
    index20 = maps['index20']
    pos = maps['pos']
    score = maps['score']
    print('point_cloud shape: ', points.shape, 'grasp_map shape: ', grasp_map.shape, 'prediction shape: ', prediction.shape, 'entry_path len: ', len(entry_path))
    np.set_printoptions(threshold=np.inf)

    for id in range(len(entry_path)):
        print('entry_path: ', entry_path[id])
        with open(entry_path[id], 'rb') as f:
            entry = pickle.load(f)
            print(entry.language)
        show_heatmap(points[id], prediction[id])

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_id', type=str, default='1706613034.3918543')
    argparser.add_argument('--batch_id', type=int, default=1)
    args = argparser.parse_args()
    params = vars(args)
    main(params)