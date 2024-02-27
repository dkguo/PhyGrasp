import pickle

import numpy as np
import pandas as pd

if __name__ == '__main__':
    seed = 100
    np.random.seed(seed)

    df = pd.read_pickle('/home/gdk/Repositories/DualArmManipulation/evaluation/results/analytical_pos_not_tested.pkl')
    print(len(df), len(df) // 10)
    max_index = len(df) // 10
    indices = np.random.choice(max_index, 10000, replace=False) * 10
    print(len(indices), min(indices), max(indices))
    df_test = df.loc[indices]
    entry_paths = [f'./data/objects/{object_id}/{object_id}_{config_id}_v1.pkl' for object_id, config_id in df_test[['object_id', 'config_id']].values]
    print(len(entry_paths))
    print(len(set(entry_paths)))
    print(entry_paths[:10])

    # pickle.dump(entry_paths, open('/home/gdk/Repositories/DualArmManipulation/data/dataset/test_entry_paths_v2.pkl', 'wb'))
    # pickle.dump(indices, open('/home/gdk/Repositories/DualArmManipulation/data/dataset/test_indices.pkl', 'wb'))


