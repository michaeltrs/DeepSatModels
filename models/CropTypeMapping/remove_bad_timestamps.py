import numpy as np
import pickle
bad = np.load('/home/data/ghana/bad_timestamp_grids_list.npy', 'r')

for split in ['train', 'val', 'test']:
    print(split)
    for dataset in ['full', 'small']:
        grid_path = f'/home/data/ghana/ghana_{dataset}_{split}'
        with open(grid_path, "rb") as f:
            grids = list(pickle.load(f))
            if '025596' in grids:
                grids.remove('025596')
            for grid in bad:
                if grid in grids:
                    print("REMOVING: ", grid)
                    grids.remove(grid)
        with open(grid_path, 'wb') as f:
            pickle.dump(grids, f)


