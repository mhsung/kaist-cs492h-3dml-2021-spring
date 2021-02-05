# Minhyuk Sung (mhsung@kaist.ac.kr)

import glob
import h5py
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_data(h5_files):
    P, L = [], []
    for h5_file in h5_files:
        print(h5_file)
        with h5py.File(h5_file, 'r') as f:
            P.append(f['data'][:])
            L.append(np.squeeze(f['label'][:]))

    P, L = np.concatenate(P, 0), np.concatenate(L)
    return P, L


if __name__ == "__main__":
    data_dir = os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048')
    train_files = glob.glob(os.path.join(data_dir, 'ply_data_train*.h5'))
    test_files = glob.glob(os.path.join(data_dir, 'ply_data_test*.h5'))
    train_files.sort()
    test_files.sort()

    train_P, train_L = read_data(train_files)
    test_P, test_L = read_data(test_files)

    # Store the data.
    out_file = os.path.join(BASE_DIR, 'modelnet_classification.h5')
    with h5py.File(out_file, 'w') as f:
        f['train_point_clouds'], f['train_class_ids'] = train_P, train_L
        f['test_point_clouds'], f['test_class_ids'] = test_P, test_L
        print("Saved '{}'.".format(out_file))
