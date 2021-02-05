# Minhyuk Sung (mhsung@kaist.ac.kr)

from joblib import Parallel, delayed
import glob
import h5py
import multiprocessing
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N_POINTS = 2048
TRAIN_RATIO = 0.8
np.random.seed(1234)


def read_point_cloud(pts_file):
    P = np.genfromtxt(pts_file)
    idxs = np.arange(np.shape(P)[0])
    np.random.shuffle(idxs)
    orig_idxs = np.copy(idxs)

    while np.size(idxs) < N_POINTS:
        n_remaining = min(N_POINTS - np.size(idxs), np.size(orig_idxs))
        idxs = np.concatenate((idxs, orig_idxs[:n_remaining]))
    idxs = idxs[:N_POINTS]
    P = P[idxs]

    # print("Loaded '{}'.".format(pts_file))
    return P


if __name__ == "__main__":
    # Read class names.
    class_names = []
    class_synsets = []
    with open(os.path.join(BASE_DIR, 'PartAnnotation'
                           'synsetoffset2category.txt'), 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            tokens = line.split('\t')
            assert(len(tokens) >= 2)
            class_names.append(tokens[0])
            class_synsets.append(tokens[1])

    print(class_names)
    print(class_synsets)

    n_classes = len(class_names)
    print('# classes: {}'.format(n_classes))

    # Write class names.
    with open(os.path.join(BASE_DIR, 'class_names.txt'), 'w') as f:
        f.write('\n'.join(class_names))

    P = []
    L = []

    # Read point clouds.
    for class_id in range(n_classes):
        class_points_dir = os.path.join(
            BASE_DIR, 'PartAnnotation', class_synsets[class_id], 'points')
        print(class_points_dir)

        pts_files = glob.glob(os.path.join(class_points_dir, '*.pts'))

        '''
        for pts_file in pts_files:
            read_point_cloud(pts_file)
        '''

        # Parallel processing.
        n_cores = multiprocessing.cpu_count()
        print('# cores: {}'.format(n_cores))
        class_P = Parallel(n_jobs=n_cores)(
            delayed(read_point_cloud)(pts_file) for pts_file in pts_files)
        class_P = np.stack(class_P)
        class_L = np.array([class_id] * np.shape(class_P)[0])

        P.append(class_P)
        L.append(class_L)

    P = np.concatenate(P, 0)
    L = np.concatenate(L)

    # Split train/test sets.
    n_data = np.shape(P)[0]
    assert(np.size(L) == n_data)
    n_train = round(float(n_data) * TRAIN_RATIO)
    print(n_data)
    print(n_train)

    idxs = np.arange(n_data)
    np.random.shuffle(idxs)
    train_idxs = idxs[:n_train]
    test_idxs = idxs[n_train:]

    train_P, train_L = P[train_idxs], L[train_idxs]
    test_P, test_L = P[test_idxs], L[test_idxs]

    # Store the data.
    out_file = os.path.join(BASE_DIR, 'shapenet_classification.h5')
    with h5py.File(out_file, 'w') as f:
        f['train_point_clouds'], f['train_class_ids'] = train_P, train_L
        f['test_point_clouds'], f['test_class_ids'] = test_P, test_L
        print("Saved '{}'.".format(out_file))
