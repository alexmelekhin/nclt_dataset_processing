import argparse
import re
from pathlib import Path
import pickle

import numpy as np
import tqdm


def is_track_subdir(p):
    return re.match(r"^201[2-3]-((0[1-9])|(1[0-2]))-(0[1-9]|[1-2][0-9]|3[0-1])$", str(p))


def index_image_timestamps(dataset_root: str):
    images_root = Path(dataset_root) / 'images'
    tracks = [x for x in images_root.iterdir() if x.is_dir() and is_track_subdir(x.name)]
    
    image_timestamps = []
    for track in tracks:
        track_images_subdir = track / 'lb3_small' / 'Cam1'
        track_timestamps = list(track_images_subdir.glob('*.png'))
        track_timestamps = [int(ts.stem) for ts in track_timestamps]
        image_timestamps.extend(track_timestamps)
    
    return np.sort(image_timestamps)


def index_lidar_timestamps(dataset_root: str):
    lidar_root = Path(dataset_root) / 'velodyne_data'
    tracks = [x for x in lidar_root.iterdir() if x.is_dir() and is_track_subdir(x.name)]
    
    lidar_timestamps = []
    for track in tracks:
        track_lidar_subdir = track / 'velodyne_sync'
        track_timestamps = list(track_lidar_subdir.glob('*.bin'))
        track_timestamps = [int(ts.stem) for ts in track_timestamps]
        lidar_timestamps.extend(track_timestamps)
    
    return np.sort(lidar_timestamps)


# find closest place timestamp with index returned
def find_closest_timestamp(A, target):
    """
    Source: https://github.com/MaverickPeter/DiSCO-pytorch
    """
    # A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx


def create_img2lidar_ndx(image_timestamps, lidar_timestamps, threshold=100):
    print('Creating img2lidar index...')
    delta_l = []
    img2lidar_ndx = {}
    count_above_thresh = 0

    for img_ts in tqdm.tqdm(image_timestamps):
        lidar_ts_ndx = find_closest_timestamp(lidar_timestamps, img_ts)
        nn_ts = lidar_timestamps[lidar_ts_ndx]
        nn_dist = np.abs(nn_ts - img_ts)
        # threshold is in miliseconds
        if (nn_dist > threshold*1000).sum() > 0:
            count_above_thresh += 1
            continue  # skip

        # Remember timestamps of closest images
        img2lidar_ndx[img_ts] = nn_ts
        delta_l.append(nn_dist)

    delta_l = np.array(delta_l, dtype=np.float32)
    s = 'Distance between the image and closest lidar scan (min/mean/max): {} / {} / {} [ms]'
    print(s.format(int(np.min(delta_l)/1000), int(np.mean(delta_l)/1000), int(np.max(delta_l)/1000)))
    print('{} images without scan within {} [ms] threshold'.format(count_above_thresh, int(threshold)))

    return img2lidar_ndx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    assert dataset_root.exists()

    image_timestamps = index_image_timestamps(dataset_root)
    print(f"Number of image timestamps: {image_timestamps.shape[0]}")

    lidar_timestamps = index_lidar_timestamps(dataset_root)
    print(f"Number of lidar timestamps: {lidar_timestamps.shape[0]}")

    img2lidar_ndx = create_img2lidar_ndx(image_timestamps, lidar_timestamps)
    print(f"Timestamps in index: {len(img2lidar_ndx)}")

    ndx_filepath = dataset_root / 'img2lidar_ndx.pickle'
    with open(ndx_filepath, 'wb') as f:
        pickle.dump(img2lidar_ndx, f)
    print(f"Saved index to: {ndx_filepath}")
