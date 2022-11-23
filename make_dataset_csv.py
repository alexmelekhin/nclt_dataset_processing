import argparse
import re
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import tqdm


def is_track_subdir(p):
    return re.match(r"^201[2-3]-((0[1-9])|(1[0-2]))-(0[1-9]|[1-2][0-9]|3[0-1])$", str(p))


def read_poses(filepath):
    gt = np.loadtxt(filepath, delimiter = ",")
    gt = gt[1:]  # skip first row with nans
    poses = gt[:, [0, 1, 2]]
    return poses


def index_image_files(dataset_root):
    images_root = Path(dataset_root) / 'images'
    tracks = [x for x in images_root.iterdir() if x.is_dir() and is_track_subdir(x.name)]
    
    images_list = []
    for track in tracks:
        track_images_subdir = track / 'lb3_small' / 'Cam1'
        track_images = list(track_images_subdir.glob('*.png'))
        images_list.extend(track_images)
    
    return sorted(images_list)


def index_lidar_files(dataset_root):
    lidar_root = Path(dataset_root) / 'velodyne_data'
    tracks = [x for x in lidar_root.iterdir() if x.is_dir() and is_track_subdir(x.name)]
    
    lidars_list = []
    for track in tracks:
        track_lidar_subdir = track / 'velodyne_sync'
        track_lidars = list(track_lidar_subdir.glob('*.bin'))
        lidars_list.extend(track_lidars)
    
    return sorted(lidars_list)


def index_poses(dataset_root):
    gt_root = Path(dataset_root) / 'ground_truth'
    gt_files = sorted(list(gt_root.glob('groundtruth_*.csv')))

    poses = []
    for track_file in gt_files:
        track_poses = read_poses(track_file)
        poses.append(track_poses)
    poses = np.concatenate(poses, axis=0)

    return poses


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


def timestamps_from_paths(filepaths):
    return np.array([int(filepath.stem) for filepath in filepaths])


def create_csv_index(image_filepaths, lidar_filepaths, poses, threshold=50):
    print('Creating dataset index...')
    count_above_thresh = 0

    image_timestamps = timestamps_from_paths(image_filepaths)
    lidar_timestamps = timestamps_from_paths(lidar_filepaths)
    poses_timestamps = poses[:, 0].astype(int)

    csv_data = []

    for img_ndx, img_ts in tqdm.tqdm(enumerate(image_timestamps), total=len(image_timestamps)):
        lidar_ts_ndx = find_closest_timestamp(lidar_timestamps, img_ts)
        pose_ts_ndx = find_closest_timestamp(poses_timestamps, img_ts)
        lidar_ts = lidar_timestamps[lidar_ts_ndx]
        pose_ts = poses_timestamps[pose_ts_ndx]
        lidar_dist = np.abs(lidar_ts - img_ts)
        pose_dist = np.abs(pose_ts - img_ts)
        # threshold is in miliseconds
        if lidar_dist > threshold*1000 or pose_dist > threshold*1000:
            count_above_thresh += 1
            continue  # skip

        image_filename = image_filepaths[img_ndx].name
        track = image_filepaths[img_ndx].parts[-4]
        assert track == lidar_filepaths[lidar_ts_ndx].parts[-3]
        lidar_filepath = lidar_filepaths[lidar_ts_ndx].name
        pose_x = poses[pose_ts_ndx, 1]
        pose_y = poses[pose_ts_ndx, 2]
        csv_data.append((track, image_filename, lidar_filepath, pose_x, pose_y))

    index_df = pd.DataFrame(data = csv_data, columns=['track', 'image', 'lidar', 'northing', 'easting'])

    print('{} images without scan or pose within {} [ms] threshold'.format(count_above_thresh, int(threshold)))

    return index_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    assert dataset_root.exists()

    image_filepaths = index_image_files(dataset_root)
    print(f"Number of image files: {len(image_filepaths)}")

    lidar_filepaths = index_lidar_files(dataset_root)
    print(f"Number of lidar files: {len(lidar_filepaths)}")

    poses = index_poses(dataset_root)
    print(f"Number of poses: {len(poses)}")

    index_df = create_csv_index(image_filepaths, lidar_filepaths, poses)
    print(f"Index len: {len(index_df)}")
    print(index_df.head())

    csv_filepath = (dataset_root / 'dataset_index.csv').resolve()
    index_df.to_csv(csv_filepath)
    print(f"Saved index to: {csv_filepath}")    
