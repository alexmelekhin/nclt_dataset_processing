"""
Create video for all cameras in given track
"""

import argparse
from pathlib import Path
import struct

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas
import matplotlib
matplotlib.use('Agg')  # using non-GUI backend solves OOM issue and fasten the processing


def read_pointcloud(bin_filename):
    """
    Source: https://robots.engin.umich.edu/nclt/index.html
    Nicholas Carlevaris-Bianco, Arash K. Ushani, and Ryan M. Eustice,
        University of Michigan North Campus Long-Term Vision and Lidar Dataset,
        International Journal of Robotics Research, 2016.
    """
    def convert(x_s, y_s, z_s):

        scaling = 0.005 # 5 mm
        offset = -100.0

        x = x_s * scaling + offset
        y = y_s * scaling + offset
        z = z_s * scaling + offset

        return x, y, z

    f_bin = open(bin_filename, "rb")

    hits = []

    while True:

        x_str = f_bin.read(2)
        if x_str == b'': # eof
            break

        x = struct.unpack('<H', x_str)[0]
        y = struct.unpack('<H', f_bin.read(2))[0]
        z = struct.unpack('<H', f_bin.read(2))[0]
        i = struct.unpack('B', f_bin.read(1))[0]
        l = struct.unpack('B', f_bin.read(1))[0]

        x, y, z = convert(x, y, z)

        hits += [[x, y, z]]

    f_bin.close()

    hits = np.asarray(hits)

    return hits


def fig_to_image(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    # convert canvas to image
    graph_image = np.array(fig.canvas.get_renderer()._renderer)
    # it still is rgb, convert to opencv's default bgr
    graph_image = cv2.cvtColor(graph_image,cv2.COLOR_RGB2BGR)
    return graph_image


def make_pc_image(points):
    def center_crop(img, dim):
        """Returns center cropped image
        Args:
        img: image to be center cropped
        dim: dimensions (width, height) to be cropped
        """
        width, height = img.shape[1], img.shape[0]

        # process crop width and height for max available dimension
        crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
        crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
        mid_x, mid_y = int(width/2), int(height/2)
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
        return crop_img
    dists = np.linalg.norm(points, axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=60., azim=-180)
    ax.scatter(points[:, 0], points[:, 1], -points[:, 2], c=dists, s=0.5, linewidths=0)
    plt.axis('off')
    plt.tight_layout()
    
    graph_image = fig_to_image(fig)
    graph_image = center_crop(graph_image, (400, 400))

    plt.close()

    return graph_image


def read_camera_frame(imagename, cam_dirs):
    v_border = np.full((512, 2, 3), 255, dtype=np.uint8)  # little vertical border between frames
    h_border = np.full((2, 1932, 3), 255, dtype=np.uint8)  # horisontal top-down border
    cam_imgs = [v_border]
    for cam_dir in cam_dirs:
        cam_imgs.append(cv2.imread(str(cam_dir / imagename)))
        cam_imgs.append(v_border)
    full_frame = np.concatenate(cam_imgs, axis=1)
    full_frame = np.concatenate([h_border, full_frame, h_border], axis=0)
    return full_frame


def read_lidar_frame(bin_filename, lidar_dir):
    pc = read_pointcloud(lidar_dir / bin_filename)
    pc_image = make_pc_image(pc)
    return pc_image


def make_position_frame(cur_ndx, poses):
    x = poses[:, 0]
    y = poses[:, 1]

    cur_x = x[cur_ndx]
    cur_y = y[cur_ndx]

    fig = plt.figure()
    plt.scatter(y, x, s=1, linewidth=0)
    plt.scatter(cur_y, cur_x, s=30, c='red', linewidth=0)
    plt.axis('equal')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.tight_layout()

    graph_image = fig_to_image(fig)

    plt.close()

    return graph_image


def main(args):
    dataset_root = Path(args.dataset_root)
    assert dataset_root.exists()

    track = args.track
    
    images_dir = dataset_root / 'images' / track / 'lb3_small'
    assert images_dir.exists()

    lidar_dir = dataset_root / 'velodyne_data' / track / 'velodyne_sync'
    assert lidar_dir.exists()

    cam_dirs = [(images_dir / f'Cam{i}') for i in [2, 1, 5, 4, 3]]  # the exact order of cameras
    for cam_dir in cam_dirs:
        assert cam_dir.exists(), f"Camera directory {cam_dir} does not exist"

    dataset_csv = pd.read_csv(dataset_root / 'dataset_index.csv', index_col=0)
    dataset_csv = dataset_csv[dataset_csv['track'] == track]

    poses = dataset_csv[['northing', 'easting']].to_numpy()

    VIDEO_SIZE = (1518, 720)  # little bit smaller size
    FPS = 5  # the frames were recorded at 5Hz (from documentation)
    FOURCC = cv2.VideoWriter_fourcc('M','J','P','G')
    OUT_FILENAME = 'out_video.avi'
    video_writer = cv2.VideoWriter(OUT_FILENAME, FOURCC, FPS, VIDEO_SIZE)

    for i, row in tqdm(dataset_csv.iterrows(), total=len(dataset_csv)):
        img_filename = row['image']
        lidar_filename = row['lidar']

        camera_frame = read_camera_frame(img_filename, cam_dirs)

        lidar_frame = read_lidar_frame(lidar_filename, lidar_dir)

        position_frame = make_position_frame(i, poses)
        position_frame = cv2.resize(position_frame, (533, 400))

        v_space = np.full((400, 333, 3), fill_value=255, dtype=np.uint8)
        bottom_frame = np.concatenate([v_space, lidar_frame, v_space, position_frame, v_space], axis=1)
        
        full_frame = np.concatenate([camera_frame, bottom_frame], axis=0)

        full_frame = cv2.resize(full_frame, VIDEO_SIZE)

        assert full_frame.shape[0] == VIDEO_SIZE[1] and full_frame.shape[1] == VIDEO_SIZE[0], \
            f"frame.shape = {full_frame.shape}, VIDEO_SIZE = {VIDEO_SIZE}"
        video_writer.write(full_frame)
        
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True,
                        help="Dataset root directory")
    parser.add_argument('--track', type=str, default='2012-01-08')
    args = parser.parse_args()
    
    main(args)
