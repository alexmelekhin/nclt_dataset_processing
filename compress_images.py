import os
import os.path as osp
import argparse
import re

import numpy as np
import cv2
from tqdm import tqdm


class Undistort(object):
    """
    Source: http://robots.engin.umich.edu/nclt/
    Nicholas Carlevaris-Bianco, Arash K. Ushani, and Ryan M. Eustice,
        University of Michigan North Campus Long-Term Vision and Lidar Dataset,
        International Journal of Robotics Research, 2016.
    """

    def __init__(self, fin):
        self.fin = fin
        # read in distort
        with open(fin, 'r') as f:
            #chunks = f.readline().rstrip().split(' ')
            header = f.readline().rstrip()
            chunks = re.sub(r'[^0-9,]', '', header).split(',')
            self.mapu = np.zeros((int(chunks[1]),int(chunks[0])),
                    dtype=np.float32)
            self.mapv = np.zeros((int(chunks[1]),int(chunks[0])),
                    dtype=np.float32)
            for line in f.readlines():
                chunks = line.rstrip().split(' ')
                self.mapu[int(chunks[0]),int(chunks[1])] = float(chunks[3])
                self.mapv[int(chunks[0]),int(chunks[1])] = float(chunks[2])
        # generate a mask
        self.mask = np.ones(self.mapu.shape, dtype=np.uint8)
        self.mask = cv2.remap(self.mask, self.mapu, self.mapv, cv2.INTER_LINEAR)
        kernel = np.ones((30,30),np.uint8)
        self.mask = cv2.erode(self.mask, kernel, iterations=1)

    """
    Use OpenCV to undistorted the given image
    """
    def undistort(self, img):
        return cv2.resize(cv2.remap(img, self.mapu, self.mapv, cv2.INTER_LINEAR),
                          (self.mask.shape[1], self.mask.shape[0]),
                          interpolation=cv2.INTER_CUBIC)


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


def rotate_crop_resize(img, resize_dim=(384, 512)):
    im_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    im_cropped = center_crop(im_rotated, (810, 1080))
    im_resized = cv2.resize(im_cropped, resize_dim)
    return im_resized


def main():
    parser = argparse.ArgumentParser(description="Undistort images")
    parser.add_argument('--images_dir', type=str, help='images directory')
    parser.add_argument('--maps_dir',type=str, help='undistortion maps directory')

    args = parser.parse_args()

    maps_filenames = os.listdir(args.maps_dir)
    maps_filenames = [m for m in maps_filenames if m.startswith('U2D') and m.endswith('.txt')]
    maps_filenames = sorted(maps_filenames)

    assert len(maps_filenames) == 6, 'Not all undistortion maps present'

    undistorts = [Undistort(osp.join(args.maps_dir, fname)) for fname in maps_filenames]
    cams = [f'Cam{i}' for i in range(6)]

    tracks = os.listdir(args.images_dir)
    tracks = [t for t in tracks  # filter out dirs that are not correct track names
              if re.match(r"^201[2-3]-((0[1-9])|(1[0-2]))-(0[1-9]|[1-2][0-9]|3[0-1])$", t)]
    tracks = sorted(tracks)
    print(f"Found image directories for tracks: {tracks}")

    for track in tracks:
        in_path = osp.join(args.images_dir, track, 'lb3')
        out_path = osp.join(args.images_dir, track, 'lb3_small')

        print(f"\nProccessing track {track}:")

        for cam, undist in zip(cams, undistorts):
            if cam == 'Cam0':
                print(f"Cam0: skipping camera that looks in the sky")
                continue
            os.makedirs(osp.join(out_path, cam), exist_ok=True)
            im_dir = osp.join(in_path, cam)
            im_names = sorted(os.listdir(im_dir))
            for im_name in tqdm(im_names, desc=cam):
                im = cv2.imread(osp.join(im_dir, im_name))
                im_undistorted = undist.undistort(im)
                im_out = rotate_crop_resize(im_undistorted)
                im_out_filename = osp.join(out_path, cam, im_name.split('.')[0]+'.png')
                cv2.imwrite(im_out_filename, im_out)


if __name__ == "__main__":
    main()
