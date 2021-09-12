import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import argparse


def crop_border(x, constant=0.0):
    top = 0
    while True:
        if np.sum(x[top] != constant) != 0.0:
            break
        top += 1
    bottom = x.shape[0] - 1
    while True:
        if np.sum(x[bottom] != constant) != 0.0:
            bottom += 1
            break
        bottom -= 1
    left = 0
    while True:
        if np.sum(x[:, left] != constant) != 0.0:
            break
        left += 1
    right = x.shape[1] - 1
    while True:
        if np.sum(x[:, right] != constant) != 0.0:
            right += 1
            break
        right -= 1
    return x[top:bottom, left:right]


def crop_images(path, constant_value):
    print('Crop image borders...')
    for f in tqdm(os.listdir(path)):
        img = Image.open(os.path.join(path, f))
        img = crop_border(np.array(img), constant=constant_value)
        img = Image.fromarray(img)
        img.save(os.path.join(path, f))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='Path to the image directory.')
    parser.add_argument('--constant_value', type=float, default=0.0, help='Value of the border that should be cropped.')
    
    args = parser.parse_args()

    crop_images(args.image_dir, args.constant_value)

