import tensorflow as tf
import numpy as np
from PIL import Image
from typing import Sequence
from tqdm import tqdm
import argparse
import json
import os


def images_to_tfrecords(image_dir, data_dir, has_labels):
    """
    Converts a folder of images to a TFRecord file.

    The image directory should have one of the following structures:
    
    If has_labels = False, image_dir should look like this:

    path/to/image_dir/
        0.jpg
        1.jpg
        2.jpg
        4.jpg
        ...


    If has_labels = True, image_dir should look like this:

    path/to/image_dir/
        label0/
            0.jpg
            1.jpg
            ...
        label1/
            a.jpg
            b.jpg
            c.jpg
            ...
        ...
    

    The labels will be label0 -> 0, label1 -> 1.

    Args:
        image_dir (str): Path to images.
        data_dir (str): Path where the TFrecords dataset is stored.
        has_labels (bool): If True, 'image_dir' contains label directories.

    Returns:
        (dict): Dataset info.
    """
    
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    os.makedirs(data_dir, exist_ok=True)
    writer = tf.io.TFRecordWriter(os.path.join(data_dir, 'dataset.tfrecords'))

    num_examples = 0
    num_classes = 0

    if has_labels:
        for label_dir in os.listdir(image_dir):
            if not os.path.isdir(os.path.join(image_dir, label_dir)):
                print('The image directory should contain one directory for each label.')
                print('These label directories should contain the image files.')
                if os.path.exists(os.path.join(data_dir, 'dataset.tfrecords')):
                    os.remove(os.path.join(data_dir, 'dataset.tfrecords'))
                return
            
            for img_file in tqdm(os.listdir(os.path.join(image_dir, label_dir))):
                file_format = img_file[img_file.rfind('.') + 1:]
                if file_format not in ['png', 'jpg', 'jpeg']:
                    continue

                #img = Image.open(os.path.join(image_dir, label_dir, img_file)).resize(img_size)
                img = Image.open(os.path.join(image_dir, label_dir, img_file))
                img = np.array(img, dtype=np.uint8)

                height = img.shape[0]
                width = img.shape[1]
                channels = img.shape[2]

                img_encoded = img.tobytes()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'channels': _int64_feature(channels),
                    'image': _bytes_feature(img_encoded),
                    'label': _int64_feature(num_classes)}))

                writer.write(example.SerializeToString())
                num_examples += 1

            num_classes += 1
    else:
        for img_file in tqdm(os.listdir(os.path.join(image_dir))):
            file_format = img_file[img_file.rfind('.') + 1:]
            if file_format not in ['png', 'jpg', 'jpeg']:
                continue

            #img = Image.open(os.path.join(image_dir, label_dir, img_file)).resize(img_size)
            img = Image.open(os.path.join(image_dir, img_file))
            img = np.array(img, dtype=np.uint8)

            height = img.shape[0]
            width = img.shape[1]
            channels = img.shape[2]

            img_encoded = img.tobytes()

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channels': _int64_feature(channels),
                'image': _bytes_feature(img_encoded),
                'label': _int64_feature(num_classes)})) # dummy label

            writer.write(example.SerializeToString())
            num_examples += 1

    writer.close()

    dataset_info = {'num_examples': num_examples, 'num_classes': num_classes}
    with open(os.path.join(data_dir, 'dataset_info.json'), 'w') as fout:
        json.dump(dataset_info, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='Path to the image directory.')
    parser.add_argument('--data_dir', type=str, help='Path where the TFRecords dataset is stored.')
    parser.add_argument('--has_labels', action='store_true', help='If True, image_dir contains label directories.')
    
    args = parser.parse_args()

    images_to_tfrecords(args.image_dir, args.data_dir, args.has_labels)
    
