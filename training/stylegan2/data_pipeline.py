import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import flax
import numpy as np
from PIL import Image
import os
from typing import Sequence
from tqdm import tqdm
import json
from tqdm import tqdm


def prefetch(dataset, n_prefetch):
    # Taken from: https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py
    ds_iter = iter(dataset)
    ds_iter = map(lambda x: jax.tree_map(lambda t: np.asarray(memoryview(t)), x),
                  ds_iter)
    if n_prefetch:
        ds_iter = flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)
    return ds_iter


def get_data(data_dir, img_size, img_channels, num_classes, num_devices, batch_size, shuffle_buffer=1000):
    """

    Args:
        data_dir (str): Root directory of the dataset.
        img_size (int): Image size for training.
        img_channels (int): Number of image channels.
        num_classes (int): Number of classes, 0 for no classes.
        num_devices (int): Number of devices.
        batch_size (int): Batch size (per device).
        shuffle_buffer (int): Buffer used for shuffling the dataset.

    Returns:
        (tf.data.Dataset): Dataset.
    """

    def pre_process(serialized_example):
        feature = {'height': tf.io.FixedLenFeature([], tf.int64),
                   'width': tf.io.FixedLenFeature([], tf.int64),
                   'channels': tf.io.FixedLenFeature([], tf.int64),
                   'image': tf.io.FixedLenFeature([], tf.string),
                   'label': tf.io.FixedLenFeature([], tf.int64)}
        example = tf.io.parse_single_example(serialized_example, feature)

        height = tf.cast(example['height'], dtype=tf.int64)
        width = tf.cast(example['width'], dtype=tf.int64)
        channels = tf.cast(example['channels'], dtype=tf.int64)

        image = tf.io.decode_raw(example['image'], out_type=tf.uint8)
        image = tf.reshape(image, shape=[height, width, channels])

        image = tf.cast(image, dtype='float32')
        image = tf.image.resize(image, size=[img_size, img_size], method='bicubic', antialias=True)
        image = tf.image.random_flip_left_right(image)
        
        image = (image - 127.5) / 127.5
        
        label = tf.one_hot(example['label'], num_classes)
        return {'image': image, 'label': label}

    def shard(data):
        # Reshape images from [num_devices * batch_size, H, W, C] to [num_devices, batch_size, H, W, C]
        # because the first dimension will be mapped across devices using jax.pmap
        data['image'] = tf.reshape(data['image'], [num_devices, -1, img_size, img_size, img_channels])
        data['label'] = tf.reshape(data['label'], [num_devices, -1, num_classes])
        return data

    print('Loading TFRecord...')
    with open(os.path.join(data_dir, 'dataset_info.json'), 'r') as fin:
        dataset_info = json.load(fin)

    ds = tf.data.TFRecordDataset(filenames=os.path.join(data_dir, 'dataset.tfrecords'))
    
    ds = ds.shuffle(min(dataset_info['num_examples'], shuffle_buffer))
    ds = ds.map(pre_process, tf.data.AUTOTUNE)
    ds = ds.batch(batch_size * num_devices, drop_remainder=True)
    ds = ds.map(shard, tf.data.AUTOTUNE)
    ds = ds.prefetch(1)
    return ds, dataset_info



