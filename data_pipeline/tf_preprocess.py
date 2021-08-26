import os
import math
import pathlib
import tensorflow as tf
from typing import Tuple
import numpy as np

NUM_OF_CLASSES = 91


def create_tfrecord(img_path_list, labels, tfrecord_file_name):
    """
    concatenate listed images and respective labels into a TFRecord file
    locating in the new generated folder 'tfrecordData' under the same directory
    :param tfrecord_file_name: (str) specify prefix of tfrecord files
    :param labels: (list) labels
    :param img_path_list: (list) image path
    :return None
    """
    dataset_folder = pathlib.Path(__file__).parent.joinpath("coco_2017")
    # divide data into tfrecord files of around 100MB
    total_size = 0
    for img in img_path_list:
        total_size += os.path.getsize(img)
    average_image_size = math.ceil(total_size / len(img_path_list))
    num_tfrecord_files = math.ceil(total_size / (100 * 1024 * 1000))
    num_img_per_tfrecord_file = math.ceil((100 * 1024 * 1000) / average_image_size)
    dataset_folder.joinpath("tfrecord").mkdir(exist_ok=True)

    for i in range(num_tfrecord_files):
        tfrecord_file_path = dataset_folder.joinpath("tfrecord", "{}_{}.tfrecords".format(tfrecord_file_name, i + 1))
        if not tfrecord_file_path.exists():  # do not re-create
            with tf.io.TFRecordWriter(str(tfrecord_file_path)) as writer:
                for j in range(num_img_per_tfrecord_file):
                    index = i * num_img_per_tfrecord_file + j
                    try:
                        img_path = img_path_list[index]
                    except:
                        break
                    image_data = open(img_path, "rb").read()
                    label = labels[index]

                    tf_example = tf.train.Example(features=tf.train.Features(feature={
                        "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
                        "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=label))
                    }))
                    writer.write(tf_example.SerializeToString())

            print("System creates {}/{} tfrecord file.".format(i + 1, num_tfrecord_files))
    print("All images have been written into TFRecord files.")


def read_tfrecord(dataset):
    """
    read and parse tfrecord file
    :param dataset: "val" or "train"
    :return: tf.Dataset object
    """

    def parse_image_function(proto):
        proto = tf.io.parse_single_example(proto,
                                           features={
                                               'image_raw': tf.io.FixedLenFeature([], tf.string),
                                               'labels': tf.io.VarLenFeature(tf.int64)})
        return proto

    path = pathlib.Path(__file__).parent.joinpath("coco_2017", "tfrecord", dataset + "_*.tfrecords")
    raw_dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(str(path)))
    parsed_dataset = raw_dataset.map(parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return parsed_dataset


def prepare_for_inceptionv3(dataset):
    origin_label = dataset["labels"]
    """
        Modify the label to multi-one-hot coding, for example
            [2, 5] -> [0, 0, 1, 0, 0, 1, 0, 0, 0]
    """
    label = tf.zeros(NUM_OF_CLASSES)  # there are 90 categories in COCO 2017
    for idx in origin_label.values:
        label += tf.one_hot(indices=idx, depth=NUM_OF_CLASSES)
    """
        Modify images with regard to different models
    """
    image = tf.image.decode_jpeg(dataset["image_raw"], channels=3)
    image = preprocess_single_img(image, 299)
    image = tf.keras.applications.inception_v3.preprocess_input(image)

    return image, label


def prepare_for_vgg16(dataset):
    origin_label = dataset["labels"]
    label = tf.zeros(NUM_OF_CLASSES)  # there are 90 categories in COCO 2017
    for idx in origin_label.values:
        label += tf.one_hot(indices=idx, depth=NUM_OF_CLASSES)

    image = tf.image.decode_jpeg(dataset["image_raw"], channels=3)
    image = preprocess_single_img(image, 224)
    image = tf.keras.applications.vgg16.preprocess_input(image)

    return image, label


def prepare_for_resnet50(dataset):
    origin_label = dataset["labels"]
    label = tf.zeros(NUM_OF_CLASSES)  # there are 90 categories in COCO 2017
    for idx in origin_label.values:
        label += tf.one_hot(indices=idx, depth=NUM_OF_CLASSES)

    image = tf.image.decode_jpeg(dataset["image_raw"], channels=3)
    image = preprocess_single_img(image, 224)
    image = tf.keras.applications.resnet.preprocess_input(image)

    return image, label


def preprocess_single_img(image, size):
    image = tf.image.resize(image, (size, size))
    image = tf.cast(image, tf.float32) / 255.
    image = mean_image_subtraction(image, (0.485, 0.456, 0.406))
    image = standardize_image(image, (0.229, 0.224, 0.225))
    return image

"""
    reference: https://github.com/tensorflow/models/blob/master/official/vision/image_classification/preprocessing.py
"""


def mean_image_subtraction(image_bytes: tf.Tensor,
                           means: Tuple[float, ...],
                           num_channels: int = 3,
                           dtype: tf.dtypes.DType = tf.float32, ) -> tf.Tensor:
    """ Subtracts the given means from each image channel.

    For example:
        means = (123.68, 116.779, 103.939)
        image_bytes = mean_image_subtraction(image_bytes, means)

    Note that the rank of `image` must be known.

    Args:
        image_bytes: a tensor of size [height, width, C].
        means: a C-vector of values to subtract from each channel.
        num_channels: number of color channels in the image that will be distorted.
        dtype: the dtype to convert the images to. Set to `None` to skip conversion.

    Returns:
        the centered image.

    Raises:
        ValueError: If the rank of `image` is unknown, if `image` has a rank other
          than three or if the number of channels in `image` doesn't match the
          number of values in `means`.
    """
    if image_bytes.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    means = tf.broadcast_to(means, tf.shape(image_bytes))
    if dtype is not None:
        means = tf.cast(means, dtype=dtype)

    return image_bytes - means


def standardize_image(image_bytes: tf.Tensor,
                      stddev: Tuple[float, ...],
                      num_channels: int = 3,
                      dtype: tf.dtypes.DType = tf.float32, ) -> tf.Tensor:
    """ Divides the given stddev from each image channel.

    For example:
        stddev = (123.68, 116.779, 103.939)
        image_bytes = standardize_image(image_bytes, stddev)

    Note that the rank of `image` must be known.

    Args:
        image_bytes: a tensor of size [height, width, C].
        stddev: a C-vector of values to divide from each channel.
        num_channels: number of color channels in the image that will be distorted.
        dtype: the dtype to convert the images to. Set to `None` to skip conversion.

    Returns:
        the centered image.

    Raises:
        ValueError: If the rank of `image` is unknown, if `image` has a rank other
          than three or if the number of channels in `image` doesn't match the
          number of values in `stddev`.
    """
    if image_bytes.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    if len(stddev) != num_channels:
        raise ValueError('len(stddev) must match the number of channels')

    stddev = tf.broadcast_to(stddev, tf.shape(image_bytes))
    if dtype is not None:
        stddev = tf.cast(stddev, dtype=dtype)

    return image_bytes / stddev


def tf_load_and_preprocess_single_img(img_path: str, size: int) -> np.ndarray:
    """Load an image as numpy array and pre-process

    Preprocess include:
        - resize to `size`
        - rescale to [0-1]
        - zero center
        - normalization

    Args:
        img_path (str): specify image path
        size (int): specify size of final image

    Return:
        np.ndarray: numpy array of image
    """

    image = tf.keras.preprocessing.image.load_img(img_path,
                                                  target_size=(size, size))
    image = tf.keras.preprocessing.image.img_to_array(image)
    # rescale to 0 - 1
    image = tf.cast(image, tf.float32) / 255.0
    # normalization
    mean = np.array([0.485, 0.456, 0.406])
    dev = np.array([0.229, 0.224, 0.225])
    image = image - mean.reshape((1, 1, 3))
    image = image / dev.reshape((1, 1, 3))
    # convert single image to a batch
    input_arr = np.array([image])

    return input_arr
