import os
import math
import pathlib
import tensorflow as tf


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
    label = tf.zeros(90)  # there are 90 categories in COCO 2017
    for idx in origin_label.values:
        label += tf.one_hot(indices=idx, depth=90)
    """
        Modify images with regard to different models
    """
    image = tf.image.decode_jpeg(dataset["image_raw"], channels=3)
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.inception_v3.preprocess_input(image)

    return image, label


def prepare_for_vgg16(dataset):
    origin_label = dataset["labels"]
    label = tf.zeros(90)  # there are 90 categories in COCO 2017
    for idx in origin_label.values:
        label += tf.one_hot(indices=idx, depth=90)

    image = tf.image.decode_jpeg(dataset["image_raw"], channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.vgg16.preprocess_input(image)

    return image, label


def prepare_for_resnet50(dataset):
    origin_label = dataset["labels"]
    label = tf.zeros(90)  # there are 90 categories in COCO 2017
    for idx in origin_label.values:
        label += tf.one_hot(indices=idx, depth=90)

    image = tf.image.decode_jpeg(dataset["image_raw"], channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.resnet.preprocess_input(image)

    return image, label