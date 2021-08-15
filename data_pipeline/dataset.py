import json
import time
import pathlib
import numpy as np
import tensorflow as tf
from collections import defaultdict
from data_pipeline.tf_preprocess import create_tfrecord, read_tfrecord, prepare_for_resnet50, prepare_for_inceptionv3, \
    prepare_for_vgg16
from data_pipeline.torch_datareader import Torch_DataReader
import torch
from torchvision import transforms


class Dataset_loader:
    """
    Folder structure:
        - coco_2017
            - images
                - train
                - val
            - annotations
            - tfrecords
    """

    def __init__(self, batch=64, caching=True):
        # Todo: coco dataset
        self.dataset_folder = pathlib.Path(__file__).parent.joinpath("coco_2017")
        self.dataset_folder.mkdir(exist_ok=True)
        self.batch = batch
        self.caching = caching

        # download annotations
        self.download_coco2017_dataset()
        self.image_id_to_categories = defaultdict(list)

    def load_dataset(self, framework, dataset, model_to_use,
                     num_images_to_choose_in_train=20000):
        """
        get train/val dataset for specific framework
        :param model_to_use: (str) specify for which model the dataset will be used
        :param framework: (str) specific which DL framework
        :param dataset: (str) one of "train", "val"
        :param num_images_to_choose_in_train: (int) number of images to choose
                                                    if use train dataset
        :return: dataset object
        """
        framework = framework.lower()
        assert framework in ["tensorflow", "pytorch"], "Supported frameworks: tensorflow, pytorch"
        assert dataset in ["train", "val"], "Please choose train or val dataset"

        annotation_file = self.dataset_folder.joinpath("annotations", "instances_" + dataset + "2017.json")
        with open(annotation_file, 'r') as f:
            tic = time.time()
            print("[System] Load annotation file...")
            annotation = json.load(f)
            print('[System] Done (t={:0.2f}s)'.format(time.time() - tic))

        # create dictionaries / mappings
        for ann in annotation["annotations"]:
            image_id = ann["image_id"]
            self.image_id_to_categories[str(image_id)].append(ann["category_id"])

        self.download_coco2017_dataset(dataset=dataset)
        img_id_list = list(self.image_id_to_categories.keys())
        if dataset == "train":
            np.random.shuffle(img_id_list)
            img_id_list = img_id_list[:num_images_to_choose_in_train]
            # delete the rest images
            # for img_id in img_id_list[num_images_to_choose_in_train:]:
            #     self.dataset_folder.joinpath("images", dataset,
            #                                  "{:012d}.jpg".format(int(img_id))).unlink(missing_ok=True)

        img_path_list = [self.dataset_folder.joinpath("images", dataset, "{:012d}.jpg".format(int(img_id)))
                         for img_id in img_id_list]
        labels = [list(set(self.image_id_to_categories[img_id])) for img_id in img_id_list]

        if framework == "tensorflow":

            create_tfrecord(img_path_list, labels, tfrecord_file_name=dataset)
            ds_raw = read_tfrecord(dataset)

            if model_to_use == "inceptionv3":
                ds = ds_raw.map(prepare_for_inceptionv3, tf.data.experimental.AUTOTUNE)
            elif model_to_use == "resnet50":
                ds = ds_raw.map(prepare_for_resnet50, tf.data.experimental.AUTOTUNE)
            else:
                ds = ds_raw.map(prepare_for_vgg16, tf.data.experimental.AUTOTUNE)
            if self.caching:
                ds = ds.cache()
            ds = ds.batch(self.batch)
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE).repeat()

            return ds

        elif framework == "pytorch":

            size = 299 if model_to_use == "inceptionv3" else 224
            composed_transforms = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            torch_dataReader = Torch_DataReader(img_path_list, labels, transform=composed_transforms)

            return torch.utils.data.DataLoader(torch_dataReader, batch_size=64,
                                               shuffle=True, num_workers=6)

    def download_coco2017_dataset(self, dataset=None):
        """
        download COCO 2017 dataset
        :param dataset: (str) can be "val" or "train"
        :param num:
        :return:
        """
        image_folder = self.dataset_folder.joinpath("images")
        annotation_folder = self.dataset_folder.joinpath("annotations")

        if dataset:
            assert dataset in ["val", "train"], "Only validation and train dataset can be download"
            if not image_folder.joinpath(dataset).exists():
                tic = time.time()
                print("[System] Now download {} dataset...".format(dataset))
                image_zip = tf.keras.utils.get_file(str(image_folder.joinpath(dataset + ".zip")),
                                                    cache_subdir=image_folder,
                                                    origin="http://images.cocodataset.org/zips/" + dataset + "2017.zip",
                                                    extract=True)
                pathlib.Path(image_zip).unlink()
                rename_folder = image_folder.joinpath(dataset)
                image_folder.joinpath(dataset + "2017").rename(rename_folder)
                print("[System] Finish download {} dataset.(t={})".format(dataset, tic - time.time()))
            else:
                print("[System] % {} % dataset already exists".format(dataset))

        if not annotation_folder.exists():
            tic = time.time()
            print("[System] Now download annotations...")
            annotation_zip = tf.keras.utils.get_file(str(annotation_folder.parent) + "/annotations.zip",
                                                     cache_subdir=annotation_folder.parent,
                                                     origin="http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                                                     extract=True)
            pathlib.Path(annotation_zip).unlink()
            print("[System] Finish download annotations.(t={})".format(tic - time.time()))
