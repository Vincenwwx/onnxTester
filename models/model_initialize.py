import os
import pathlib
import tensorflow as tf
import torch
import onnx
import matlab.engine
from data_pipeline.dataset import Dataset_loader
from models.tf_models import tf_initialise_model
from models.torch_model import TorchTest
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Model_Initializer:

    def __init__(self, origin_framework, num_classes, learning_rate, epoch, steps_per_epoch,
                 model_path="", model_name=""):
        self.origin_framework = origin_framework.lower()
        self.num_classes = num_classes
        self.model_path = model_path
        self.model_name = model_name.lower()
        self.model = None
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.steps_per_epoch = steps_per_epoch
        """
        Validation:
            1. framework should be one of "tensorflow", "pytorch" and "matlab"
            2. model name should be one of "vgg16", "inceptionv3" and "resnet50"
        """
        assert self.origin_framework in {"tensorflow", "pytorch", "matlab"}
        if self.model_name:
            assert self.model_name in {"vgg16", "inceptionv3", "resnet50"}

        self._set_model()

    def _set_model(self, momentum=0.9):
        if self.model_path:     # load own trained model

            if self.origin_framework == "tensorflow":
                self.model = tf.keras.models.load_model(self.model_path)

            elif self.origin_framework == "pytorch":
                self.model = torch.load(self.model_path)

            else:
                """
                If the model is build in MATLAB, it should be manually converted to onnx model first
                before being loaded
                """
                self.model = onnx.load(self.model_path)

        elif not self.model_path and self.model_name:       # auto generate model

            if self.origin_framework == "tensorflow":
                self.model = tf_initialise_model(self.model_name)
                dataset_loader = Dataset_loader()
                dataset = dataset_loader.load_dataset(self.origin_framework, "train", model_to_use=self.model_name)

                self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=self.learning_rate, momentum=momentum),
                                   loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
                self.model.summary()
                history = self.model.fit(dataset, epochs=self.epoch, steps_per_epoch=self.steps_per_epoch)

            # Todo: PyTorch
            elif self.origin_framework == "pytorch":
                torch_test = TorchTest(self.num_classes)
                torch_test.initialize_model(self.model_name)
                torch_test.set_optimizer(optimizer="SGD",
                                         learning_rate=self.learning_rate,
                                         momentum=momentum)
                dataset_loader = Dataset_loader()
                dataset_loader.load_dataset(self.origin_framework, "train", model_to_use=self.model_name)
                torch_test.torch_train_model(num_epochs=self.epoch)

            else:
                engine = matlab.engine.start_matlab()
                if self.model_name == "ResNet50":
                    self.model = engine.mat_resNet50()
                elif self.model_name == "InceptionV3":
                    self.model = engine.mat_inceptionV3()
                else:
                    self.model = engine.mat_vgg16()

        else:
            pass

