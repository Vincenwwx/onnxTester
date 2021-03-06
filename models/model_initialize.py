import pathlib
import logging
import gin
import tensorflow as tf
import torch
import onnx
import matlab.engine
from data_pipeline.dataset import Dataset_loader
from models.tf_models import tf_initialise_model
from models.torch_model import Torch_Test
import os
import tf2onnx

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@gin.configurable
class Model_Initializer:

    def __init__(self, origin_framework, paths, num_classes, learning_rate=0.001,
                 epoch=10, steps_per_epoch=30, model_path="", model_name="", momentum=0.9):
        self.origin_framework = origin_framework.lower()
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.model_path = model_path
        self.model = None
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.steps_per_epoch = steps_per_epoch
        self.momentum = momentum     # hard-code momentum for SGD
        self.paths = paths
        """
        Validation:
            1. framework should be one of "tensorflow", "pytorch" and "matlab"
            2. model name should be one of "vgg16", "inceptionv3" and "resnet50"
        """
        assert self.origin_framework in {"tensorflow", "pytorch", "matlab"}, \
            "please specify framework to be one of 'TensorFlow', 'PyTorch' and 'MATLAB'"
        if self.model_name:
            assert self.model_name in {"vgg16", "inceptionv3", "resnet50"}, \
                "Please specify model to be one of 'VGG16', InceptionV3' and 'ResNet50'"

        self._set_and_save_model()

    def _set_and_save_model(self):

        if self.model_path:     # load own trained model

            logging.info("[System] Load user owned model...")

            if self.origin_framework == "tensorflow":
                self.model = tf.keras.models.load_model(self.model_path)

            elif self.origin_framework == "pytorch":
                self.model = torch.load(self.model_path)

            else:
                """
                load MATLAB model saved in .onnx format
                """
                self.model = onnx.load(self.model_path)

        else:       # auto generate model

            logging.info("[System] Auto generate {} model in {}...".format(self.model_name, self.origin_framework))

            if self.origin_framework == "tensorflow":

                dataset_loader = Dataset_loader()
                dataset = dataset_loader.load_dataset(framework=self.origin_framework,
                                                      dataset="train", model_to_use=self.model_name)
                self.model = tf_initialise_model(model_name=self.model_name,
                                                 n_classes=self.num_classes)
                self.model.compile(
                    optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum),
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
                self.model.summary()
                self.model.fit(dataset, epochs=self.epoch, steps_per_epoch=self.steps_per_epoch)
                # save tf model
                save_model_path = str(self.paths["saved_models"].joinpath("origin_{}_{}".format(self.origin_framework,
                                                                                                self.model_name), "1"))
                self.model.save(save_model_path)
                logging.info("[System] Model is saved to: {}".format(save_model_path))

            elif self.origin_framework == "pytorch":
                # get dataset loader
                dataset_loader = Dataset_loader()
                dataset = dataset_loader.load_dataset(framework=self.origin_framework, dataset="train",
                                                      model_to_use=self.model_name)
                # get torch model
                torch_test = Torch_Test(num_classes=self.num_classes)
                torch_test.initialize_model(model_name=self.model_name, use_pretrained=True)
                torch_test.set_optimizer(optimizer="SGD", learning_rate=self.learning_rate,
                                         momentum=self.momentum)
                # transfer learning
                torch_test.torch_train_model(dataset, num_epochs=self.epoch, steps_per_epoch=self.steps_per_epoch)
                self.model = torch_test.model_ft
                # save torch model
                save_model_path = self.paths["saved_models"].joinpath("origin_{}_{}.pt".format(self.origin_framework,
                                                                                               self.model_name))
                torch.save(self.model, str(save_model_path))
                logging.info("[System] Model is saved to: {}".format(save_model_path))

            else:

                # Init model using MATLAB engine
                eng = matlab.engine.start_matlab()
                info = {
                    "modelName": self.model_name,
                    "dataRoot": str(self.paths["coco_dataset"]),
                    "savePath": str(self.paths["saved_models"])
                }
                eng.addpath(str(pathlib.Path(__file__).parent))     # Add current path to MATLAB working path
                save_model_path = eng.init_and_export_matlab_model(info)
                logging.info("[System] Model is saved to: {}".format(str(save_model_path)))

    def export_model_to_onnx(self):
        """Export trained model to onnx format.

        The onnx model will be saved in `run_path["saved_models"]` with the name of format
            origin_{origin framework}_{model name}.onnx

        Example:
            "origin_tensorflow_vgg16.onnx"
        """
        logging.info("[System] Now export {} model of framework {} to .onnx ...".format(self.model_name,
                                                                                        self.origin_framework))
        output_path = str(self.paths["saved_models"].joinpath("origin_{}_{}.onnx".format(self.origin_framework,
                                                                                         self.model_name)))

        if self.origin_framework == "tensorflow":

            if self.model_name == "resnet50" or self.model_name == "vgg16":
                spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
            else:
                spec = (tf.TensorSpec((None, 299, 299, 3), tf.float32, name="input"),)
            model_proto, _ = tf2onnx.convert.from_keras(self.model, input_signature=spec,
                                                        opset=8, output_path=output_path)

        elif self.origin_framework == "pytorch":

            if self.model_name == "resnet50" or self.model_name == "vgg16":
                dummy_input = torch.randn(1, 3, 224, 224)
            else:
                dummy_input = torch.randn(1, 3, 299, 299)
            # Invoke export
            torch.onnx.export(self.model, dummy_input, output_path)

        logging.info("[System] {} model has been exported as .onnx file.".format(self.model_name))
