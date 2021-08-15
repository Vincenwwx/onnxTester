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

    def __init__(self, origin_framework, num_classes, learning_rate, epoch, paths,
                 steps_per_epoch, model_path="", model_name=""):
        self.origin_framework = origin_framework.lower()
        self.num_classes = num_classes
        self.model_path = model_path
        self.model_name = model_name.lower()
        self.model = None
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.steps_per_epoch = steps_per_epoch
        self.momentum = 0.9     # hard-code momentum for SGD
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

                self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=self.learning_rate, momentum=self.momentum),
                                   loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
                self.model.summary()
                self.model.fit(dataset, epochs=self.epoch, steps_per_epoch=self.steps_per_epoch)

                self.model.save(self.paths["saved_models"])
                self.save_model_to_onnx()

            elif self.origin_framework == "pytorch":
                torch_test = Torch_Test(num_classes=self.num_classes,
                                        feature_extract=True)
                torch_test.initialize_model(model_name=self.model_name,
                                            use_pretrained=True)
                torch_test.set_optimizer(optimizer="SGD",
                                         learning_rate=self.learning_rate,
                                         momentum=self.momentum)
                print("Now initialise # {} # model under framework {}".format(self.model_name,
                                                                              self.origin_framework))
                dataset_loader = Dataset_loader()
                dataset = dataset_loader.load_dataset(framework=self.origin_framework,
                                                      dataset="train",
                                                      model_to_use=self.model_name)
                torch_test.torch_train_model(dataset, num_epochs=self.epoch)
                self.model = torch_test.model_ft

                torch.save(str(self.paths["saved_models"]))
                self.save_model_to_onnx()

            else:

                engine = matlab.engine.start_matlab()
                engine.init_and_output_matlab_model(self.model_name,
                                                    self.paths["coco_dataset"],
                                                    self.paths["saved_models"])

    def save_model_to_onnx(self):

        print("[System] Now export {} model of framework {} to .onnx ...".format(self.model_name, self.origin_framework))

        if self.origin_framework == "tensorflow":

            if self.model_name == "resnet50" or self.model_name == "vgg16":
                spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
            else:
                spec = (tf.TensorSpec((None, 299, 299, 3), tf.float32, name="input"),)
            output_path = str(self.paths["saved_models"].joinpath(self.model_name + ".onnx"))
            model_proto, _ = tf2onnx.convert.from_keras(self.model, input_signature=spec, opset=13, output_path=output_path)

        elif self.origin_framework == "pytorch":
            pass
