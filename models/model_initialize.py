import os
import gin
#import matlab.engine
from models.tf_models import get_resNet152, get_inceptionV3


@gin.configurable
class ModelInitializer:
    def __init__(self, origin_framework, dataset_name, model_path="", model_name=""):
        self.model_path = model_path
        self.model_name = model_name
        self.origin_framework = origin_framework
        self.dataset_name = dataset_name
        # validate the chosen framework
        assert self.origin_framework in {"tensorflow", "pytorch", "matlab"}

    def get_model(self):
        # Load local DL model
        if self.model_path and not self.model_name:
            origin_model = load_model(self.model_path, self.origin_framework)

        # generate DL model
        # transfer learning should be done here
        elif not self.model_path and self.model_name:
            origin_model = generate_model(self.model_name, self.origin_framework)
            dataset = load_dataset(self.dataset_name, self.origin_framework)
            train_model(origin_model, dataset, self.origin_framework)

        else:
            raise Exception("Model initialization failed.")
        return origin_model


def train_model(model, dataset, framework):
    if framework == "tensorflow":
        pass
    elif framework == "pytorch":
        pass
    else:
        pass


def generate_model(model_name, framework):
    """ Auto generate DL models
    Parameters:
        @model_name     string, can be "ResNet 152" or "Inception V4"
        @framework      string, can be "tensorflow", "pytorch" or "matlab"
    Returns:
        model object
    """
    if framework == "tensorflow":
        if model_name == "ResNet 152":
            model = tf.keras.applications.ResNet152(
                        include_top=True, weights='imagenet', input_tensor=None,
                        input_shape=None, pooling=None, classes=1000, **kwargs)
        elif model_name == "Inception V4":
            pass
        else:
            print("Please specify a model in the list.")
            raise
    elif framework == "pytorch":
        if model_name == "ResNet V2":
            pass
        elif model_name == "Inception V4":
            pass
        else:
            print("Please specify a model in the list.")
            raise
    elif framework == "matlab":
        if model_name == "ResNet V2":
            pass
        elif model_name == "Inception V4":
            pass
        else:
            print("Please specify a model in the list.")
            raise
    else:
        print("Invalid framework! Please choose one from tensorflow, pytorch and matlab.")
        raise


def load_model(path, framework):
    """ Load DL model
    Parameters:
        @path        string, path of DL model file
        @framework   string, can be "tensorflow", "pytorch" or "matlab"
    Returns:
        model object
    """
    if framework == "tensorflow":
        return tf.keras.models.load_model(path)
    elif framework == "pytorch":
        return torch.load(path)
    # To-Do: load saved matlab DL model file
    elif framework == "matlab":
        pass
    else:
        print("Invalid framework! Please choose one from tensorflow, pytorch and matlab.")
        raise


def load_dataset(dataset_name, framework):
    pass