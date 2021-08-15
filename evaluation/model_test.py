import tensorflow as tf
import torch
from data_pipeline.dataset import Dataset_loader


class Performance_Tester(object):
    def __init__(self, origin_model, origin_framework, onnx_path):
        self.origin_framework = origin_framework
        self.origin_model = origin_model
        self.onnx_path = onnx_path

    def test_models(self):
        if self.origin_framework == "tensorflow":
            """ Todo:
            TensorFlow. Steps:
            1. init test dataset
            2. get models of another two frameworks
            3. inference and compare
            """
            pass

        elif self.origin_framework == "pytorch":
            """
            PyTorch. Steps:
            1. init test dataset
            2. get models of another two frameworks
            3. inference and compare
            """
            pass

        else:
            """
            MATLAB. Steps:
            1. get models of another two frameworks
            3. inference and compare
            """
            pass


def test_tf_model():
    pass


def test_torch_model():
    pass


def test_matlab_model():
    pass