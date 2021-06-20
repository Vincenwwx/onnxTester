import tensorflow as tf
import torch


class Tester(object):
    def __init__(self, origin_model, converted_models, origin_framework):
        self.origin_framework = origin_framework
        self.origin_model = origin_model
        self.converted_models = converted_models

    def test_models(self):
        if self.origin_framework == "tensorflow":
            pass


def test_tf_model():
    pass


def test_torch_model():
    pass


def test_matlab_model():
    pass