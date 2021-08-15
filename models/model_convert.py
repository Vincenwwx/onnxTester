import tensorflow as tf
import torch
import tf2onnx


def convert_origin_model(model, origin_framework, paths):
    """
    Convert the origin model to models of other frameworks
    :param model:               model object
    :param origin_framework:    origin framework
    :param paths:               path to save model
    :return: None
    """
    assert origin_framework in ["tensorflow", "pytorch", "matlab"], \
        "Unknown framework"

    if origin_framework == "tensorflow":

        spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13,
                                                    output_path=str(paths["saved_models"]))

    elif origin_framework == "pytorch":
        torch.onnx.export(model, dummy_input, "mnist.onnx")

    else:

