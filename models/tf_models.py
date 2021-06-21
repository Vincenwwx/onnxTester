import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


def get_resNet152(input_shape, n_classes):
    inputs = tf.keras.layers.Input(input_shape)
    resNet152 = tf.keras.applications.ResNet152(weights="imagenet",
                                                include_top=False,
                                                input_tensor=inputs,
                                                )
    resNet152.trainable = False
    x = resNet152.output
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    output = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, output=output, name="tf_resNet152")

    return model


def get_inceptionV3():
    return tf.keras.applications.InceptionV3()
