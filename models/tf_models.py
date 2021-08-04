import tensorflow as tf
import logging


def tf_initialise_model(model_name, n_classes=90):

    if model_name == "resnet50":
        logging.info("Now initialize resNet-50 model")
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
        base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet',
                                                             input_tensor=inputs)
        base_model.trainable = False

        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        output = tf.keras.layers.Dense(n_classes)(x)
        model = tf.keras.Model(inputs=inputs, outputs=output, name='tl_resNet50')

    elif model_name == "inceptionv3":
        logging.info("Now initialize inception-V3 model")
        inputs = tf.keras.layers.Input(shape=(299, 299, 3))
        base_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                                                    input_tensor=inputs)
        base_model.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        output = tf.keras.layers.Dense(n_classes)(x)
        model = tf.keras.Model(inputs=inputs, outputs=output, name='tl_inceptionv3')

    else:
        logging.info("Now initialize VGG-16 model")
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
        base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
        base_model.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        output = tf.keras.layers.Dense(n_classes)(x)
        model = tf.keras.Model(inputs=inputs, outputs=output, name='tl_vgg16')

    return model
