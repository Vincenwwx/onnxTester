import numpy as np
import tensorflow as tf
import torch
import onnx
from onnx_tf.backend import prepare
import matlab.engine
import pathlib
import time
import onnxruntime


class Performance_Tester(object):
    def __init__(self, model_name, origin_framework, paths, model_object=None, top_k=5):
        self.origin_framework = origin_framework.lower()
        self.model_name = model_name.lower()
        self.model_object = model_object
        self.top_k = top_k
        self.paths = paths

        if model_name == "inceptionv3":
            self.size = 299
            self.is_inception = True
        else:
            self.size = 224
            self.is_inception = False

    def test_models(self):

        onnx_path = self.paths["saved_models"].joinpath("origin_{}_{}.onnx".format(self.origin_framework,
                                                                                   self.model_name))
        # Todo: change folder back
        dataset_path = self.paths["coco_dataset"].joinpath("images", "temp")
        imgs_path = dataset_path.glob("*.jpg")

        if self.origin_framework == "tensorflow":

            acc_tf_mat = 0
            acc_tf_ort = 0
            tf_test_time = 0
            ort_test_time = 0
            sess = onnxruntime.InferenceSession(onnx_path) # init onnxruntime session

            filename_list, matlab_preds, matlab_avg_time = test_model_in_matlab(onnx_path=str(onnx_path),
                                                                                dataset_path=str(dataset_path),
                                                                                isInception=self.is_inception,
                                                                                topK=self.top_k)
            num_files = len(filename_list)
            for i in range(num_files):
                mat_pred = np.array(matlab_preds[i]) - 1    # indexing issue
                mat_pred = np.sort(np.array(mat_pred.reshape((self.top_k))))
                image = load_and_preprocess_single_img(filename_list[i], size=self.size)
                # predict with tf model and compare with matlab
                tf_start_time = time.time()
                tf_predictions = self.model_object.predict(image)
                tf_test_time += (time.time() - tf_start_time)
                tf_top = np.sort(np.argpartition(tf_predictions, -self.top_k, axis=1)[0][-self.top_k:])
                acc_tf_mat += np.array_equal(mat_pred, tf_top)

                # model inference with onnxruntime
                input_name = sess.get_inputs()[0].name
                ort_start_time = time.time()
                pred_onx = sess.run(None, {input_name: image.astype(np.float32)})[0]
                ort_test_time += time.time() - ort_start_time
                ort_top = np.sort(np.argpartition(pred_onx, -self.top_k, axis=1)[0][-self.top_k:])
                acc_tf_ort += np.array_equal(mat_pred, ort_top)

            acc_tf_ort = acc_tf_ort / num_files * 100
            acc_tf_mat = acc_tf_mat / num_files * 100
            average_pred_time_ort = ort_test_time / num_files
            average_pred_time_tf = tf_test_time / num_files
            print(f"Accuracy of tf -> onnxruntim: {acc_tf_ort}%")
            print(f"Accuracy of tf -> matlab: {acc_tf_ort}%")
            print(f"Average prediction time of tensorflow: {average_pred_time_ort}s")
            print(f"Average prediction time of onnxtime: {average_pred_time_ort}s")
            print(f"Average prediction time of MATALB: {matlab_avg_time}s")


        # Todo: test pytorch
        elif self.origin_framework == "pytorch":
            """
            PyTorch. Steps:
            1. init test dataset
            2. get models of another two frameworks
            3. inference and compare
            """
            # load origin model in pytorch
            """ Reference: Convert a PyTorch model to Tensorflow using ONNX
            https://github.com/onnx/tutorials/blob/master/tutorials/PytorchTensorflowMnist.ipynb
            """
            if not self.model_object:
                self.model_object = torch.load(
                    self.paths["saved_models"].joinpath("origin_{}_{}.pt".format(self.origin_framework,
                                                                                 self.model_name)))
            self.model_object.eval()

            # onnx-tf
            model = onnx.load('output/mnist.onnx')
            tf_rep = prepare(model)

            # load onnx into MATLAB
            print("Now initialize {} model in MATLAB...".format(self.model_name))
            eng = matlab.engine.start_matlab()
            info = {
                "modelName": self.model_name,
                "dataRoot": str(self.paths["coco_dataset"]),
                "savePath": str(self.paths["saved_models"])
            }
            eng.addpath(str(pathlib.Path(__file__).parent))
            _ = eng.init_and_export_matlab_model(info)
            print("...{} model in MATLAB is successfully generated!".format(self.model_name))

        else:
            """
            MATLAB. Steps:
            1. get models of another two frameworks
            3. inference and compare
            """
            pass


def test_model_in_matlab(onnx_path, dataset_path, isInception, topK):
    """Test a model of onnx in matlab.

    Args:
        onnx_path (str): path of onnx model file
        dataset_path (str): path of dataset
        isInception (boolean): if the model is an inception model
        topK (int): specify number of top classes to return

    Returns:
        filename_list (list): list of data file/images-name found in the dataset path
        predictions (list): list of predictions for each file
        average_time (float): total prediction time

    """
    print("Test model in matlab...")
    eng = matlab.engine.start_matlab()

    eng.addpath(str(pathlib.Path(__file__).parent))
    filename_list, predictions, average_time = eng.test_model_in_matlab(onnx_path, dataset_path,
                                                                        isInception, topK, nargout=3)
    print("...model is successfully tested in MATLAB.")

    return filename_list, predictions, average_time


def get_top_k_predictions(predictions, k):
    ind = np.argpartition(predictions, -k, axis=1)[0][-k:]

    return ind


def load_and_preprocess_single_img(img_path: str, size: int) -> np.ndarray:
    """Load an image as numpy array and pre-process

    Preprocess include:
        - resize to `size`
        - rescale to [0-1]
        - zero center
        - normalization

    Args:
        img_path (str): specify image path
        size (int): specify size of final image

    Return:
        np.ndarray: numpy array of image
    """

    image = tf.keras.preprocessing.image.load_img(img_path,
                                                  target_size=(size, size))
    image = tf.keras.preprocessing.image.img_to_array(image)
    # rescale to 0 - 1
    image = image / 255.0
    # normalization
    mean = np.array([0.485, 0.456, 0.406])
    dev = np.array([0.229, 0.224, 0.225])
    image = image - mean.reshape((1, 1, 3))
    image = image / dev.reshape((1, 1, 3))
    # convert single image to a batch
    input_arr = np.array([image])

    return input_arr



