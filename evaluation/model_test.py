import numpy as np
import tensorflow as tf
import torch
import onnx
from onnx_tf.backend import prepare
import matlab.engine
import pathlib
import time
import onnxruntime
from data_pipeline.tf_preprocess import tf_load_and_preprocess_single_img
from data_pipeline.torch_datareader import torch_load_and_preprocess_single_img


class Performance_Tester(object):
    def __init__(self, model_name, origin_framework, paths, model_object=None, top_k=5):
        self.origin_framework = origin_framework.lower()
        self.model_name = model_name.lower()
        self.top_k = top_k
        self.paths = paths
        self.onnx_path = self.paths["saved_models"].joinpath("origin_{}_{}.onnx".format(origin_framework,
                                                                                        model_name))
        self.onnx_object = onnx.load(str(self.onnx_path))
        if model_object:
            # if model object is passed
            self.model_object = model_object
        else:
            # when model object not passed, load it from the directory
            if origin_framework == "pytorch":
                self.model_object = torch.load(str(
                    self.paths["saved_models"].joinpath("origin_{}_{}.pt".format(origin_framework, model_name))))
                self.model_object.eval()  # set pytorch model to evaluation model
            elif origin_framework == "tensorflow":
                self.model_object = tf.keras.models.load_model(str(
                    self.paths["saved_models"].joinpath("origin_{}_{}.pt".format(origin_framework, model_name))))
            else:
                self.model_object = None

        if model_name == "inceptionv3":
            self.size = 299
            self.is_inception = True
        else:
            self.size = 224
            self.is_inception = False

    def test_model_conversion(self):
        """Test onnx's ability to convert model by comparing the performance of exported models with
        the origin model.
        """
        print("{ Model conversion test }")
        # Todo: change folder back
        dataset_path = self.paths["coco_dataset"].joinpath("images", "temp")

        # test model in MATLAB
        print("Model inference using MATLAB...")
        imgs_name_list, matlab_preds, matlab_avg_time = test_model_in_matlab(onnx_path=str(self.onnx_path),
                                                                             dataset_path=str(dataset_path),
                                                                             is_inception=self.is_inception,
                                                                             top_k=self.top_k)
        print("Model inference in MATLAB finished!")
        num_imgs = len(imgs_name_list)

        if self.origin_framework == "tensorflow":
            """ When the origin model is in tf, compare the model with model in MATLAB.
            """
            tf_test_time = 0
            acc_tf_mat = 0

            for i in range(num_imgs):
                print("Now testing the {}/{} image...".format(i, num_imgs))
                mat_pred = np.array(matlab_preds[i]) - 1    # indices of MATLAB start from 1
                mat_top = np.sort(np.array(mat_pred.reshape(self.top_k)))
                image = tf_load_and_preprocess_single_img(imgs_name_list[i], size=self.size)
                # predict with tf model and compare with matlab
                tf_start_time = time.time()
                tf_predictions = self.model_object.predict(image)
                tf_test_time += (time.time() - tf_start_time)
                tf_top = np.sort(np.argpartition(tf_predictions, -self.top_k, axis=1)[0][-self.top_k:])
                acc_tf_mat += np.array_equal(mat_top, tf_top)

            acc_tf_mat = acc_tf_mat / num_imgs * 100
            tf_average_pred_time  = tf_test_time / num_imgs
            print(f"Accuracy of tf <-> matlab: {acc_tf_mat}%")
            print(f"Average prediction time of tensorflow: {tf_average_pred_time }s")
            print(f"Average prediction time of MATALB: {matlab_avg_time}s")

        # Todo
        elif self.origin_framework == "pytorch":
            """ When the origin model is in torch, compare it with models in tf and MATLAB.
            """
            tf_test_time = 0
            torch_test_time = 0
            acc_torch_tf = 0
            acc_torch_mat = 0

            tf_model = prepare(self.onnx_object)  # get tf model
            exported_tf_path = str(self.paths["saved_models"].joinpath("exported_tf"))
            tf_model.export_graph(exported_tf_path)
            loaded = tf.saved_model.load(exported_tf_path)
            infer = loaded.signatures["serving_default"]
            for k, _ in infer.structured_outputs.items():
                output_layer_name = k

            for i in range(num_imgs):

                print("Now testing the {}/{} image...".format(i, num_imgs))
                """ Reference: Convert a PyTorch model to Tensorflow using ONNX
                https://github.com/onnx/tutorials/blob/master/tutorials/PytorchTensorflowMnist.ipynb
                """
                mat_pred = np.array(matlab_preds[i]) - 1  # indexing issue
                mat_top = np.sort(np.array(mat_pred.reshape(self.top_k)))
                # test with torch
                image = torch_load_and_preprocess_single_img(imgs_name_list[i], size=self.size)
                torch_start_time = time.time()
                torch_predictions = self.model_object(image)
                torch_test_time += time.time() - torch_start_time
                torch_top = np.sort(torch.topk(torch_predictions, 5)[0])
                # test with tf
                image = tf_load_and_preprocess_single_img(imgs_name_list[i], size=self.size)
                tf_start_time = time.time()
                tf_predictions = infer(tf.constant(image))[output_layer_name]
                tf_test_time += (time.time() - tf_start_time)
                tf_top = np.sort(np.argpartition(tf_predictions, -self.top_k, axis=1)[0][-self.top_k:])

                acc_torch_tf += np.array_equal(torch_top, tf_top)
                acc_torch_mat += np.array_equal(torch_top, mat_top)

            acc_torch_tf = acc_torch_tf / num_imgs * 100
            acc_torch_mat = acc_torch_mat / num_imgs * 100
            torch_average_pred_time = torch_test_time / num_imgs
            tf_average_pred_time = tf_test_time / num_imgs

            print(f"Accuracy of torch <-> matlab: {acc_torch_mat}%")
            print(f"Accuracy of torch <-> tensorflow: {acc_torch_tf}%")
            print(f"Average prediction time of PyTorch: {torch_average_pred_time}s")
            print(f"Average prediction time of tensorflow: {tf_average_pred_time}s")
            print(f"Average prediction time of MATALB: {matlab_avg_time}s")

        # Todo
        else:
            """ When the origin model is in MATLAB, compare it with model in tf.
            """
            tf_test_time = 0
            acc_mat_tf = 0

            tf_model = prepare(self.onnx_object)  # get tf model
            exported_tf_path = str(self.paths["saved_models"].joinpath("exported_tf"))
            tf_model.export_graph(exported_tf_path)
            loaded = tf.saved_model.load(exported_tf_path)
            infer = loaded.signatures["serving_default"]
            for k, _ in infer.structured_outputs.items():
                output_layer_name = k   # get last layer name
            assert output_layer_name, "Unrecognized name of last layer"

            for i in range(num_imgs):

                print("Now testing the {}/{} image...".format(i, num_imgs))

                mat_pred = np.array(matlab_preds[i]) - 1  # indexing issue
                mat_top = np.sort(np.array(mat_pred.reshape(self.top_k)))
                # test with tf
                image = tf_load_and_preprocess_single_img(imgs_name_list[i], size=self.size)
                tf_start_time = time.time()
                tf_predictions = infer(tf.constant(image))[output_layer_name]
                tf_test_time += (time.time() - tf_start_time)
                tf_top = np.sort(np.argpartition(tf_predictions, -self.top_k, axis=1)[0][-self.top_k:])

                acc_mat_tf += np.array_equal(mat_top, tf_top)

            acc_mat_tf = acc_mat_tf / num_imgs * 100
            tf_average_pred_time = tf_test_time / num_imgs

            print(f"Accuracy of MATLAB <-> tensorflow: {acc_mat_tf}%")
            print(f"Average prediction time of tensorflow: {tf_average_pred_time}s")
            print(f"Average prediction time of MATALB: {matlab_avg_time}s")

    def test_model_inference(self):
        """ Test exported onnx model regarding model inference with different runtime backends,
        which in our case include onnxruntime, onnx-tf and MATLAB.
        """
        sess = onnxruntime.InferenceSession(self.onnx_path)  # init onnxruntime session

        #   model inference with onnxruntime
        input_name = sess.get_inputs()[0].name
        ort_start_time = time.time()
        pred_onx = sess.run(None, {input_name: image.astype(np.float32)})[0]
        ort_test_time += time.time() - ort_start_time
        ort_top = np.sort(np.argpartition(pred_onx, -self.top_k, axis=1)[0][-self.top_k:])
        acc_tf_ort += np.array_equal(mat_pred, ort_top)

        if self.origin_framework == "tensorflow":
            pass
        elif self.origin_framework == "pytorch":
            pass
        else:
            pass


def test_model_in_matlab(onnx_path, dataset_path, is_inception, top_k):
    """Test a model of onnx in matlab.

    Args:
        onnx_path (str): path of onnx model file
        dataset_path (str): path of dataset
        is_inception (boolean): if the model is an inception model
        top_k (int): specify number of top classes to return

    Returns:
        filename_list (list): list of data file/images-name found in the dataset path
        predictions (list): list of predictions for each file
        average_time (float): total prediction time

    """
    print("Test model in matlab...")
    eng = matlab.engine.start_matlab()

    eng.addpath(str(pathlib.Path(__file__).parent))
    filename_list, predictions, average_time = eng.test_model_in_matlab(onnx_path, dataset_path,
                                                                        is_inception, top_k, nargout=3)
    print("...model is successfully tested in MATLAB.")

    return filename_list, predictions, average_time

