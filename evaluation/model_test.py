import numpy as np
import tensorflow as tf
import torch
import onnx
from onnx_tf.backend import prepare
import matlab.engine
import logging
import pathlib
import time
import onnxruntime
from data_pipeline.tf_preprocess import tf_load_and_preprocess_single_img
from data_pipeline.torch_datareader import torch_load_and_preprocess_single_img
from evaluation.inference_helper import send_single_img_to_tensorflow_serving, \
    init_tf_serving_docker, is_top_k_identical


class Performance_Tester(object):
    """Performance tester

    Args:
        model_name (str): specify name of model ("vgg16", "resnet50" or "inceptionv3")
        origin_framework (str): specify with which framework the model is built
        paths (dict): specify a dictionary of paths, where the model will be saved and read
        top_k (int): specify number of top indices

    """

    def __init__(self, model_name, origin_framework, paths, top_k=5):

        self.origin_framework = origin_framework.lower()
        self.model_name = model_name.lower()
        self.paths = paths
        self.top_k = top_k

        # load origin model from path
        if origin_framework == "pytorch":
            model_path = str(
                self.paths["saved_models"].joinpath("origin_{}_{}.pt".format(origin_framework, model_name)))
            self.model_object = torch.load(model_path)
            self.model_object.eval()  # set pytorch model to evaluation model
            print("[System] Successfully load saved model {} under {}".format(model_name, origin_framework))
        elif origin_framework == "tensorflow":
            model_path = str(
                self.paths["saved_models"].joinpath("origin_{}_{}".format(origin_framework, model_name), "1"))
            self.model_object = tf.keras.models.load_model(model_path)
            print("[System] Successfully load saved model {} under {}".format(model_name, origin_framework))
        else:
            model_path = self.paths["saved_models"].joinpath("origin_{}_{}.onnx".format(origin_framework, model_name))
            assert model_path.exists(), "[Error] MATLAB saved model doesn't exist!"

        if model_name == "inceptionv3":
            self.size = 299
            self.is_inception = True
        else:
            self.size = 224
            self.is_inception = False

        # get ready for onnx
        self.onnx_path = self.paths["saved_models"].joinpath(
            "origin_{}_{}.onnx".format(origin_framework, model_name))
        self.onnx_object = onnx.load(str(self.onnx_path))

    def test_model_conversion(self, test_dataset="val"):
        """Test onnx's ability to convert model by comparing the performance of exported models with
        the origin model.
        """
        print("{ Model conversion test }")
        dataset_path = self.paths["coco_dataset"].joinpath("images", test_dataset)

        # test model in MATLAB
        imgs_name_list, matlab_preds, matlab_avg_time = self.test_model_in_matlab(dataset_path=str(dataset_path))
        num_imgs = len(imgs_name_list)

        if self.origin_framework == "tensorflow":
            """ When the origin model is in tf, compare the model with model in MATLAB.
            """
            tf_test_time = 0
            acc_tf_mat = 0

            for i in range(num_imgs):
                print("\tNow testing the {}/{} image...".format(i, num_imgs))

                image = tf_load_and_preprocess_single_img(imgs_name_list[i], size=self.size)
                # predict with tf model and compare with matlab
                tf_start_time = time.time()
                tf_predictions = self.model_object.predict(image)
                tf_test_time += (time.time() - tf_start_time)
                acc_tf_mat += is_top_k_identical(matlab_preds[i], tf_predictions)

            acc_tf_mat = acc_tf_mat / num_imgs * 100
            tf_average_pred_time = tf_test_time / num_imgs

            logging.info("{:=^50}".format(" Model Conversion Test "))
            basic = " {} in {} ".format(self.model_name, self.origin_framework)
            logging.info("{:-^50}".format(basic))
            logging.info("Top-{} ")
            logging.info(f"Accuracy of tf <-> matlab: {acc_tf_mat}%")
            logging.info(f"Average prediction time of tensorflow: {tf_average_pred_time}s")
            logging.info(f"Average prediction time of MATALB: {matlab_avg_time}s\n")

        elif self.origin_framework == "pytorch":
            """ When the origin model is in torch, compare it with models in tf and MATLAB.
            """
            tf_test_time = 0
            torch_test_time = 0
            acc_torch_tf = 0
            acc_torch_mat = 0

            onnx_tf_sess = prepare(self.onnx_object)  # get tf model
            exported_tf_path = self.paths["saved_models"].joinpath("exported_tf", "1")
            if not exported_tf_path.exists():
                onnx_tf_sess.export_graph(str(exported_tf_path))
            loaded = tf.saved_model.load(str(exported_tf_path))
            infer = loaded.signatures["serving_default"]
            for k, _ in infer.structured_outputs.items():
                output_layer_name = k

            for i in range(num_imgs):
                print("\tNow testing the {}/{} image...".format(i + 1, num_imgs))
                """ Reference: Convert a PyTorch model to Tensorflow using ONNX
                https://github.com/onnx/tutorials/blob/master/tutorials/PytorchTensorflowMnist.ipynb
                """
                # test with torch
                image = torch_load_and_preprocess_single_img(imgs_name_list[i], size=self.size)
                torch_start_time = time.time()
                torch_predictions = self.model_object(image)
                torch_test_time += time.time() - torch_start_time

                # test with tf
                tf_start_time = time.time()
                tf_predictions = infer(tf.constant(image))[output_layer_name]
                tf_test_time += (time.time() - tf_start_time)

                torch_predictions = torch_predictions.detach().numpy()
                acc_torch_tf += is_top_k_identical(torch_predictions, tf_predictions)
                acc_torch_mat += is_top_k_identical(torch_predictions, matlab_preds[i])

            acc_torch_tf = acc_torch_tf / num_imgs * 100
            acc_torch_mat = acc_torch_mat / num_imgs * 100
            torch_average_pred_time = torch_test_time / num_imgs
            tf_average_pred_time = tf_test_time / num_imgs

            logging.info("------------ Conversion Test Result -------------")
            logging.info("\t--- {} in {} ---".format(self.model_name, self.origin_framework))
            logging.info(f"Accuracy of torch <-> matlab: {acc_torch_mat}%")
            logging.info(f"Accuracy of torch <-> tensorflow: {acc_torch_tf}%")
            logging.info(f"Average prediction time of PyTorch: {torch_average_pred_time}s")
            logging.info(f"Average prediction time of tensorflow: {tf_average_pred_time}s")
            logging.info(f"Average prediction time of MATALB: {matlab_avg_time}s\n")

        else:
            """ When the origin model is in MATLAB, compare it with model in tf.
            """
            tf_test_time = 0
            acc_mat_tf = 0

            # export onnx model to tf
            onnx_tf_sess = prepare(self.onnx_object)  # get tf model
            exported_tf_path = self.paths["saved_models"].joinpath("exported_tf", "1")
            if not exported_tf_path.exists():
                onnx_tf_sess.export_graph(str(exported_tf_path))
            # load tf (saved) model
            loaded = tf.saved_model.load(str(exported_tf_path))
            infer = loaded.signatures["serving_default"]
            for k, _ in infer.structured_outputs.items():
                output_layer_name = k  # get last layer name
            assert output_layer_name, "Unrecognized name of last layer"

            for i in range(num_imgs):
                print("Now testing the {}/{} image...".format(i, num_imgs))

                # test with tf
                image = torch_load_and_preprocess_single_img(imgs_name_list[i], size=self.size)
                tf_start_time = time.time()
                tf_predictions = infer(tf.constant(image))[output_layer_name]
                tf_test_time += (time.time() - tf_start_time)

                acc_mat_tf += is_top_k_identical(matlab_preds[i], tf_predictions)

            acc_mat_tf = acc_mat_tf / num_imgs * 100
            tf_average_pred_time = tf_test_time / num_imgs

            logging.info("\n------------ Conversion Test Result -------------")
            logging.info("\t--- {} in {} ---".format(self.model_name, self.origin_framework))
            logging.info(f"Accuracy of MATLAB <-> tensorflow: {acc_mat_tf}%")
            logging.info(f"Average prediction time of tensorflow: {tf_average_pred_time}s")
            logging.info(f"Average prediction time of MATALB: {matlab_avg_time}s")

        print("Finished model conversion test!")

    def test_model_inference(self, test_dataset="val", percentile=90):
        """ Test exported onnx model regarding model inference with different runtime backends,
        which in our case includes onnxruntime, onnx-tf and MATLAB.
        """
        print("{ Model Inference Test }")
        # prepare test dataset
        dataset_path = self.paths["coco_dataset"].joinpath("images", test_dataset)
        imgs_path_list = list(sorted(dataset_path.glob("*.jpg")))
        num_imgs = len(imgs_path_list)
        print("Find {} test images".format(num_imgs))

        # init onnxruntime
        ort_sess = onnxruntime.InferenceSession(str(self.onnx_path))  # init onnxruntime session
        ort_input_name = ort_sess.get_inputs()[0].name
        # init onnx-tf
        onnx_tf_sess = prepare(self.onnx_object)

        acc_origin_and_onnx_tf = 0
        acc_origin_and_onnxruntime = 0
        acc_origin_and_tf_serving = 0
        tf_serving_test_time = []
        onnx_tf_test_time = []
        onnxruntime_test_time = []

        if self.origin_framework == "tensorflow":

            # init TF-Serving in docker
            tf_model_path = str(
                self.paths["saved_models"].joinpath("origin_{}_{}".format(self.origin_framework, self.model_name)))
            server = init_tf_serving_docker(model_path=tf_model_path, model_name=self.model_name)

            for i in range(num_imgs):
                print("\tNow testing the {}/{} image...".format(i, num_imgs))
                image = tf_load_and_preprocess_single_img(imgs_path_list[i], size=self.size)
                # predict with tf model and compare with matlab
                tf_preds = self.model_object.predict(image)

                # model inference: tensorflow serving
                tf_serving_preds, interval = send_single_img_to_tensorflow_serving(image, model_name=self.model_name)
                tf_serving_test_time.append(interval)
                # model inference: onnx-tf
                ts = time.time()
                onnx_tf_preds = onnx_tf_sess.run(image)  # return type: np.ndarray
                onnx_tf_test_time.append(time.time() - ts)
                # model inference: onnxruntime
                ts = time.time()
                ort_preds = ort_sess.run(None, {ort_input_name: image.astype(np.float32)})[0]
                onnxruntime_test_time.append(time.time() - ts)

                acc_origin_and_onnx_tf += is_top_k_identical(tf_preds, onnx_tf_preds)
                acc_origin_and_onnxruntime += is_top_k_identical(tf_preds, ort_preds)
                acc_origin_and_tf_serving += is_top_k_identical(tf_preds, tf_serving_preds)

        elif self.origin_framework == "pytorch":

            # export to tensorflow model
            exported_tf_path = self.paths["saved_models"].joinpath("exported_tf", "1")
            if not exported_tf_path.exists():
                onnx_tf_sess.export_graph(str(exported_tf_path))
                print("[System] Successfully export {} model to TensorFlow model.".format(self.origin_framework))
            else:
                print("[System] Detect existent exported TensorFlow model!")
            # init TensorFlow Serving in docker
            server = init_tf_serving_docker(model_path=str(exported_tf_path.parent), model_name=self.model_name)

            for i in range(num_imgs):
                print("Now testing the {}/{} image...".format(i, num_imgs))
                # get reference model prediction
                image = torch_load_and_preprocess_single_img(imgs_path_list[i],
                                                             size=self.size)  # of shape [N * 3 * size * size]
                ref_predictions = self.model_object(image)
                ref_predictions = ref_predictions.detach().numpy()

                # image = tf_load_and_preprocess_single_img(imgs_path_list[i], size=self.size)
                # model inference: tensorflow serving
                tf_serving_preds, interval = send_single_img_to_tensorflow_serving(image=image.numpy(),
                                                                                   model_name=self.model_name)
                tf_serving_test_time.append(interval)
                # model inference: onnx-tf
                ts = time.time()
                onnx_tf_preds = onnx_tf_sess.run(image.numpy())  # return type: np.ndarray
                onnx_tf_test_time.append(time.time() - ts)
                # model inference: onnxruntime
                ts = time.time()
                ort_preds = ort_sess.run(None, {ort_input_name: image.numpy().astype(np.float32)})[0]
                # init TensorFlow Serving in docker
                onnxruntime_test_time.append(time.time() - ts)

                acc_origin_and_onnx_tf += is_top_k_identical(ref_predictions, onnx_tf_preds)
                acc_origin_and_onnxruntime += is_top_k_identical(ref_predictions, ort_preds)
                acc_origin_and_tf_serving += is_top_k_identical(ref_predictions, tf_serving_preds)

        else:

            # export to tensorflow model
            exported_tf_path = self.paths["saved_models"].joinpath("exported_tf", "1")
            # init TensorFlow Serving in docker
            if not exported_tf_path.exists():
                onnx_tf_sess.export_graph(str(exported_tf_path))
                print("[System] Successfully export {} model to TensorFlow model.".format(self.origin_framework))
            else:
                print("[System] Detect existent exported TensorFlow model!")
            server = init_tf_serving_docker(model_path=str(exported_tf_path.parent), model_name=self.model_name)

            _, matlab_preds, _ = self.test_model_in_matlab(dataset_path=str(dataset_path))

            for i in range(num_imgs):
                print("Now testing the {}/{} image...".format(i, num_imgs))
                # get reference model prediction
                ref_predictions = np.array(matlab_preds[i]).reshape(91)
                image = torch_load_and_preprocess_single_img(imgs_path_list[i], size=self.size)
                image = image.numpy()

                # image = tf_load_and_preprocess_single_img(imgs_path_list[i], size=self.size)
                # model inference: tensorflow serving
                tf_serving_preds, interval = send_single_img_to_tensorflow_serving(image=image,
                                                                                   model_name=self.model_name)
                tf_serving_test_time.append(interval)
                # model inference: onnx-tf
                ts = time.time()
                onnx_tf_preds = onnx_tf_sess.run(image)  # return type: np.ndarray
                onnx_tf_test_time.append(time.time() - ts)
                # model inference: onnxruntime
                ts = time.time()
                ort_preds = ort_sess.run(None, {ort_input_name: image.astype(np.float32)})[0]
                onnxruntime_test_time.append(time.time() - ts)

                acc_origin_and_onnx_tf += is_top_k_identical(ref_predictions, onnx_tf_preds)
                acc_origin_and_onnxruntime += is_top_k_identical(ref_predictions, ort_preds)
                acc_origin_and_tf_serving += is_top_k_identical(ref_predictions, tf_serving_preds)

        acc_origin_and_onnx_tf = acc_origin_and_onnx_tf / num_imgs * 100
        acc_origin_and_tf_serving = acc_origin_and_tf_serving / num_imgs * 100
        acc_origin_and_onnxruntime = acc_origin_and_onnxruntime / num_imgs * 100

        onnx_tf_test_time = np.percentile(onnx_tf_test_time, percentile)
        tf_serving_test_time = np.percentile(tf_serving_test_time, percentile)
        onnxruntime_test_time = np.percentile(onnxruntime_test_time, percentile)

        logging.info("------------ Inference Test Result -------------")
        logging.info("- {} in {}".format(self.model_name, self.origin_framework))
        logging.info("- Dataset: {}\n".format(test_dataset))

        logging.info("Top-{} accuracy".format(self.top_k))
        logging.info("\t{} <--> {:^18} : {}%".format(self.origin_framework, "onnx-tf", acc_origin_and_onnx_tf))
        logging.info(
            "\t{} <--> {:^18} : {}%".format(self.origin_framework, "TensorFlow Serving", acc_origin_and_tf_serving))
        logging.info(
            "\t{} <--> {:^18} : {}%\n".format(self.origin_framework, "onnxruntime", acc_origin_and_onnxruntime))

        logging.info("Latency ({} percentile)".format(percentile))
        logging.info("\t{:^18} : {}s".format("onnx-tf", onnx_tf_test_time))
        logging.info("\t{:^18} : {}s".format("TensorFlow Serving", tf_serving_test_time))
        logging.info("\t{:^18} : {}s".format("onnxruntime", onnxruntime_test_time))

        server.kill()  # stop TensorFlow Serving container
        print("Finished inference test!")

    def test_model_in_matlab(self, dataset_path: str):
        """Test a model of onnx in matlab.

        Args:
            dataset_path (str): path of dataset

        Returns:
            filename_list (list): list of data file/images-name found in the dataset path
            predictions (list): list of predictions for each file
            average_time (float): total prediction time

        """
        print("Test model in matlab...")
        eng = matlab.engine.start_matlab()
        eng.addpath(str(pathlib.Path(__file__).parent))
        filename_list, predictions, average_time = eng.test_model_in_matlab(str(self.onnx_path),
                                                                            self.origin_framework,
                                                                            dataset_path,
                                                                            self.is_inception,
                                                                            nargout=3)
        print("...model is successfully tested in MATLAB.")

        return filename_list, predictions, average_time

    def _generate_repoort(self, test_type: str):
        """Generate test report and write to `report` file

        Args:
            test_type (str): specify test type, either "conversion" or "inference"

        """
        assert test_type in ["conversion", "inference"], "Test type can only be either `conversion` or `inference`"

        report_file_path = self.paths["report"]

        if test_type == "conversion":
            pass
