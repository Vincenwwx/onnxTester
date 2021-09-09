import numpy as np
import json
import time
import requests
import docker
import logging


def init_tf_serving_docker(model_path: str, model_name: str,
                           port: str = "8501", auto_delete: bool = True):
    """Init TensorFlow Serving in docker.

    Args:
        model_path (str): path of target model
        model_name (str): model name
        port (str): port of container to be mapped
        auto_delete (bool): whether to delete container after running

    Return:
        None

    """
    target = "/models/{}".format(model_name)
    source = model_path
    volumes = ["{}:{}".format(source, target)]
    port = {'{}/tcp'.format(port): port}
    env = ["MODEL_NAME={}".format(model_name)]
    image = "tensorflow/serving"

    client = docker.from_env()

    # stop already existent TensorFlow Serving container
    for container in client.containers.list():
        if "tensorflow/serving" in str(container.image):
            container.kill()

    c = client.containers.run(image, detach=True, remove=auto_delete,
                              ports=port, environment=env, volumes=volumes)
    time.sleep(8)  # time to start the container
    logging.info("[System] Successfully start TensorFlow Serving container for model {}!".format(model_name))

    return c


def send_single_img_to_tensorflow_serving(image: np.ndarray,
                                          model_name: str,
                                          server: str = "localhost",
                                          port: str = "8501") -> [np.ndarray, float]:
    """Send single test image to docker server and get predictions

    Args:
        image (np.ndarray): test image of numpy array format
        model_name (str): model name
        server (str): address of docker server
        port (str): port number

    Return:
        np.ndarray: top k indices of return predictions
        float: interval of time used to get prediction

    """
    # send payload
    data = json.dumps({"signature_name": "serving_default",
                       "instances": image.tolist()})
    headers = {"content-type": "application/json"}
    target = f"http://{server}:{port}/v1/models/{model_name}:predict"
    ts = time.time()
    json_response = requests.post(target, data=data, headers=headers)
    interval = time.time() - ts
    # get prediction
    predictions = json.loads(json_response.text)['predictions'][0]

    return [predictions, interval]


def is_top_k_identical(pred_1, pred_2, top_k=5):
    """Compare two input and check if top k indices of both are identical.

    Args:
        pred_1 (unknown): first input
        pred_2 (unknown): second input
        top_k (int): number of top indices

    Return:
        boolean: True if identical, otherwise False

    """
    if not isinstance(pred_1, np.ndarray):
        pred_1 = np.array(pred_1)
    if not isinstance(pred_2, np.ndarray):
        pred_2 = np.array(pred_2)

    pred_1 = np.squeeze(pred_1)  # flatten
    pred_2 = np.squeeze(pred_2)
    assert pred_1.shape[-1] == pred_2.shape[-1], "Error: Sizes of inputs are not identical"
    assert top_k <= pred_1.shape[-1], "Error: top k should be smaller or equal than size of input"

    top_k_1 = np.argpartition(pred_1, -top_k, axis=0)[-top_k:]
    top_k_2 = np.argpartition(pred_2, -top_k, axis=0)[-top_k:]

    return np.array_equal(top_k_1, top_k_2)
