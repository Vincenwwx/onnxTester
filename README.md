# onnxTester

## Introduction

The project onnxTester aims at analysing the performance of the open neural exchange (ONNX) format, which is used by 
data scientists or machine learning engineers to decouple the neural network models from frameworks like Tensorflow, 
Pytorch. 

### ONNX and ONNXruntime
[ONNX](https://github.com/onnx/onnx) is an open intermediate format for many prevalent Deep Learning (DL) frameworks, 
for instance TensorFlow, PyTorch, MXNET, enabling interoperability between different frameworks and streamlining the 
path from research to production helps increase the speed of innovation in the AI community.

[ONNXruntime](https://github.com/microsoft/onnxruntime) is a high-performance cross-platform inference engine and 
training machine-learning accelerator.

### Scenario
Let's say you are a [TensorFlow](https://www.tensorflow.org/) expert but someday receive a well-trained 
[PyTorch](https://pytorch.org/) model from another colleague who is absolutely a PyTorch freak. 
To work directly on such model file, you can then ask that colleague to export the neural 
network as a .onnx file, which can be later loaded and operated in TensorFlow. 

**However, you question if the model can still perform well as it is in PyTorch...**

## Evaluation
### Target platforms
The deep learning frameworks to test in the project include:
- TensorFlow
- PyTorch
- [MATLAB deep learning toolbox](https://www.mathworks.com/products/deep-learning.html)

Inference engines/backends to be benchmarked include:
- TensorFlow Backend for ONNX: [onnx-tf](https://github.com/onnx/onnx-tensorflow)
- Official model serving system by TensorFlow: [TensorFlow Serving](https://github.com/tensorflow/serving)
- onnxruntime

### Tasks 
_onnxTester_ evaluates the interoperability of ONNX (and ONNXruntime) by conducting the following task:
1. inter-converting a trained model in different frameworks
2. comparing the latency and accuracy of model inference using different runtimes
by comparing:
- top-5 accuracy consistency
- average prediction time
- 90-percentile prediction latency

# Installation

## Pre-requisite

### Software
[MATLAB 2020b](https://www.mathworks.com/products/matlab.html)

[docker](https://www.docker.com/)

### Python and dependencies
python == 3.7

onnx
```shell
pip install numpy protobuf==3.16.0
pip install onnx
```
onnxruntime
```shell
pip install onnxruntime
```
tensorflow == 2.6.0
```shell
pip install tensorflow
```
onnx-tf
```shell
pip install onnx-tf
```
tf2onnx
```shell
pip install -U tf2onnx
```
PyTorch and Torchvision
```shell
pip install torch
pip install torchvision
```
h5py
```shell
pip install h5py
```
gin-config
```shell
pip install gin-config
```
pillow
```shell
pip install Pillow
```
scikit-image
```shell
pip install scikit-image
```
docker python SDK
```shell
pip install docker
```
MATLAB Engine API for Python
```shell
# At a MAC or Linux OS
cd "matlabroot/extern/engines/python"
python setup.py install

# At a Window OS
cd "matlabroot\extern\engines\python"
python setup.py install
```
For details please refer to this [official tutorial](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).
## Install onnxTester from source
```shell
git clone https://github.com/Vincenwwx/onnxTester.git
cd onnxTester
```

# Usage
To run the software following command can be used:
```shell
# Add execution perssion to main.py
chmod +x main.py
# Run the software
./main.py (-p MODEL_PATH | -n {inceptionv3, resnet50, vgg16}) {origin_framework} {test_type}
```
### Recommended
The software will automatically download the COCO dataset with TensorFlow interfaces but normally it will be quicker
for user to download the dataset manually.

Therefore before running the software, download the COCO 2017 dataset and unzip it under the `data_pipeline` folder 
and the folder structure should be like:
```
- data_pipeline
    - coco_2017
        - annotations (Annotation files are kept under this folder)
        - images
            - val (Images for validation are kept uder this folder)
            - train (Images for training are kept under this folder)
```
And the [link](https://cocodataset.org/#download) for downloading COCO dataset.
## Examples
### Execute model conversion test
Suppose we want to auto generate a `inception-V3` model in `TensorFlow` and benchmark with that:
```shell
./main.py -n inceptionv3 tensorflow conversion
```
### Execute model inference test
Suppose we want to auto generate a `ResNet-50` model in `MATLAB` and benchmark with that:
```shell
./main.py -n resnet50 matlab inference
```

# Test results
## Model conversion test
### Top-5 Accuracy
Top-5 Accuracy of `VGG-16 model` conversion test

VGG-16     | PyTorch | TensorFlow | MATLAB
---------- | ------- | ---------- | ------
PyTorch    | -       | 100%       | 61.52%
TensorFlow | -       | -          | 87.38%
MATLAB     | -       | 99.98%     | -

Top-5 Accuracy of `Inception-V3` model conversion test


VGG-16     | PyTorch | TensorFlow | MATLAB
---------- | ------- | ---------- | ------
PyTorch    | -       | 100%       | 92.72%
TensorFlow | -       | -          | 97.20%
MATLAB     | -       | 100%       | -

Top-5 Accuracy of `VGG-16 model` conversion test

VGG-16     | PyTorch | TensorFlow | MATLAB
---------- | ------- | ---------- | ------
PyTorch    | -       | 100%       | 74.10%
TensorFlow | -       | -          | 94.50%
MATLAB     | -       | 99.52%     | -

> We can see that when converting a CNN model whatever built by PyTorch or TensorFlow,
there is an obvious prediction unconsistency.

### Average prediction time
![APT_torch](https://user-images.githubusercontent.com/49132368/132762016-2ca6be5c-0718-4eb9-9d74-4aa0b0b8be89.png)

_Figure 1. Average prediction time for models originate from PyTorch_

> Use PyTorch models in TensorFlow can in general achieve some performance improvement.

![APT_tensorflow](https://user-images.githubusercontent.com/49132368/132762018-9eb337b7-7ba0-4784-8b97-49897450b426.png)

_Figure 2. Average prediction time for models originate from TensorFlow_

![APT_matlab](https://user-images.githubusercontent.com/49132368/132762014-a31fa6b8-97ad-44dc-9097-8936482a056c.png)

_Figure 3. Average prediction time for models originate from MATLAB_

> MATLAB takes much longer time to inference model originating from TensorFlow.

## Model inference test

### Top-5 Accuracy

Top-5 Accuracy of `VGG-16` model inference test

VGG-16     | onnx-tf | TensorFlow Serving | onnxruntime
---------- | ------- | ------------------ | -----------
PyTorch    | 99.98%  | 99.98%             | 100%
TensorFlow | 100%    | 100%               | 100%
MATLAB     | 99.98%  | 99.98%             | 99.98%

Top-5 Accuracy of `Inception-V3` model inference test

Inception-V3 | onnx-tf | TensorFlow Serving | onnxruntime
------------ | ------- | ------------------ | -----------
PyTorch      | 100%    | 100%               | 99.98%
TensorFlow   | 99.98%  | 99.98%             | 100%
MATLAB       | 100%    | 100%               | 100%

Top-5 Accuracy of `ResNet-50` model inference test

ResNet-50  | onnx-tf | TensorFlow Serving | onnxruntime
---------- | ------- | ------------------ | -----------
PyTorch    | 99.50%  | 99.50%             | 99.50%
TensorFlow | 100%    | 100%               | 100%
MATLAB     | 99.50%  | 99.50%             | 99.50%

### 90-Percentile Latency
![L_vgg16](https://user-images.githubusercontent.com/49132368/132762026-b0e48d86-00f7-43fb-9635-fdfa0b354854.png)

*Figure 4. Latency of VGG-16 model inference test*

![L_resnet](https://user-images.githubusercontent.com/49132368/132762024-fcd1570d-94e7-40d4-808d-e601016c1df7.png)

_Figure 5. Latency of ResNet-50 model inference test_

![L_incpt](https://user-images.githubusercontent.com/49132368/132762020-4601f48f-3f89-4858-95e8-371556210568.png)

_Figure 6. Latency of Inception-V3 model inference test_
