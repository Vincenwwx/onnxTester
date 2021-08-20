import matlab.engine
import pathlib


eng = matlab.engine.start_matlab()
eng.addpath(str(pathlib.Path(__file__).parent))

onnxPath = "/Users/vincen/Knowledge Bank/Artificial Intellegence/Forschungsarbeit/Codebase/onnxTester/test_result/2021-08-18@11-48-30-468902_tf-incpt_load/saved_models/origin_tensorflow_inceptionv3.onnx";
isInception = True
dataPath = "/Users/vincen/Knowledge Bank/Artificial Intellegence/Forschungsarbeit/Codebase/backup/onnx/test"
topK = 5

filename_list, predictions, runtime = eng.test_model_in_matlab(onnxPath, dataPath, isInception, topK, nargout=3)
print(predictions)
# print(runtime)