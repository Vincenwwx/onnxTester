import logging
import gin
import argparse
from utils import utils_params, utils_misc
from models.model_initialize import Model_Initializer
from evaluation.model_test import Performance_Tester


def main(*argv):
    # parse arguments
    parser = argparse.ArgumentParser(description="onnxTester parameters")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--model_path",
                       help="Set model path if you want to use your own model")
    group.add_argument("-n", "--model_name", choices=["inceptionv3", "resnet50", "vgg16"],
                       help="Specify which model you want to generate")
    parser.add_argument("origin_framework", choices=['tensorflow', 'pytorch', 'matlab'],
                        help="Define the origin DL framework of your model")
    parser.add_argument("test", choices=["conversion", "inference"],
                        help="Specify which test to conduct")
    args = parser.parse_args()

    # generate folder structure to save test result
    run_paths = utils_params.gen_run_folder(part_name="TEST")

    # set loggers
    utils_misc.set_loggers(path_log=run_paths['program_log'], logging_level=logging.INFO, print_out=True)

    # gin-config
    gin.parse_config_file(str(run_paths['gin_file']))
    utils_params.save_config(str(run_paths['gin_log']), gin.config_str())

    # Get DL origin DL model by either loading or auto-generation
    model_initializer = Model_Initializer(model_name=args.model_name,
                                          model_path=args.model_path,
                                          origin_framework=args.origin_framework,
                                          paths=run_paths)

    # Export the origin model to onnx
    model_initializer.export_model_to_onnx()

    # Test and compare the origin and exported models
    tester = Performance_Tester(origin_framework=args.origin_framework,
                                model_name=args.model_name, paths=run_paths)
    if args.test == "conversion":
        tester.test_model_conversion("temp")
    else:
        tester.test_model_inference("temp")


if __name__ == "__main__":
    main()
