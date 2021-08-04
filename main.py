import logging
import gin
import argparse
from utils import utils_params, utils_misc
from models.model_initialize import Model_Initializer
from evaluation.model_test import Tester
from model_convert import convert_origin_model


def main(*argv):
    # parse arguments
    parser = argparse.ArgumentParser(description="Process parameters")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--model_path",
                       help="Set model path if you want to use your own model")
    group.add_argument("-n", "--model_name",
                       help="Specify which model you want to generate")
    parser.add_argument("origin_framework", choices=['tensorflow', 'pytorch', 'matlab'],
                        help="Define the origin DL framework of your model")
    parser.add_argument("mode", choices=["convert", "reference"],
                        help="Set running mode")
    args = parser.parse_args()

    # generate folder structure
    run_paths = utils_params.gen_run_folder(mode=args.mode, test_id="TEST")

    # set loggers
    utils_misc.set_loggers(run_paths['program_logs'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(run_paths['gin_file'], logging.INFO)
    utils_params.save_config(run_paths['gin_log'], gin.config_str())

    # Get DL origin DL model by either loading or auto-generation
    model_init = Model_Initializer(model_name=args.model_name,
                                   model_path=args.model_path,
                                   origin_framework=args.origin_framework)
    model = model_init.get_model()

    # Convert the origin model to models on other frameworks
    converted_models = convert_origin_model(model, args.origin_framework)

    # Test and compare the origin and exported models
    tester = Tester(origin_framework=args.origin_framework,
                    origin_model=model,
                    converted_models=converted_models)
    result = tester.test_models()


if __name__ == "__main__":
    main()
