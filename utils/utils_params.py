import pathlib
import datetime


def gen_run_folder(mode, test_id=''):
    run_paths = dict()
    date_creation = datetime.datetime.now().strftime('%Y-%m-%d@%H-%M-%S-%f')
    if test_id:
        test_folder_root = pathlib.Path(__file__).resolve().parents[1]\
            .joinpath("test_result", date_creation, "_", test_id)
    else:
        test_folder_root = pathlib.Path(__file__).resolve().parents[1]\
            .joinpath("test_result", date_creation)

    run_paths['root'] = test_folder_root
    run_paths['gin'] = test_folder_root.joinpath("config_operative.gin")
    run_paths['report'] = test_folder_root.joinpath("report")
    if mode == "convert":
        run_paths["saved_models"] = test_folder_root.joinpath("saved_models")
        run_paths["saved_models"].mkdir(parents=True, exist_ok=True)

    # Create folders
    for k, v in run_paths.items():
        if any([x in k for x in ["root", "saved_models"]]):
            v.mkdir(parents=True, exist_ok=True)

    # Create files
    for k, v in run_paths.items():
        if any([x in k for x in ['gin', 'report']]):
            v.touch(exist_ok=True)

    return run_paths
