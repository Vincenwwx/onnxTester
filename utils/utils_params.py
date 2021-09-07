import pathlib
import datetime


def gen_run_folder(part_name="", whole_name=""):
    """Generate folder paths

    User can choose to specify part of the folder name, the whole folder name
    or let the software define the whole name.

    Example:
        $: gen_run_folder(part_name="test")
        will generate a folder name "2021-07-21@00-29-14-123456_tesr"
        in test_result folder

    Args:
        part_name (str): specify part of the folder name
        whole_name (str): specify the whole folder name
    """
    run_paths = dict()
    date_creation = datetime.datetime.now().strftime('%Y-%m-%d@%H-%M-%S-%f')
    if part_name:
        test_folder_root = pathlib.Path(__file__).resolve().parents[1]\
            .joinpath("test_result", date_creation+"_"+part_name)
    else:
        test_folder_root = pathlib.Path(__file__).resolve().parents[1] \
            .joinpath("test_result", date_creation)
    if whole_name:
        test_folder_root = pathlib.Path(__file__).resolve().parents[1]\
            .joinpath("test_result", whole_name)

    run_paths['root'] = test_folder_root
    run_paths['program_log'] = test_folder_root.joinpath("program_log")
    run_paths['gin_log'] = test_folder_root.joinpath("config_operative.gin")
    run_paths['report'] = test_folder_root.joinpath("report")
    run_paths['coco_dataset'] = pathlib.Path(__file__).resolve().parents[1]\
        .joinpath("data_pipeline", "coco_2017")
    run_paths["saved_models"] = test_folder_root.joinpath("saved_models")
    run_paths["saved_models"].mkdir(parents=True, exist_ok=True)

    # Create folders
    for k, v in run_paths.items():
        if any([x in k for x in ["root", "saved_models"]]):
            v.mkdir(parents=True, exist_ok=True)

    # Create files
    for k, v in run_paths.items():
        if any([x in k for x in ['log', 'report']]):
            v.touch(exist_ok=True)

    return run_paths


def save_config(path, configuration):
    p = pathlib.Path(path)
    p.write_text(configuration)
