import logging


def set_loggers(path_log=None, logging_level=0, print_out=False):
    """
    set loggers
    :param path_log: path of log file
    :param logging_level: specify from which level of level the information should be logged
                          int, can be 0 to 5
    :param print_out: boolean, if True, the logging info will also be printed to console
    :return: None
    """
    # standard logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    # set file handler to record log info to file
    file_handler = logging.FileHandler(path_log)
    # set formatter for file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # print log info to console
    if print_out:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)
