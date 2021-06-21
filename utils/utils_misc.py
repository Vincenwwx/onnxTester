import logging


def set_loggers(path_log=None, logging_level=0, b_stream=False, b_debug=False):

    # standard logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    # set file handler to record log info to file
    file_handler = logging.FileHandler(path_log)
    logger.addHandler(file_handler)

    # print log info to console
    if b_stream:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)
