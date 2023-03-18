import logging
import sys


def _get_logger():
    # logging.basicConfig(level=logging.INFO,
    #                 filemode = 'w',
    #                 filename='/nas/xd/transformers/notebooks/lxy/output.log',
    #                 datefmt='%Y/%m/%d %H:%M:%S',
    #                 format='[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d] - %(message)s')
    log = logging.getLogger('log')
    log.setLevel(logging.INFO)

    # StreamHandler
    console_handle = logging.StreamHandler(sys.stdout)
    console_handle.setLevel(logging.INFO)
    console_handle.setFormatter(logging.Formatter('[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d] - %(message)s',
                                                  datefmt='%Y-%m-%d %H:%M:%S'))
    log.addHandler(console_handle)


    file_handler = logging.FileHandler('/nas/xd/projects/transformers/notebooks/lxy/log/output.log')
    file_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d] - %(message)s',
                                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    return log


# 日志句柄
logger = _get_logger()