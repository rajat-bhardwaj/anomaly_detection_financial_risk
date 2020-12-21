import logging
import os
import sys


class CustomLogger:

    def __init__(self, filename):
        self.filename = filename

    def setLogconfig(self):

        log = logging.getLogger()
        log.setLevel(logging.INFO)

        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # create file handler which logs even debug messages
        path = (os.path.dirname(os.path.realpath('__file__')))
        path = os.path.join(path, 'logs')
        file_path = os.path.join(path, self.filename)

        fh = logging.FileHandler(file_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
        log.addHandler(logging.StreamHandler(sys.stdout))
