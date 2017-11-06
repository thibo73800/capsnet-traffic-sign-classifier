# coding: utf-8

import numpy as np
import json
import sys
import os


class Utils(object):
    """
        Util class to store all common method use in this project
    """

    def __init__(self, arg):
        super(Utils, self).__init__()

    @staticmethod
    def progress(count, total, suffix=''):
        """
            Utils method to display a progress bar
            **input: **
                *count: current progression
                *total: Max progress bar length
        """
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
        sys.stdout.flush()

    @staticmethod
    def read_json_file(path):
        """
            Utils method to open, read and return a json file content
            **input: **
                *path: (String) Path to the json file to read
        """
        with open(path, "r") as f:
            json_content = json.loads(f.read())
        return json_content
