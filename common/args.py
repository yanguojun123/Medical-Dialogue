
import argparse
import os
import torch
import logging
from datetime import datetime
#import ipdb


class ArgsParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser = argparse.ArgumentParser()
        parser.add_argument('--evaluation_path', default='result.json', type=str, required=False, help='evaluation_path')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()

        return args

