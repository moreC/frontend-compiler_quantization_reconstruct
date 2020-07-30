import os
import joblib
from collections import namedtuple

VERSION = '1.0'
SubProblemRecord = namedtuple('SubProblemRecord', ['xtx', 'yty', 'xty', 'kernel', 'bias', 'shape'])


class PruningDataset:
    def __init__(self):
        self.record_dict = dict()

    def insert_record(self, op_name, xtx, yty, xty, shape, kernel=None, bias=None):
        if op_name in self.record_dict:
            raise KeyError(f'op_name {op_name} already exists')
        self.record_dict[op_name] = SubProblemRecord(xtx=xtx, yty=yty, xty=xty, shape=shape, kernel=kernel, bias=bias)

    def get_record_dict(self):
        return self.record_dict

    def dump(self, file_path):
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        joblib.dump(self.record_dict, file_path, compress=3, protocol=-1)

    def load(self, file_path):
        self.record_dict = joblib.load(file_path)
