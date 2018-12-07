import abc
import csv
from typing import List, Dict


class Experiment(abc.ABC):

    def __init__(self, datadir, balanced=True, classmetrics_filepath=''):
        self.datadir = datadir
        self.balanced = balanced
        self.classmetrics_filepath = classmetrics_filepath

    @abc.abstractmethod
    def run():
        pass

