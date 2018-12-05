import abc
import csv
from typing import List, Dict


class Experiment(abc.ABC):

    def __init__(self, datadir, balanced=True, stats_filepath=''):
        self.datadir = datadir
        self.balanced = balanced
        self.stats_filepath = stats_filepath

    @abc.abstractmethod
    def run():
        pass

    def save_class_stats(self, stats: List[Dict]):
        keys = list(stats[0].keys())
        # put name in first place
        keys.remove('name')
        keys = ['name'] + keys
        with open(self.stats_filepath, 'w') as output_file:
            writer = csv.DictWriter(output_file, keys)
            writer.writeheader()
            writer.writerows(stats)
