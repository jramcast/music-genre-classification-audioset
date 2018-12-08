import abc
from typing import Tuple


class Persistence(abc.ABC):
    """
    Saves and load models to file system
    """

    @abc.abstractmethod
    def save(self, model):
        pass

    @abc.abstractmethod
    def load():
        pass


class DatasetLoader(abc.ABC):
    """
    Loads datasets
    """

    @abc.abstractmethod
    def load() -> Tuple:
        """
        Returns a tuple: (X, y, X_test, y_test)
        """
        pass


class Evaluator(abc.ABC):
    """
    Evaluates a model
    and logs the results
    """
    @abc.abstractmethod
    def evaluate(self, model, X, y, X_test, y_test):
        pass


class Experiment(abc.ABC):

    def __init__(
            self,
            data_loader: DatasetLoader,
            persistence: Persistence,
            evaluator: Evaluator,
            balanced=True):
        self.persistence = persistence
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.data_loader = data_loader
        self.balanced = balanced

    @abc.abstractmethod
    def run():
        pass
