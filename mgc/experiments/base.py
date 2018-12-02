import abc


class Experiment(abc.ABC):

    @abc.abstractmethod
    def run():
        pass
