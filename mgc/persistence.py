import logging
from joblib import dump, load
from sklearn.base import BaseEstimator
from keras.models import Model
from mgc.experiments.base import Persistence


class SklearnModelPersistence(Persistence):

    def __init__(self, filepath):
        self.filepath = filepath

    def save(self, model: BaseEstimator):
        dump(model, self.filepath)
        logging.info('Model saved to {}'.format(self.filepath))

    def load(self) -> BaseEstimator:
        return load(self.filepath)


class KerasModelPersistence(Persistence):

    def __init__(self, filepath):
        self.filepath = filepath

    def save(self, model: Model):
        model.save_weights(self.filepath)
        logging.info('Model weights saved to {}'.format(self.filepath))

    def load(self, model) -> Model:
        model.load_weights(self.filepath)
        return model
