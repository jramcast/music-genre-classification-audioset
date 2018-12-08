import logging
from sklearn.base import BaseEstimator
from keras.models import Model
from mgc.experiments.base import Evaluator
from mgc.metrics import MetricsLogger
from mgc import audioset


class SklearnModelEvaluator(Evaluator):

    def __init__(self, classmetrics_filepath):
        self.classmetrics_filepath = classmetrics_filepath

    def evaluate(self, model: BaseEstimator, X, y, X_test, y_test):
        metrics_logger = MetricsLogger(
            classes=audioset.ontology.MUSIC_GENRE_CLASSES,
            classsmetrics_filepath=self.classmetrics_filepath,
            show_top_classes=25,
            class_sort_key='ap'
        )

        logging.info('---- Train stats ----')
        predictions = model.predict(X)
        metrics_logger.log(predictions, y)

        logging.info('---- Test stats ----')
        predictions = model.predict(X_test)
        metrics_logger.log(predictions, y_test, show_classes=True)


class KerasModelEvaluator(Evaluator):

    def __init__(self, classmetrics_filepath):
        self.classmetrics_filepath = classmetrics_filepath

    def evaluate(self, model: Model, X, y, X_test, y_test):
        metrics_logger = MetricsLogger(
            classes=audioset.ontology.MUSIC_GENRE_CLASSES,
            classsmetrics_filepath=self.classmetrics_filepath,
            show_top_classes=25,
            class_sort_key='ap'
        )

        logging.info('---- Test stats ----')
        predictions = model.predict(X_test)
        metrics_logger.log(predictions, y_test, show_classes=True)
