import csv
import logging
import time

from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB

from mgc import audioset, metrics
from mgc.audioset.loaders import MusicGenreSubsetLoader
from mgc.audioset.transform import flatten_features, tensor_to_numpy
from mgc.experiments.base import Experiment
from mgc.metrics import MetricsLogger


class BayesExperiment(Experiment):

    def run(self):
        '''
        Runs the experiment
        '''
        X, y, X_test, y_test = self.load_data()
        classifier = self.train(X, y)
        self.evaluate(classifier, X, y, X_test, y_test)
        print('Done. Check the logs/ folder for results')

    def load_data(self):
        loader = MusicGenreSubsetLoader(self.datadir, repeat=False)
        if self.balanced:
            ids, X, y = loader.load_bal()
        else:
            ids, X, y = loader.load_unbal()
        ids, X, y = tensor_to_numpy(ids, X, y)
        X = flatten_features(X)

        ids_test, X_test, y_test = loader.load_eval()
        _, X_test, y_test = tensor_to_numpy(ids_test, X_test, y_test)
        X_test = flatten_features(X_test)

        return X, y, X_test, y_test

    def train(self, X, y):
        logging.info('Training dataset X shape: %s', X.shape)
        logging.info('Training dataset y shape: %s', y.shape)

        classifier = OneVsRestClassifier(GaussianNB(), n_jobs=4)

        logging.info('Training...')
        start_time = time.time()

        classifier.fit(X, y)
        logging.info('Training done: {:.3f} s'.format(time.time() - start_time))
        return classifier

    def evaluate(self, classifier, X, y, X_test, y_test):

        metrics_logger = MetricsLogger(
            classes=audioset.ontology.MUSIC_GENRE_CLASSES,
            classsmetrics_filepath=self.classmetrics_filepath,
            show_top_classes=25,
            class_sort_key='ap'
        )

        logging.info('---- Train stats ----')
        predictions = classifier.predict(X)
        metrics_logger.log(predictions, y)

        logging.info('---- Test stats ----')
        predictions = classifier.predict(X_test)
        metrics_logger.log(predictions, y_test, show_classes=True)


