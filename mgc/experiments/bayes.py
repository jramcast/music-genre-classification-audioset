import time
import logging
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB


from mgc import metrics, audioset
from mgc.audioset.transform import tensor_to_numpy, flatten_features
from mgc.audioset.loaders import MusicGenreSubsetLoader
from mgc.experiments.base import Experiment


class BayesExperiment(Experiment):

    def __init__(self, datadir, balanced=True):
        self.datadir = datadir
        self.balanced = balanced

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
        logging.info('---- Train stats ----')
        predictions = classifier.predict(X)
        metrics.get_avg_stats(predictions, y)

        logging.info('---- Test stats ----')
        predictions = classifier.predict(X_test)
        mAP, mAUC, d_prime = metrics.get_avg_stats(
            predictions,
            y_test,
            audioset.ontology.MUSIC_GENRE_CLASSES,
            num_classes=10
        )
