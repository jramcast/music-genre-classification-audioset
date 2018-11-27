import os
import time
import logging
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB


from mgc import metrics, audioset
from mgc.audioset.transform import tensor_to_numpy, flatten_features
from mgc.audioset.loaders import MusicGenreSubsetLoader


def run():
    '''
    Runs the experiment
    '''
    X, y, X_test, y_test = load_data()
    classifier = train(X, y)
    evaluate(classifier, X, y, X_test, y_test)
    print('Done. Check the logs/ folder for results')


def load_data():
    datadir = os.environ.get(
        'DATA_DIR',
        './downloads/audioset/audioset_v1_embeddings/'
    )
    datadir = os.path.abspath(datadir)
    logging.debug('Data dir: {}'.format(datadir))

    loader = MusicGenreSubsetLoader(datadir, repeat=False)

    ids, X, y = loader.load_bal()
    ids, X, y = tensor_to_numpy(ids, X, y)
    X = flatten_features(X)

    ids_test, X_test, y_test = loader.load_eval()
    _, X_test, y_test = tensor_to_numpy(ids_test, X_test, y_test)
    X_test = flatten_features(X_test)

    return X, y, X_test, y_test


def train(X, y):
    logging.info('Training dataset X shape: %s', X.shape)
    logging.info('Training dataset y shape: %s', y.shape)

    classifier = OneVsRestClassifier(GaussianNB(), n_jobs=4)

    logging.info('Training...')
    start_time = time.time()

    classifier.fit(X, y)
    logging.info('Training done: {:.3f} s'.format(time.time() - start_time))
    return classifier


def evaluate(classifier, X, y, X_test, y_test):
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
