import os
import time
import logging
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB


from mgc import metrics, audioset
from mgc.audioset import transform


def run():
    """
    Runs the experiment
    """
    X, y, X_test, y_test = load_data()
    classifier = train(X, y)
    evaluate(classifier, X, y, X_test, y_test)
    print('Done. Check the logs/ folder for results')


def load_data():
    datadir = os.environ.get(
        'DATA_DIR',
        './downloads/audioset/audioset_v1_embeddings/bal_train'
    )
    datadir = os.path.abspath(datadir)
    datadir_test = os.environ.get(
        'DATA_DIR_TEST',
        './downloads/audioset/audioset_v1_embeddings/eval'
    )
    datadir_test = os.path.abspath(datadir_test)

    ids, X, y = audioset.load_music_genre_subset_as_numpy(datadir)
    X = transform.flatten_features(X)

    _, X_test, y_test = audioset.load_music_genre_subset_as_numpy(datadir_test)
    X_test = transform.flatten_features(X_test)
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
        audioset.MUSIC_GENRE_CLASSES,
        num_classes=10
    )
