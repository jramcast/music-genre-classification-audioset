import os
import time
import logging
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB


from mgc import metrics, audioset


def run():
    """
    Runs the experiment
    """
    datadir = os.environ.get(
        'DATA_DIR',
        './downloads/audioset/audioset_v1_embeddings/bal_train'
    )
    datadir = os.path.abspath(datadir)

    X, y = audioset.load_music_genre_instances(datadir)

    (X_train,
     X_validate,
     y_train,
     y_validate) = train_test_split(X, y, test_size=0.3, random_state=42)

    logging.info('Training dataset X shape: %s', X_train.shape)
    logging.info('Training dataset y shape: %s', y_train.shape)

    classifier = OneVsRestClassifier(GaussianNB(), n_jobs=6)

    logging.info('Training started')
    start_time = time.time()

    classifier.fit(X_train, y_train)
    logging.info('Training finished: {:.3f} s'.format(time.time() - start_time))

    print('---- Train stats ----')
    predictions = classifier.predict(X_train)
    metrics.get_avg_stats(predictions, y_train)

    print('---- Validation stats ----')
    predictions = classifier.predict(X_validate)
    mAP, mAUC, d_prime = metrics.get_avg_stats(
        predictions,
        y_validate,
        audioset.MUSIC_GENRE_CLASSES,
        num_classes=10
    )
