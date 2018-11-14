import os
import time
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB


from mgc import metrics, audioset


def run():
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

    print(X_train.shape)
    print(y_train.shape)

    classifier = OneVsRestClassifier(GaussianNB(), n_jobs=-1)

    print('Training {}...'.format(classifier))
    start_time = time.time()

    classifier.fit(X_train, y_train)
    print('Training data time: {:.3f} s'.format(time.time() - start_time))

    print('---- Train stats ----')
    predictions = classifier.predict(X_train)
    metrics.get_avg_stats(predictions, y_train)

    print('---- Validation stats ----')
    predictions = classifier.predict(X_validate)
    mAP, mAUC, d_prime = metrics.get_avg_stats(
        predictions,
        y_validate,
        audioset.MUSIC_GENRE_CLASSES
    )
