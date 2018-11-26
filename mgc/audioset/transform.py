import numpy as np
import tensorflow as tf
from mgc.audioset.ontology import MUSIC_GENRE_CLASSES


def subset_by_class(X, y, classes=[]):
    classes_ids = [c['index'] for c in classes]
    # Select only samples that have any of the classes active
    sample_indexes = np.unique(np.nonzero(y[:, classes_ids])[0])
    filtered_X = X[sample_indexes, :]
    filtered_y = y[sample_indexes, :]
    filtered_y = take_y_for_classes(filtered_y, classes)
    return filtered_X, filtered_y


def take_y_for_classes(y, classes=[]):
    classes_ids = [c['index'] for c in classes]
    return y[:, classes_ids]


def flatten_features(X: np.array) -> np.array:
    '''
    Flattens a (num_samples x 10 x 128) array to (num_samples x 1280).
    Audioset provides 128 features per second, with 10 seconds per sample.
    Use this method when you need a single dimension of features.
    '''
    return np.array(X).reshape(-1, 1280)


def tensor_to_numpy(ids_tensor, X_tensor, y_tensor):
    with tf.Session() as sess:
        ids = np.array([])
        X = np.ndarray((0, 10, 128))
        y = np.ndarray((0, len(MUSIC_GENRE_CLASSES)))
        while True:
            try:
                (ids_batch, features_batch, labels_batch) = sess.run(
                    (ids_tensor, X_tensor, y_tensor)
                )
                ids = np.concatenate([ids, ids_batch])
                X = np.concatenate([X, features_batch], axis=0)
                y = np.concatenate([y, labels_batch], axis=0)
            except tf.errors.OutOfRangeError:
                break

    return ids, X, y
