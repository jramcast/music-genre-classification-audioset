import numpy as np


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
