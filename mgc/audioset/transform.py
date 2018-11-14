import numpy as np


def subset_by_class(X, y, classes=[]):
    classes_ids = [c['index'] for c in classes]
    # Select only samples that have any of the classes active
    sample_indexes = np.unique(np.nonzero(y[:, classes_ids])[0])
    filtered_X = X[sample_indexes, :]
    filtered_y = y[sample_indexes, :]
    filtered_y = take_y_for_class(filtered_y, classes)
    return filtered_X, filtered_y


def take_y_for_class(y, classes=[]):
    classes_ids = [c['index'] for c in classes]
    return y[:, classes_ids]
