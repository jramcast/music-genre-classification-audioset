import logging
from typing import Tuple

import numpy as np
import tensorflow as tf

from mgc.audioset.loaders import MusicGenreSubsetLoader
from mgc.audioset.transform import flatten_features, tensor_to_numpy
from mgc.experiments.base import DatasetLoader


class NumpyMusicGenreSetLoader(DatasetLoader):

    def __init__(self, datadir):
        self.datadir = datadir

    def load(
            self,
            balanced=False,
            repeat=False) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Returns a tuple including train and test sets
        as np arrays: (X, y, X_test, y_test)
        """
        loader = MusicGenreSubsetLoader(
            self.datadir,
            repeat=repeat
        )

        if balanced:
            ids, X, y = loader.load_bal()
        else:
            ids, X, y = loader.load_unbal()

        ids, X, y = tensor_to_numpy(ids, X, y)
        X = flatten_features(X)

        ids_test, X_test, y_test = loader.load_eval()
        _, X_test, y_test = tensor_to_numpy(ids_test, X_test, y_test)
        X_test = flatten_features(X_test)

        logging.info('Training dataset X shape: %s', X.shape)
        logging.info('Training dataset y shape: %s', y.shape)

        return X, y, X_test, y_test


class TFMusicGenreSetLoader(DatasetLoader):
    """
    Loads data set as tensorflow tensors
    """

    def __init__(self, datadir):
        self.datadir = datadir

    def load(
            self,
            batch_size,
            balanced=False,
            repeat=False) -> Tuple[tf.Tensor, tf.Tensor, np.array, np.array]:
        """
        Returns a tuple including train and test (X, y, X_test, y_test)
        Train set is loaded as TF Tensors
        Test set is loaded as np arrays
        """
        loader = MusicGenreSubsetLoader(
            self.datadir,
            repeat=True,
            batch_size=batch_size
        )

        if balanced:
            ids, X, y = loader.load_bal()
        else:
            ids, X, y = loader.load_unbal()

        test_loader = MusicGenreSubsetLoader(
            self.datadir,
            repeat=False,
            batch_size=2500
        )
        ids_test, X_test, y_test = test_loader.load_eval()
        ids_test, X_test, y_test = tensor_to_numpy(ids_test, X_test, y_test)

        return X, y, X_test, y_test
