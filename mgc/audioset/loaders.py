import os
import abc
from typing import List, Dict, Tuple
import tensorflow as tf
import numpy as np
import logging


BATCH_SIZE = 1000
NUM_CLASSES = 527


class AudiosetLoader(abc.ABC):

    def __init__(self, datadir: List[str], class_indexes: Dict):
        self.filenames = list(self._discover_filenames(datadir))
        self.class_indexes = class_indexes

    @abc.abstractmethod
    def load() -> Tuple:
        pass

    def _init_dataset(self):
        dataset = tf.data.TFRecordDataset(self.filenames)
        # Parse every sample of the dataset
        dataset = dataset.map(self._read_record, num_parallel_calls=8)
        # Filter only certain data
        dataset = dataset.filter(self._only_samples_for_classes)
        dataset = dataset.filter(self._only_10_seconds)
        # Set the batchsize
        dataset = dataset.batch(BATCH_SIZE)
        return dataset

    def _only_10_seconds(self, video_id, features, labels):
        shape = tf.shape(features)
        res = tf.equal(shape[0], 10)
        return res

    def _only_samples_for_classes(self, video_id, features, labels):
        # we convert 1-dimension arrays to 2-dimension arrays
        # because set_intersection requires at least 2 dimensions
        wanted = tf.constant(self.class_indexes)[None, :]
        # labels are int64 and wanted values are int32 so we need to cast them
        present = tf.cast(labels.values, tf.int32)[None, :]
        intersection = tf.sets.set_intersection(wanted, present)
        intersection_not_empty = tf.not_equal(tf.size(intersection), 0)
        return intersection_not_empty

    def _read_record(self, serialized_example):
        # Decode the record read by the reader
        context, features = tf.parse_single_sequence_example(
            serialized_example,
            context_features={
                "video_id": tf.FixedLenFeature([], tf.string),
                "labels": tf.VarLenFeature(tf.int64)
            },
            sequence_features={
                'audio_embedding': tf.FixedLenSequenceFeature(
                    [], dtype=tf.string)
            }
        )

        # Convert the data from string back to the numbers
        features = tf.decode_raw(features['audio_embedding'], tf.uint8)
        # Cast features into float32
        features = tf.cast(features, tf.float32)
        # Reshape features into original size
        features = tf.reshape(features, [-1, 128])
        # Any preprocessing here ...
        video_id = context['video_id']
        labels = context['labels']
        return video_id, features, labels

    def _discover_filenames(self, datadir):
        for root, dirs, files in os.walk(datadir):
            for filename in files:
                yield os.path.join(datadir, filename)


class TensorLoader(AudiosetLoader):
    '''
    Reads the dataset as tensorflow tensors
    '''

    def __init__(self, datadir, class_indexes, repeat=True):
        self.repeat = repeat
        return super().__init__(datadir, class_indexes)

    def load(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        dataset = self._init_dataset()

        # Start over when we are finished reading the dataset
        if self.repeat:
            dataset = dataset.repeat()

        # Create an iterator
        iterator = dataset.make_one_shot_iterator()

        # Create your tf representation of the iterator
        video_id, features, labels = iterator.get_next()

        # Set a fixed shape of features
        features = tf.reshape(features, [-1, 10, 128])

        # Create a one hot array for multilabel classification
        labels = tf.sparse_to_indicator(labels, NUM_CLASSES)

        # Only take the required classes
        labels = tf.gather(labels, self.class_indexes, axis=1)

        # cast to a supported data type
        labels = tf.cast(labels, tf.float32)

        return video_id, features, labels


class NPArrayLoader(AudiosetLoader):
    '''
    Reads the dataset as numpy arrays
    '''

    def __init__(self, datadir, class_indexes):
        self.filenames = list(self._discover_filenames(datadir))
        self.class_indexes = class_indexes

    def load(self) -> Tuple[np.array, np.array, np.array]:
        dataset = self._init_dataset()
        iterator = dataset.make_one_shot_iterator()
        video_id, features, labels = iterator.get_next()
        labels = tf.sparse_to_indicator(labels, NUM_CLASSES)
        labels = tf.gather(labels, self.class_indexes, axis=1)
        labels = tf.cast(labels, tf.float32)

        with tf.Session() as sess:

            ids = np.array([])
            X = np.ndarray((0, 10, 128))
            y = np.ndarray((0, len(self.class_indexes)))

            while True:
                logging.debug('Loading')
                try:
                    (ids_batch, features_batch, labels_batch) = sess.run((
                        video_id, features, labels
                    ))
                    # labels_batch = tf.sparse.to_dense(labels_batch).eval()
                    # # only trim 0s from the back(b) of the array
                    # labels_batch = np.array(
                    #     [np.trim_zeros(row, trim='b') for row in labels_batch]
                    # )
                    # lb = preprocessing.MultiLabelBinarizer(classes=range(NUM_CLASSES))
                    # labels_batch = lb.fit_transform(labels_batch)
                    # logging.debug('Loaded')
                    ids = np.concatenate([ids, ids_batch])
                    X = np.concatenate([X, features_batch], axis=0)
                    y = np.concatenate([y, labels_batch], axis=0)
                    # logging.debug('Added to numpy array %s', X.shape)
                except tf.errors.OutOfRangeError:
                    break
                    # raise StopIteration

            return ids, X, y
