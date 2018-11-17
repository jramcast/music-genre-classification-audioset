import os
import tensorflow as tf
import numpy as np
import logging
from sklearn import preprocessing


class AudiosetDataLoader:

    def __init__(self, datadir, class_indexes):
        self.filenames = list(self._discover_filenames(datadir))
        self.class_indexes = class_indexes

    def load(self):

        iterator = self._create_data_load_graph()

        with tf.Session() as sess:

            ids = np.array([])
            X = np.ndarray((0, 10, 128))
            y = np.ndarray((0, 527))

            while True:
                logging.debug('Loading')
                try:
                    next_element = iterator.get_next()
                    (ids_batch, features_batch, labels_batch) = sess.run(
                        next_element)
                    labels_batch = tf.sparse.to_dense(labels_batch).eval()
                    # only trim 0s from the back(b) of the array
                    labels_batch = np.array(
                        [np.trim_zeros(row, trim='b') for row in labels_batch]
                    )
                    lb = preprocessing.MultiLabelBinarizer(classes=range(527))
                    labels_batch = lb.fit_transform(labels_batch)
                    logging.debug('Loaded')
                    ids = np.concatenate([ids, ids_batch])
                    X = np.concatenate([X, features_batch], axis=0)
                    y = np.concatenate([y, labels_batch], axis=0)
                    logging.debug('Added to numpy array %s', X.shape)
                except tf.errors.OutOfRangeError:
                    break
                    # raise StopIteration

            return ids, X, y

    def _create_data_load_graph(self):

        def only_10_seconds(video_id, features, labels):
            shape = tf.shape(features)
            res = tf.equal(shape[0], 10)
            return res

        def only_samples_for_classes(video_id, features, labels):
            # we convert 1-dimension arrays to 2-dimension arrays
            # because set_intersection requires at least 2 dimensions
            wanted = tf.constant(self.class_indexes)[None, :]
            # labels are int64 and wanted values are int32 so we need to cast them
            present = tf.cast(labels.values, tf.int32)[None, :]
            intersection = tf.sets.set_intersection(wanted, present)
            intersection_not_empty = tf.not_equal(tf.size(intersection), 0)
            return intersection_not_empty

        # Create a list of filenames and pass it to a queue
        dataset = tf.data.TFRecordDataset(self.filenames)
        dataset = dataset.map(self._read_record, num_parallel_calls=8)
        dataset = dataset.filter(only_samples_for_classes)
        dataset = dataset.filter(only_10_seconds)
        dataset = dataset.batch(200000)
        iterator = dataset.make_one_shot_iterator()
        return iterator

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
        return (video_id, features, labels)

    def _discover_filenames(self, datadir):
        for root, dirs, files in os.walk(datadir):
            for filename in files:
                yield os.path.join(datadir, filename)
