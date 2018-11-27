import os
from typing import List, Tuple
import tensorflow as tf
from mgc.audioset.ontology import MUSIC_GENRE_CLASSES, NUM_TOTAL_CLASSES


class MusicGenreSubsetLoader:
    '''
    Loads the subset of music genre samples from Audioset
    '''

    def __init__(self, datadir: List[str], repeat=True, batch_size=1000):
        self.datadir = datadir
        self.class_indexes = [c['index'] for c in MUSIC_GENRE_CLASSES]
        self.repeat = repeat
        self.batch_size = batch_size

    def load_bal(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        ids, X, y = self._load('bal_train')
        return ids, X, y

    def load_unbal(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        ids, X, y = self._load('unbal_train')
        return ids, X, y

    def load_eval(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        ids, X, y = self._load('eval')
        return ids, X, y

    def _load(self, splitname: str) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # create the dataset
        filenames = list(self._discover_filenames(splitname))
        dataset = tf.data.TFRecordDataset(filenames)
        # Parse every sample of the dataset
        dataset = dataset.map(self._read_record, num_parallel_calls=8)
        # Filter only certain data
        dataset = dataset.filter(self._only_music_genre_samples)
        dataset = dataset.filter(self._only_10_second_samples)
        # Set the batchsize
        dataset = dataset.batch(self.batch_size)
        # Start over when we are finished reading the dataset
        if self.repeat:
            dataset = dataset.repeat()
        # Create an iterator
        iterator = dataset.make_one_shot_iterator()
        # Create your tf representation of the iterator
        video_id, features, labels = iterator.get_next()
        # Set a fixed shape of features (the first dimension is the batch)
        features = tf.reshape(features, [-1, 10, 128])
        # Create a one hot array for multilabel classification
        labels = tf.sparse_to_indicator(labels, NUM_TOTAL_CLASSES)
        # Only take the required music genre classes
        labels = tf.gather(labels, self.class_indexes, axis=1)
        # cast to a supported data type
        labels = tf.cast(labels, tf.float32)
        # return ids, features and labels
        return video_id, features, labels

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

        video_id = context['video_id']
        labels = context['labels']
        # Convert the data from string back to the numbers
        features = tf.decode_raw(features['audio_embedding'], tf.uint8)
        # Cast features into float32
        features = tf.cast(features, tf.float32)
        # Reshape features into original size
        features = tf.reshape(features, [-1, 128])

        return video_id, features, labels

    def _discover_filenames(self, splitname):
        datadir = os.path.join(self.datadir, splitname)
        for root, dirs, files in os.walk(datadir):
            for filename in files:
                yield os.path.join(datadir, filename)

    def _only_10_second_samples(self, video_id, features, labels):
        shape = tf.shape(features)
        res = tf.equal(shape[0], 10)
        return res

    def _only_music_genre_samples(self, video_id, features, labels):
        # we convert 1-dimension arrays to 2-dimension arrays
        # because set_intersection requires at least 2 dimensions
        wanted = tf.constant(self.class_indexes)[None, :]
        # labels are int64 and wanted values are int32 so we need to cast them
        present = tf.cast(labels.values, tf.int32)[None, :]
        intersection = tf.sets.set_intersection(wanted, present)
        intersection_not_empty = tf.not_equal(tf.size(intersection), 0)
        return intersection_not_empty
