import os
import tensorflow as tf
import numpy as np


class AudiosetDataLoader:

    def __init__(self, datadir):
        self.filenames = list(self._discover_filenames(datadir))

    def load(self):

        iterator = self._create_data_load_graph()

        with tf.Session() as sess:

            ids = np.array([])
            X = np.ndarray((0, 10, 128))
            y = np.ndarray((0, 527))

            while True:
                try:
                    next_element = iterator.get_next()
                    (ids_batch, features_batch, labels_batch) = sess.run(
                        next_element)
                    # print(ids_batch.shape)
                    # print('X', features_batch.shape)
                    # print('y', labels_batch.dense_shape)
                    # sdf = 0
                    # for idx in labels_batch.indices:
                    #     # print(batch_idx)
                    #     print(idx, labels_batch[idx.astype(int)])
                    print(labels_batch.indices.shape)
                    labels_batch = tf.sparse.to_dense(labels_batch, default_value=-1).eval()
                    print(labels_batch)
                    print(labels_batch.shape)
                    print('++++++++' * 2)

                    # for i in range(len(labels_batch)):

                    ids = np.concatenate([ids, ids_batch])
                    X = np.concatenate([X, features_batch], axis=0)
                    # y = np.concatenate([y, labels_batch], axis=0)



                except tf.errors.OutOfRangeError:
                    break
                    # raise StopIteration

            return ids, X, y

    def _create_data_load_graph(self):

        def only_10_seconds(video_id, features, labels):
            shape = tf.shape(features)
            res = tf.equal(shape[0], 10)
            # tf.print(shape[0], output_stream=sys.stdout)
            # tf.print(res, output_stream=sys.stdout)
            return res

        # Create a list of filenames and pass it to a queue
        dataset = tf.data.TFRecordDataset(self.filenames)
        dataset = dataset.map(self._read_record, num_parallel_calls=8)
        dataset = dataset.filter(only_10_seconds)
        dataset = dataset.batch(2048)
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
