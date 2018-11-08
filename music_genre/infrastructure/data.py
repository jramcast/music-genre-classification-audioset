import os
import sys
import tensorflow as tf


class AudiosetDataLoader:

    def __init__(self, datadir):
        self.filenames = list(self._discover_filenames(datadir))

    def load(self):

        iterator  = self._create_data_load_graph()

        total = 0

        with tf.Session() as sess:

            while True:
                try:
                    next_element = iterator.get_next()
                    (video_id, features, labels) = sess.run(next_element)
                    total +=1
                    print(video_id)
                    # print(features)
                    # print(labels)
                    print(total)
                    print('----------------------------------------')
                except tf.errors.OutOfRangeError:
                    break


        # (video_ids, labels, decoded_features) = self._create_data_load_graph()

        # extracted_ids = []
        # extracted_labels = []
        # extracted_features = []

        # # run the tensorflow session
        # with tf.Session() as sess:
        #     # Initialize all global and local variables
        #     init_op = tf.group(
        #         tf.global_variables_initializer(),
        #         tf.local_variables_initializer()
        #     )
        #     sess.run(init_op)

        #     # Create a coordinator and run all QueueRunner objects
        #     # Coordinators are necessary to run queues
        #     coord = tf.train.Coordinator()

        #     threads = tf.train.start_queue_runners(coord=coord)
        #     n = 0

        #     try:
        #         while not coord.should_stop():
        #             vid, vlabels, vfeatures = sess.run(
        #                 (video_ids, labels, decoded_features)
        #             )
        #             extracted_ids.append(vid)
        #             extracted_labels.append(vlabels)
        #             extracted_features.append(vfeatures)
        #             n = n + 1
        #             print('Done {}'.format(n))

        #     except tf.errors.OutOfRangeError:
        #         # When done, ask the threads to stop
        #         print('Done reading')
        #     finally:
        #         # Stop the threads
        #         coord.request_stop()

        #     coord.join(threads)
        #     sess.close()

        # print('Num videos {}'.format(len(extracted_ids)))
        # print('Num labels {}'.format(len(extracted_labels)))
        # print('Num features {}'.format(len(extracted_features)))

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
        dataset = dataset.batch(4096)
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
        # NUM_CLASSES = 527
        # labels = tf.one_hot(context['labels'], NUM_CLASSES)
        return (video_id, features, labels)


    def _discover_filenames(self, datadir):
        for root, dirs, files in os.walk(datadir):
            for filename in files:
                yield os.path.join(datadir, filename)
