import os
import tensorflow as tf


class AudiosetDataLoader:

    def __init__(self, datadir):
        self.filenames = self._discover_filenames(datadir)

    def load(self):
        (video_ids, labels, decoded_features) = self._create_data_load_graph()

        extracted_ids = []
        extracted_labels = []
        extracted_features = []

        # run the tensorflow session
        with tf.Session() as sess:
            # Initialize all global and local variables
            init_op = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            )
            sess.run(init_op)

            # Create a coordinator and run all QueueRunner objects
            # Coordinators are necessary to run queues
            coord = tf.train.Coordinator()

            threads = tf.train.start_queue_runners(coord=coord)
            n = 0

            try:
                while not coord.should_stop():
                    vid, vlabels, vfeatures = sess.run(
                        (video_ids, labels, decoded_features)
                    )
                    extracted_ids.append(vid)
                    extracted_labels.append(vlabels)
                    extracted_features.append(vfeatures)
                    n = n + 1
                    print('Done {}'.format(n))

            except tf.errors.OutOfRangeError:
                # When done, ask the threads to stop
                print('Done reading')
            finally:
                # Stop the threads
                coord.request_stop()

            coord.join(threads)
            sess.close()


        print('Num videos {}'.format(len(extracted_ids)))
        print('Num labels {}'.format(len(extracted_labels)))
        print('Num features {}'.format(len(extracted_features)))

    def _create_data_load_graph(self):
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer(
            self.filenames,
            num_epochs=1)

        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        # Decode the record read by the reader
        contexts, features = tf.parse_single_sequence_example(
            serialized_example,
            context_features={
                "video_id": tf.FixedLenFeature([], tf.string),
                "labels": tf.VarLenFeature(tf.int64)
            },
            sequence_features={
                'audio_embedding': tf.FixedLenSequenceFeature([], dtype=tf.string)
            }
        )

        # Convert the data from string back to the numbers
        features = tf.decode_raw(features['audio_embedding'], tf.uint8)
        # Cast features into float32
        features = tf.cast(features, tf.float32)
        # Reshape features into original size
        features = tf.reshape(features, [-1, 128])
        # Any preprocessing here ...
        video_ids = contexts['video_id']
        labels = contexts['labels']
        return (video_ids, features, labels)

    def _discover_filenames(self):
        for root, dirs, files in os.walk(self.datadir):
            for filename in files:
                yield os.path.join(self.datadir, filename)
