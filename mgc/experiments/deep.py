import logging
import keras
import tensorflow as tf
from keras.layers import (Input, Dense, BatchNormalization, Dropout,
                          Activation, Flatten)

from mgc import metrics, audioset
from mgc.audioset.loaders import MusicGenreSubsetLoader
from mgc.experiments.base import Experiment
from mgc.audioset.transform import tensor_to_numpy


class DeepExperiment(Experiment):

    def __init__(self, datadir, balanced=True, epochs=50, batch_size=1000):
        self.datadir = datadir
        self.balanced = balanced
        self.epochs = epochs
        self.batch_size = batch_size

    def run(self):
        """
        Runs the experiment
        """
        X, y, X_test, y_test = self.load_data()
        self.train_and_eval(X, y, X_test, y_test)
        print('Done. Check the logs/ folder for results')

    def load_data(self):
        loader = MusicGenreSubsetLoader(self.datadir, repeat=True)

        if self.balanced:
            ids, X, y = loader.load_bal()
        else:
            ids, X, y = loader.load_unbal()

        test_loader = MusicGenreSubsetLoader(
            self.datadir,
            repeat=False,
            batch_size=10
        )
        ids_test, X_test, y_test = test_loader.load_eval()

        return X, y, X_test, y_test

    def train_and_eval(self, X, y, X_test, y_test):

        model = self.build_model(X)
        model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
                      loss='binary_crossentropy',
                      target_tensors=[y])

        if self.balanced:
            aprox_total_samples = 2000
        else:
            aprox_total_samples = 200000

        metrics_cb = Metrics(X_test, y_test)
        model.fit(
            epochs=self.epochs,
            callbacks=[metrics_cb],
            steps_per_epoch=2)

        return model

    def build_model(self, X,
                    num_units=100,
                    classes_num=len(audioset.ontology.MUSIC_GENRE_CLASSES)):
        drop_rate = 0.5

        # The input layer flattens the 10 seconds as a single dimension of 1280
        input_layer = Input(tensor=X, name="model_input_tensor")
        reshape = Flatten(input_shape=(-1, 10, 128))(input_layer)

        a1 = Dense((num_units))(reshape)
        a1 = BatchNormalization()(a1)
        a1 = Activation('relu')(a1)
        a1 = Dropout(drop_rate)(a1)

        a2 = Dense(num_units)(a1)
        a2 = BatchNormalization()(a2)
        a2 = Activation('relu')(a2)
        a2 = Dropout(drop_rate)(a2)

        output_layer = Dense(classes_num, activation='sigmoid')(reshape)

        # Build model
        return keras.models.Model(inputs=input_layer, outputs=output_layer)


class Metrics(keras.callbacks.Callback):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        return super().__init__()

    def on_train_begin(self, logs={}):
        self.data = []

    # def on_train_end(self, logs=None):
    #     logging.info("- FINAL statistics:")
    #     y_pred = self.model.predict(self.X, steps=1)
    #     y_true = self.y.eval(session=tf.keras.backend.get_session())
    #     metrics.get_avg_stats(
    #         y_pred,
    #         y_true,
    #         audioset.MUSIC_GENRE_CLASSES,
    #         num_classes=10
    #     )

    def on_epoch_end(self, epoch, logs={}):
        # if epoch % 10 == 0:
        logging.info('Epoch {} stats'.format(epoch))
        y_pred = self.model.predict(self.X, steps=1)
        y_true = self.y.eval(session=tf.keras.backend.get_session())
        metrics.get_avg_stats(
            y_pred,
            y_true
        )
