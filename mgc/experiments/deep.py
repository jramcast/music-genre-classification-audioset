import os
import logging
import keras
import tensorflow as tf
from keras.layers import (Input, Dense, BatchNormalization, Dropout,
                          Activation, Flatten)

from mgc import metrics, audioset


def run():
    """
    Runs the experiment
    """
    train()
    print('Done. Check the logs/ folder for results')


def train():

    epochs = 100

    datadir = os.environ.get(
        'DATA_DIR',
        './downloads/audioset/audioset_v1_embeddings/bal_train'
    )
    datadir = os.path.abspath(datadir)
    ids, X, y = audioset.load_music_genre_subset_as_tensor(datadir)
    model = build_model(X)
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
                  loss='binary_crossentropy',
                  target_tensors=[y])

    # STEPS_PER_EPOCH= SUM_OF_ALL_DATASAMPLES / BATCHSIZE
    STEPS_PER_EPOCH = 10
    datadir_test = os.environ.get(
        'DATA_DIR_TEST',
        './downloads/audioset/audioset_v1_embeddings/eval'
    )
    datadir_test = os.path.abspath(datadir_test)
    _, X_test, y_test = audioset.load_music_genre_subset_as_tensor(
        datadir_test, audioset.MUSIC_GENRE_CLASSES)
    metrics_cb = Metrics(X_test, y_test)
    model.fit(
        epochs=epochs,
        callbacks=[metrics_cb],
        steps_per_epoch=STEPS_PER_EPOCH)


class Metrics(keras.callbacks.Callback):

    def __init__(self, X, y_true):
        self.X = X
        self.y_true = y_true
        return super().__init__()

    def on_train_begin(self, logs={}):
        self.data = []

    def on_train_end(self, logs=None):
        logging.info("- FINAL statistics:")
        y_pred = self.model.predict(self.X, steps=1)
        metrics.get_avg_stats(
            y_pred,
            self.y_true.eval(session=tf.keras.backend.get_session()),
            audioset.MUSIC_GENRE_CLASSES,
            num_classes=10
        )

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 == 0:
            logging.info('Epoch {} stats'.format(epoch))
            y_pred = self.model.predict(self.X, steps=1)
            metrics.get_avg_stats(
                y_pred,
                self.y_true.eval(session=tf.keras.backend.get_session())
            )


def build_model(features,
                num_units=100,
                classes_num=len(audioset.MUSIC_GENRE_CLASSES)):
    drop_rate = 0.5

    # The input layer flattens the 10 seconds as a single dimension of 1280
    input_layer = Input(tensor=features, name="model_input_tensor")
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
