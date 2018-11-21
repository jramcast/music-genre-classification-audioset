import os
import logging
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
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

    sess = K.get_session()
    epochs = 5
    batch_size = 10000

    datadir = os.environ.get(
        'DATA_DIR',
        './downloads/audioset/audioset_v1_embeddings/bal_train'
    )
    datadir = os.path.abspath(datadir)
    video_id, features, labels = audioset.load_music_genre_instances_as_tf(datadir)

    model = build_model(features)
    model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-3, decay=1e-5),
                  loss='binary_crossentropy',
                  target_tensors=[labels])

    # Fit the
    model.fit(
        epochs=epochs,
        steps_per_epoch=int(np.ceil(2000/ float(batch_size))))



def build_model(features, num_units=100, classes_num=527):
    drop_rate = 0.5

    # Embedded layers
    input_layer = Input(tensor=features)
    input_layer = Flatten(input_shape=(-1, 10, 128))(input_layer)

    a1 = Dense((num_units))(input_layer)
    a1 = BatchNormalization()(a1)
    a1 = Activation('relu')(a1)
    a1 = Dropout(drop_rate)(a1)

    a2 = Dense(num_units)(a1)
    a2 = BatchNormalization()(a2)
    a2 = Activation('relu')(a2)
    a2 = Dropout(drop_rate)(a2)

    output_layer = Dense(classes_num, activation='sigmoid')(a2)

    # Build model
    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def evaluate(classifier, X, y, X_test, y_test):
    logging.info('---- Train stats ----')
    predictions = classifier.predict(X)
    metrics.get_avg_stats(predictions, y)

    logging.info('---- Test stats ----')
    predictions = classifier.predict(X_test)
    mAP, mAUC, d_prime = metrics.get_avg_stats(
        predictions,
        y_test,
        audioset.MUSIC_GENRE_CLASSES,
        num_classes=10
    )


class EvaluateInputTensor(keras.callbacks.Callback):

    def __init__(self, model, X, y):
        super(EvaluateInputTensor, self).__init__()
        self.model = model
        self.X = X
        self. y = y

    def on_epoch_end(self, epoch, logs={}):
        print("- Training set statistics:")
        predictions = self.model.predict(self.X)
        audioset.metrics.get_avg_stats(predictions, self.y)
