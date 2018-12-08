import os
import math
import tempfile

import keras
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Flatten, Input)

from mgc import audioset, metrics
from mgc.audioset.loaders import MusicGenreSubsetLoader
from mgc.audioset.transform import tensor_to_numpy
from mgc.experiments.base import Experiment
import keras.backend as K


class DeepExperiment(Experiment):

    def __init__(self, datadir, balanced=True, epochs=100, batch_size=1000):
        self.datadir = datadir
        self.balanced = balanced
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_units = 100
        self.drop_rate = 0.5
        self.weights_filepath = os.path.join(tempfile.gettempdir(), 'deep_wt.h5')

    def run(self):
        """
        Runs the experiment
        """
        X, y, X_test, y_test = self.load_data()
        self.train_and_eval(X, y, X_test, y_test)
        self.evaluate(X_test, y_test)
        print('Done. Check the logs/ folder for results')

    def load_data(self):
        loader = MusicGenreSubsetLoader(
            self.datadir,
            repeat=True,
            batch_size=self.batch_size
        )

        if self.balanced:
            ids, X, y = loader.load_bal()
        else:
            ids, X, y = loader.load_unbal()

        test_loader = MusicGenreSubsetLoader(
            self.datadir,
            repeat=False,
            batch_size=self.batch_size
        )
        ids_test, X_test, y_test = test_loader.load_eval()

        ids_test, X_test, y_test = tensor_to_numpy(ids_test, X_test, y_test)

        return X, y, X_test, y_test

    def train_and_eval(self, X, y, X_test, y_test):

        model = self.build_model(X)
        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss='binary_crossentropy',
            metrics=['top_k_categorical_accuracy'],
            target_tensors=[y]
        )

        if self.balanced:
            total_samples = 2490
        else:
            total_samples = 200000

        model.fit(
            epochs=self.epochs,
            steps_per_epoch=math.ceil(total_samples/self.batch_size))

        # Save the model weights.
        model.save_weights(self.weights_filepath)

        # Clean up the TF session.
        K.clear_session()

    def evaluate(self, X_test, y_test):
        # Second session to test loading trained model without tensors.
        input_layer = Input(shape=(10, 128))
        output_layer = self.define_layers(input_layer)
        test_model = keras.models.Model(
            inputs=input_layer,
            outputs=output_layer
        )
        test_model.load_weights(self.weights_filepath)
        test_model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss='binary_crossentropy'
        )

        y_pred = test_model.predict(X_test)

        metrics.get_avg_stats(
            y_pred,
            y_test,
            audioset.ontology.MUSIC_GENRE_CLASSES,
            num_classes=25
        )

    def define_layers(self, input):
        # The input layer flattens the 10 seconds as a single dimension of 1280
        reshape = Flatten(input_shape=(-1, 10, 128))(input)

        a1 = Dense((self.num_units))(reshape)
        a1 = BatchNormalization()(a1)
        a1 = Activation('relu')(a1)
        a1 = Dropout(self.drop_rate)(a1)

        a2 = Dense(self.num_units)(a1)
        a2 = BatchNormalization()(a2)
        a2 = Activation('relu')(a2)
        a2 = Dropout(self.drop_rate)(a2)

        classes_num = len(audioset.ontology.MUSIC_GENRE_CLASSES)
        predictions = Dense(classes_num, activation='sigmoid')(a2)
        return predictions

    def build_model(self, X):
        input_layer = Input(tensor=X, name="model_input_tensor")
        output_layer = self.define_layers(input_layer)

        # Build model
        return keras.models.Model(inputs=input_layer, outputs=output_layer)
