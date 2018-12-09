import math
import logging
import time

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Flatten, Input, LSTM, concatenate)

from mgc import audioset
from mgc.experiments.base import (DatasetLoader, Evaluator, Experiment,
                                  Persistence)


class LSTMExperiment(Experiment):

    def __init__(
            self,
            data_loader: DatasetLoader,
            persistence: Persistence,
            evaluator: Evaluator,
            balanced=True,
            epochs=10,
            batch_size=5000,
            num_units=768,
            drop_rate=0.5
    ):
        super().__init__(data_loader, persistence, evaluator, balanced)
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_units = num_units
        self.drop_rate = drop_rate

    def run(self):
        """
        Runs the experiment
        """
        X, y, X_test, y_test = self.data_loader.load(
            self.batch_size,
            self.balanced,
            repeat=True
        )
        model = self.train(X, y)
        self.persistence.save(model)
        # Clean up the TF session before evaluation.
        K.clear_session()
        self.evaluate(X, y, X_test, y_test)
        print('Done. Check the logs/ folder for results')

    def train(self, X, y):
        # Build model
        inputs = Input(tensor=X, name="model_input_tensor")
        outputs = self.define_layers(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        logging.info('Model: %s', model)

        logging.info('Training...')
        start_time = time.time()

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

        logging.info(
            'Training done: {:.3f} s'.format(time.time() - start_time)
        )
        return model

    def evaluate(self, X, y, X_test, y_test):
        # Second session to test loading trained model without tensors.
        inputs = Input(shape=(10, 128))
        outputs = self.define_layers(inputs)
        test_model = Model(inputs=inputs, outputs=outputs)
        test_model = self.persistence.load(test_model)
        test_model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss='binary_crossentropy'
        )
        self.evaluator.evaluate(test_model, X, y, X_test, y_test)

    def define_layers(self, inputs):
        # The input layer flattens the 10 seconds as a single dimension of 1280
        # reshape = Flatten(input_shape=(-1, 10, 128))(inputs)

        l1 = LSTM(1280, return_sequences=False)(inputs)
        l1 = BatchNormalization()(l1)
        l1 = Activation('relu')(l1)
        l1 = Dropout(self.drop_rate)(l1)

        classes_num = len(audioset.ontology.MUSIC_GENRE_CLASSES)
        predictions = Dense(classes_num, activation='sigmoid')(l1)
        return predictions
