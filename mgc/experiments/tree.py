import logging
import time

from sklearn.tree import DecisionTreeClassifier

from mgc.experiments.base import Experiment


class DecisionTreeExperiment(Experiment):

    def run(self):
        '''
        Runs the experiment
        '''
        X, y, X_test, y_test = self.data_loader.load(
            self.balanced,
            repeat=False
        )
        model = self.train(X, y)
        self.persistence.save(model)
        self.evaluator.evaluate(model, X, y, X_test, y_test)
        print('Done. Check the logs/ folder for results')

    def train(self, X, y):
        model = DecisionTreeClassifier()
        logging.info('Model: %s', model)

        logging.info('Training...')
        start_time = time.time()

        model.fit(X, y)

        logging.info(
            'Training done: {:.3f} s'.format(time.time() - start_time)
        )
        return model
