import os
import argparse
import logging
from datetime import datetime
from mgc.experiments import bayes, deep, tree, svm, lstm
from mgc.persistence import (SklearnModelPersistence,
                             KerasModelPersistence)

from mgc.evaluation import (SklearnModelEvaluator,
                            KerasModelEvaluator)

from mgc.dataloading import (NumpyMusicGenreSetLoader,
                             TFMusicGenreSetLoader)


# Configuration of EXPERIMENTS


EXPERIMENTS = {
    'bayes': lambda args: bayes.BayesExperiment(
        data_loader=NumpyMusicGenreSetLoader(setup_datadir()),
        evaluator=SklearnModelEvaluator(
            get_output_filepath(args, extension='csv')),
        persistence=SklearnModelPersistence(
            saved_model_filepath('bayes.joblib', args)),
        balanced=args.balanced
    ),
    'deep': lambda args: deep.DeepExperiment(
        data_loader=TFMusicGenreSetLoader(setup_datadir()),
        evaluator=KerasModelEvaluator(
            get_output_filepath(args, extension='csv')),
        persistence=KerasModelPersistence(
            saved_model_filepath('deep_wt.h5', args)),
        balanced=args.balanced,
        epochs=args.epochs
    ),
    'lstm': lambda args: lstm.LSTMExperiment(
        data_loader=TFMusicGenreSetLoader(setup_datadir()),
        evaluator=KerasModelEvaluator(
            get_output_filepath(args, extension='csv')),
        persistence=KerasModelPersistence(
            saved_model_filepath('lstm_wt.h5', args)),
        balanced=args.balanced,
        epochs=args.epochs
    ),
    'svm': lambda args: svm.SVMExperiment(
        data_loader=NumpyMusicGenreSetLoader(setup_datadir()),
        evaluator=SklearnModelEvaluator(
            get_output_filepath(args, extension='csv')),
        persistence=SklearnModelPersistence(
            saved_model_filepath('svm.joblib', args)),
        balanced=args.balanced
    ),
    'tree': lambda args: tree.DecisionTreeExperiment(
        data_loader=NumpyMusicGenreSetLoader(setup_datadir()),
        evaluator=SklearnModelEvaluator(
            get_output_filepath(args, extension='csv')),
        persistence=SklearnModelPersistence(
            saved_model_filepath('tree.joblib', args)),
        balanced=args.balanced
    )
}


# Support functions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'experiment',
        help='The experiment to be executed',
        choices=EXPERIMENTS.keys()
    )
    parser.add_argument(
        '--balanced',
        action='store_true'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50
    )
    return parser.parse_args()


def setup_logging(args):
    logfile = get_output_filepath(args)
    logging.basicConfig(
        level=logging.INFO,
        filename=logfile,
        format='%(asctime)s %(message)s')
    logging.info(args)


def get_output_filepath(args, extension='log'):
    balanced = 'bal' if args.balanced else 'unbal'
    logfile = 'logs/{}_{}_{}.{}'.format(
        args.experiment,
        balanced,
        datetime.now().strftime("%Y-%m-%d_%H.%M.%S"),
        extension
    )
    return logfile


def setup_datadir():
    datadir = os.environ.get(
        'DATA_DIR',
        './downloads/audioset/audioset_v1_embeddings/'
    )
    datadir = os.path.abspath(datadir)
    logging.info('Data dir: {}'.format(datadir))
    return datadir


def saved_model_filepath(filename, args):
    final_filename = filename
    if (args.balanced):
        final_filename = 'bal_{}'.format(filename)
    else:
        final_filename = 'unbal_{}'.format(filename)

    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'saved_models',
        final_filename
    )


# Run main program

if __name__ == "__main__":
    args = parse_args()
    setup_logging(args)
    experiment = EXPERIMENTS[args.experiment](args)
    experiment.run()
