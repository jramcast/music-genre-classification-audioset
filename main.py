import os
import argparse
import logging
from datetime import datetime
from mgc.experiments import bayes, deep, svm


EXPERIMENTS = {
    'bayes': bayes.BayesExperiment,
    'deep': deep.DeepExperiment,
    'svm': svm.SVMExperiment
}


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
    return parser.parse_args()


def setup_logging(args):
    balanced = 'bal' if args.balanced else 'unbal'
    logfile = 'logs/{}_{}_{}.log'.format(
        args.experiment,
        balanced,
        datetime.now().isoformat()
    )
    logging.basicConfig(
        level=logging.INFO,
        filename=logfile,
        format='%(asctime)s %(message)s')


def setup_datadir():
    datadir = os.environ.get(
        'DATA_DIR',
        './downloads/audioset/audioset_v1_embeddings/'
    )
    datadir = os.path.abspath(datadir)
    logging.debug('Data dir: {}'.format(datadir))
    return datadir


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args)
    datadir = setup_datadir()
    Experiment = EXPERIMENTS[args.experiment]
    experiment = Experiment(datadir, balanced=args.balanced)
    experiment.run()
