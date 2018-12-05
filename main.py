import os
import argparse
import logging
from datetime import datetime
from mgc.experiments import bayes, deep, base



EXPERIMENTS = {
    'bayes': bayes.BayesExperiment,
    'deep': deep.DeepExperiment
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
    logfile = get_output_filepath(args)
    logging.basicConfig(
        level=logging.INFO,
        filename=logfile,
        format='%(asctime)s %(message)s')


def get_output_filepath(args, extension='log'):
    balanced = 'bal' if args.balanced else 'unbal'
    logfile = 'logs/{}_{}_{}.{}'.format(
        args.experiment,
        balanced,
        datetime.now().isoformat(),
        extension
    )
    return logfile


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
    ConcreteExperiment = EXPERIMENTS[args.experiment]
    experiment: base.Experiment = ConcreteExperiment(
        datadir,
        balanced=args.balanced,
        stats_filepath=get_output_filepath(args, extension='csv'),
    )
    experiment.run()
