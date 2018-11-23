import argparse
import logging
from datetime import datetime
from mgc.experiments import bayes, deep


EXPERIMENTS = {
    'bayes': bayes,
    'deep': deep
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'experiment',
        help='The experiment to be executed',
        choices=[
            'bayes', 'deep'
        ]
    )
    return parser.parse_args()


def setup_logging(experiment):
    logfile = 'logs/{}_{}.log'.format(experiment, datetime.now().isoformat())
    logging.basicConfig(
        level=logging.INFO,
        filename=logfile,
        format='%(asctime)s %(message)s')


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.experiment)
    EXPERIMENTS[args.experiment].run()
