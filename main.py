import argparse
import logging
from datetime import datetime
from mgc.experiments import bayes


EXPERIMENTS = {
    'bayes': bayes
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


def setup_logging():
    logfile = 'logs/bayes_{}.log'.format(datetime.now().isoformat())
    logging.basicConfig(
        level=logging.INFO,
        filename=logfile,
        format='%(asctime)s %(message)s')


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    EXPERIMENTS[args.experiment].run()
