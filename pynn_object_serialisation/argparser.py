import argparse

# time constants
DEFAULT_SIM_TIME = 200  # ms

# default connectivity values
DEFAULT_MINIMUM_PRUNE_LEVEL = 1  # proportion of top weights

# dir defaults
DEFAULT_JSON_DIR = 'examples/'

parser = argparse.ArgumentParser(
    description='DNN argparser',
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('model', type=str,
                    help='network architecture / model to train and test')

parser.add_argument('--sim_time', type=int,
                    help='simulation time (ms)',
                    default=DEFAULT_SIM_TIME)

parser.add_argument('--suffix', type=str,
                    help='loss function', default=None)

parser.add_argument('--dir', type=str,
                    help='directory in which to load and '
                         'store network architectures',
                    default=DEFAULT_JSON_DIR)

parser.add_argument('--prune_level', type=float,
                    help='proportion of top weights to be kept -- '
                         'e.g. a value of 1 means 100% of connections are kept,'
                         ' while a value of .1 means that the top 10% '
                         'of connections are kept (based on weight)',
                    default=DEFAULT_MINIMUM_PRUNE_LEVEL)
args = parser.parse_args()
