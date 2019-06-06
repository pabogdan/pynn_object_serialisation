import argparse

DEFAULT_SIM_TIME = 200  # ms


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

args = parser.parse_args()
