import argparse

DEFAULT_MODEL_DIR = 'models/'
DEFAULT_RESULT_DIR = 'results/'
DEFAULT_DATA_DIR = '/ILSVRC/'

DEFAULT_RATE_SCALING = 100  # Hz
DEFAULT_T_STIM = 200  # ms

DEFAULT_NO_EXAMPLES = 100

parser = argparse.ArgumentParser(
    description='converted-cifar argparser',
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('model', type=str,
                    help='network architecture / model to train and test')

parser.add_argument('--result_filename', type=str,
                    help='filename for results')

parser.add_argument('--non_categorical', dest="non_categorical",
                    help='filename for results',
                    action="store_false")

parser.add_argument('--testing_examples', type=int,
                    help='number of testing examples to show',
                    default=DEFAULT_NO_EXAMPLES)

parser.add_argument('--record_v',
                    help='record voltage for output neurons',
                    action="store_true")


parser.add_argument('--test_with_pss',
                    help='Test using only the Poisson Spike Source '
                         '(not variable)',
                    action="store_true")


parser.add_argument('--suffix', type=str,
                    help='loss function', default=None)

parser.add_argument('--model_dir', type=str,
                    help='directory in which to load and '
                         'store network architectures',
                    default=DEFAULT_MODEL_DIR)

parser.add_argument('--data_dir', type=str,
                    help='directory in which imagenet data is',
                    default=DEFAULT_DATA_DIR)

parser.add_argument('--result_dir', type=str,
                    help='directory inp which to load and '
                         'store network architectures',
                    default=DEFAULT_RESULT_DIR)

parser.add_argument('--rate_scaling', type=float,
                    help='input value scaling so as to properly be interpreted'
                         'as a rate', default=DEFAULT_RATE_SCALING)

parser.add_argument('--t_stim', type=int,
                    help='how long to present single patterns',
                    default=DEFAULT_T_STIM)


parser.add_argument('--conn_level', type=float,
                    help='useful to more quickly load MobileNet '
                         'onto SpiNNaker', default=1.)

args = parser.parse_args()
