import argparse

batch_size = 50
epochs = 20

DEFAULT_MODEL_DIR = 'models/'
DEFAULT_RESULT_DIR = 'results/'
DEFAULT_FIGURE_DIR = 'figures/'

DEFAULT_RESULT_FILENAME = "results"
DEFAULT_RATE_SCALING = 1000  # Hz
DEFAULT_T_STIM = 200  # ms
DEFAULT_TIMESCALE = None
DEFAULT_TESTING_EXAMPLES = None
DEFAULT_CHUNK_SIZE = 100
DEFAULT_NUMBER_OF_PROCESSES = 1
DEFAULT_TIME_SCALE_FACTOR = 100
DEFAULT_TIMESTEP = 0.1

parser = argparse.ArgumentParser(
    description='converted-MNIST argparser',
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('model', type=str,
                    help='network architecture / model to train and test')

parser.add_argument('-o', '--output', type=str,
                    help="name of the numpy archive (.npz) "
                         "storing simulation results",
                    dest='result_filename')

parser.add_argument('--non_categorical', dest="non_categorical",
                    help='filename for results',
                    action="store_false")

parser.add_argument('--reset_v',
                    help='Reset voltage of all neurons in the network after '
                         'each pattern presentation',
                    action="store_true")

parser.add_argument('--record_v',
                    help='record voltage for output neurons',
                    action="store_true")

parser.add_argument('--timescale', type=int,
                    help='timescale factor for the simulation',
                    default=DEFAULT_TIMESCALE)

parser.add_argument('--test_with_pss',
                    help='Test using only the Poisson Spike Source '
                         '(not variable)',
                    action="store_true")

parser.add_argument('--timestep', type=float,
                    help='simulation timestep',
                    default=DEFAULT_TIMESTEP)

parser.add_argument('--epochs', type=int,
                    help='number of epochs', default=epochs)

parser.add_argument('--dataset', type=str,
                    help='dataset for training and testing', default='mnist')

parser.add_argument('--suffix', type=str,
                    help='loss function', default=None)

parser.add_argument('--model_dir', type=str,
                    help='directory in which to load and '
                         'store network architectures',
                    default=DEFAULT_MODEL_DIR)

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

parser.add_argument('--testing_examples', type=int,
                    help='number of testing examples to show',
                    default=DEFAULT_TESTING_EXAMPLES)

parser.add_argument('--chunk_size', type=int,
                    help='the number of bins per chunk',
                    default=DEFAULT_CHUNK_SIZE)

parser.add_argument('--number_of_processes', type=int,
                    help='the number of processes to run in parallel',
                    default=DEFAULT_NUMBER_OF_PROCESSES)

parser.add_argument('--figures_dir', type=str,
                    help='directory into which to save figures',
                    default=DEFAULT_FIGURE_DIR)

parser.add_argument('--time_scale_factor', type=float,
                    help='the slowdown factor for the simulation',
                    default=DEFAULT_TIME_SCALE_FACTOR)

parser.add_argument('--force_resim',
                    help='should present results in result directory be ignored and overwritten?',
                    action="store_true")

args = parser.parse_args()
