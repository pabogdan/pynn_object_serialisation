import argparse

batch_size = 50
epochs = 20

DEFAULT_MODEL_DIR = 'models/'
DEFAULT_RESULT_DIR = 'results/'
DEFAULT_FIGURE_DIR = 'figures/'

DEFAULT_RATE_SCALING = 1000  # Hz
DEFAULT_T_STIM = 200  # ms
DEFAULT_TESTING_EXAMPLES = None

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

parser.add_argument('--record_v',
                    help='record voltage for output neurons',
                    action="store_true")

parser.add_argument('--test_with_pss',
                    help='Test using only the Poisson Spike Source '
                         '(not variable)',
                    action="store_true")

parser.add_argument('--no_slices', type=int,
                    help='number of total input data slices '
                         '(used for batch runs)',
                    default=None)

parser.add_argument('--timestep', type=float,
                    help='simulation timestep',
                    default=1.0)

parser.add_argument('--curr_slice', type=int,
                    help='number of current slice of input data '
                         '(used for batch runs)',
                    default=None)

parser.add_argument('--epochs', type=int,
                    help='number of epochs', default=epochs)

parser.add_argument('--batch', type=int,
                    help='batch size', default=batch_size)

parser.add_argument('--optimizer', type=str,
                    help='optimizer to use', default='sgd')

parser.add_argument('--dataset', type=str,
                    help='dataset for training and testing', default='mnist')

parser.add_argument('--activation', type=str,
                    help='activation type', default='relu')

parser.add_argument('--loss', type=str,
                    help='loss function', default='mse')

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

parser.add_argument('--figures_dir', type=str,
                    help='directory into which to save figures',
                    default=DEFAULT_FIGURE_DIR)

args = parser.parse_args()
