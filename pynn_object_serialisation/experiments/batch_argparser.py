import argparse

# Current defaults for [App: Motion detection]
# as of 08.10.2018

# Default values
DEFAULT_NO_CPUS = 32
DEFAULT_MAX_CONCURRENT_PROCESSES = 10
DEFAULT_SUFFIX = None
DEFAULT_NO_SLICES = 10
DEFAULT_RATE_SCALING = 1000  # Hz
DEFAULT_T_STIM = 200  # ms
DEFAULT_TIMESTEP = 1.0

# Argument parser
parser = argparse.ArgumentParser(
    description='Batch runner adapted from '
                'https://github.com/pabogdan/neurogenesis/blob/master/synaptogenesis/batch_argparser.py'
                'and '
                'https://github.com/pabogdan/neurogenesis/blob/master/synaptogenesis/batch_runner.py')

parser.add_argument('models', help='path of input .npz archive defining '
                                 'connectivity', nargs='*')

parser.add_argument('--model_script', help='path to model testing script')

parser.add_argument('--no_slices', type=int,
                    default=DEFAULT_NO_SLICES,
                    help='number of parallel runs of the same network, each '
                         'being fed a different slice of the input dataset'
                         ' -- [default {}]'.format(DEFAULT_NO_SLICES))

parser.add_argument('--suffix', type=str,
                    help="add a recognisable suffix to all the file "
                         "generated in this batch "
                         "-- [default {}]".format(DEFAULT_SUFFIX),
                    dest='suffix')

parser.add_argument('--no_cpus', type=int,
                    default=DEFAULT_NO_CPUS, dest='no_cpus',
                    help='total number of available CPUs'
                         ' -- [default {}]'.format(DEFAULT_NO_CPUS))

parser.add_argument('--t_stim', type=int,
                    help='how long to present single patterns',
                    default=DEFAULT_T_STIM)

parser.add_argument('--timestep', type=float,
                    help='simulation time step',
                    default=DEFAULT_TIMESTEP)

parser.add_argument('--reset_v',
                    help='Reset voltage of all neurons in the network after '
                         'each pattern presentation',
                    action="store_true")

parser.add_argument('--max_processes', type=int,
                    default=DEFAULT_MAX_CONCURRENT_PROCESSES,
                    dest='max_processes',
                    help='total number of concurrent processes'
                         ' -- [default {}]'.format(DEFAULT_MAX_CONCURRENT_PROCESSES))

parser.add_argument('--rate_scaling', type=float,
                    help='input value scaling so as to properly be interpreted'
                         'as a rate', default=DEFAULT_RATE_SCALING)

args = parser.parse_args()
