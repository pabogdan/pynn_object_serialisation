# import keras dataset to deal with our common use cases
from keras.datasets import mnist
from retry import retry

from pynn_object_serialisation.OutputDataProcessor import OutputDataProcessor

try:
    import spynnaker8 as sim
except:
    import pyNN.spiNNaker as sim
from pynn_object_serialisation.functions import \
    restore_simulator_from_file, set_i_offsets
from spynnaker8.extra_models import SpikeSourcePoissonVariable
from multiprocessing.pool import Pool
from spinn_front_end_common.utilities import globals_variables
import multiprocessing
import itertools
import numpy as np
import os
import sys
import logging
import signal

# Make a logger to allow warnings to go to stdout
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging.getLogger()
try_number = -1

def signal_handler(signal, frame):
    # This works around a bug in some versions of Python.
    print('You pressed Ctrl+C!')
    sys.exit(0)


def generate_run_folder_from_params(args):
    folder_details = "/model_name_{}_t_stim_{}_rate_scaling_{}_tsf_{}_testing_examples_{}_dt_{}".format \
        (args.model, args.t_stim, args.rate_scaling, args.time_scale_factor, args.testing_examples, args.dt)


@retry(tries=5, jitter=(0, 10), logger=logger)
def run(args, start_index):
    # Note that this won't be global between processes
    global try_number
    try_number += 1
    globals_variables.unset_simulator()
    signal.signal(signal.SIGINT, signal_handler)
    current = multiprocessing.current_process()
    print('Started {}'.format(current) + '\n')


    f_name = "errorlog/" + current.name + "_stdout.txt"
    g_name = "errorlog/" + current.name + "_stderror.txt"
    f = open(f_name, 'w')
    g = open(g_name, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = f
    sys.stderr = g

    # Record SCRIPT start time (wall clock)
    start_time = plt.datetime.datetime.now()

    N_layer = 28 ** 2  # number of neurons in each population
    t_stim = args.t_stim
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # reshape input to flatten data

    x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))

    testing_examples = args.chunk_size
    runtime = testing_examples * t_stim
    range_of_slots = np.arange(testing_examples)
    starts = np.ones((N_layer, testing_examples)) * (range_of_slots * t_stim)
    durations = np.ones((N_layer, testing_examples)) * t_stim
    rates = x_test[start_index:start_index + args.chunk_size, :].T
    y_test = y_test[start_index:start_index + args.chunk_size]

    # scaling rates
    _0_to_1_rates = rates / float(np.max(rates))
    rates = _0_to_1_rates * args.rate_scaling

    input_params = {
        "rates": rates,
        "durations": durations,
        "starts": starts
    }

    print("Number of testing examples to use:", testing_examples)
    print("Min rate", np.min(rates))
    print("Max rate", np.max(rates))
    print("Mean rate", np.mean(rates))

    replace = None

    timestep = args.timestep
    timescale = args.time_scale_factor

    output_v = []
    sim.setup(timestep,
              timestep,
              timestep,
              time_scale_factor=timescale)

    sim.set_number_of_neurons_per_core(SpikeSourcePoissonVariable, 16)
    sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 16)
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 64)
    sim.set_number_of_neurons_per_core(sim.IF_cond_exp, 64)

    populations, projections, custom_params = restore_simulator_from_file(
        sim, args.model,
        input_type='vrpss',
        vrpss_cellparams=input_params,
        replace_params=replace)
    dt = sim.get_time_step()
    min_delay = sim.get_min_delay()
    max_delay = sim.get_max_delay()
    print("Setting number of neurons per core...")

    old_runtime = custom_params['runtime'] if 'runtime' in custom_params else None
    print("Setting i_offsets...")
    set_i_offsets(populations, runtime, old_runtime=old_runtime)

    spikes_dict = {}
    neo_spikes_dict = {}

    def reset_membrane_voltage():
        for population in populations[1:]:
            population.set_initial_value(variable="v", value=0)
        return

    for pop in populations[:]:
        pop.record("spikes")
    if args.record_v:
        populations[-1].record("v")
    for i in range(args.chunk_size):
        # TODO add a lower level of retry that doesn't need to reload model.
        # TODO find out why the following print doesn't print.
        print('Presenting example {}'.format(start_index + i))
        sim.run(t_stim)
        reset_membrane_voltage()
    for pop in populations[:]:
        spikes_dict[pop.label] = pop.spinnaker_get_data('spikes')
    if args.record_v:
        output_v = populations[-1].spinnaker_get_data('v')



    if args.result_filename:
        results_filename = args.result_filename
    else:
        results_filename = "mnist_results"
        if args.suffix:
            results_filename += args.suffix
        else:
            pass

        # now = pylab.datetime.datetime.now()
        # results_filename += "_" + now.strftime("_%H%M%S_%d%m%Y")

    np.savez_compressed(os.path.join(args.result_dir, results_filename + "_" + str(start_index)),
                        output_v=output_v,
                        neo_spikes_dict=neo_spikes_dict,
                        y_test=y_test,
                        N_layer=N_layer,
                        t_stim=t_stim,
                        runtime=runtime,
                        sim_time=runtime,
                        dt=dt,
                        **spikes_dict)
    sim.end()
    return


if __name__ == "__main__":
    from pynn_object_serialisation.experiments.mnist_testing.mnist_argparser import args

    sub_folder_name = generate_run_folder_from_params(args)
    args.result_dir += sub_folder_name

    # Checking directory structure exists
    if not os.path.isdir(args.result_dir) and not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    if not os.path.exists('errorlog'):
        os.makedirs('errorlog')

    # Make a pool
    p = Pool(args.number_of_threads)
    # Run the pool
    p.starmap(run, zip(itertools.repeat(args), list(range(0, args.testing_examples, args.chunk_size))))



    print("Simulations complete. Gathering data...")

    accuracies = []
    for filename in os.listdir(args.result_dir):
        data_processor = OutputDataProcessor(os.path.join(args.result_dir, filename))
        accuracy = data_processor.get_accuracy()
        print("File: {} Accuracy: {}".format(str(filename), str(accuracy)))
        accuracies.append(accuracy)
    print("Accuracy = {}".format(str(sum(accuracies) / len(accuracies))))
