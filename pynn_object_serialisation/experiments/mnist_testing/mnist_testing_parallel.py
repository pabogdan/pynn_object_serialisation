# import keras dataset to deal with our common use cases
from keras.datasets import mnist, cifar10, cifar100
import keras
from pynn_object_serialisation.OutputDataProcessor import OutputDataProcessor
from retry import retry

try:
    import spynnaker8 as sim
except:
    import pyNN.spiNNaker as sim
import matplotlib.pyplot as plt
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

# Make a logger to allow warnings to go to stdout
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging.getLogger()


def signal_handler(signal, frame):
    # This works around a bug in some versions of Python.
    print('You pressed Ctrl+C!')
    sys.exit(0)


try_number = -1


@retry(tries=5, jitter=(0, 10), logger=logger)
def run(args, start_index):
    # Note that this won't be global between processes
    global try_number
    try_number += 1
    if try_number > 0:
        pass
    globals_variables.unset_simulator()
    signal.signal(signal.SIGINT, signal_handler)
    current = multiprocessing.current_process()
    print('Started {}'.format(current) + '\n')
    if not os.path.exists('errorlog'):
        os.makedirs('errorlog')

    f_name = "errorlog/" + current.name + "_stdout.txt"
    g_name = "errorlog/" + current.name + "_stderror.txt"
    f = open(f_name, 'w')
    g = open(g_name, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = f
    sys.stderr = g



    N_layer = 28 ** 2  # number of neurons in each population
    t_stim = args.t_stim
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # reshape input to flatten data
    x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
    x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))

    testing_examples = args.chunk_size
    runtime = testing_examples * t_stim
    number_of_slots = int(runtime / t_stim)
    range_of_slots = np.arange(number_of_slots)
    starts = np.ones((N_layer, number_of_slots)) * (range_of_slots * t_stim)
    durations = np.ones((N_layer, number_of_slots)) * t_stim

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

    replace = None
    setup_params = {}
    setup_params['machine_time_step'] = args.dt * 1000
    output_v = []
    populations, projections, custom_params = restore_simulator_from_file(
        sim, args.model,
        input_type='vrpss',
        time_scale_factor=args.time_scale_factor,
        vrpss_cellparams=input_params,
        replace_params=replace,
        replace_setup_params=setup_params)
    dt = sim.get_time_step()
    min_delay = sim.get_min_delay()
    max_delay = sim.get_max_delay()
    print("Setting number of neurons per core...")
    sim.set_number_of_neurons_per_core(SpikeSourcePoissonVariable, 16)
    sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 16)
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 64)
    sim.set_number_of_neurons_per_core(sim.IF_cond_exp, 64)
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

    # Checking directory structure exists
    if not os.path.isdir(args.result_dir) and not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    if args.result_filename:
        results_filename = args.result_filename
    else:
        results_filename = "mnist_results"
        if args.suffix:
            results_filename += args.suffix
        else:
            import pylab

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
    import signal
    import mnist_argparser

    args = mnist_argparser.main()

    folder_details = "/model_name_{}_t_stim_{}_rate_scaling_{}_tsf_{}_testing_examples_{}_dt_{}".format \
        (args.model, args.t_stim, args.rate_scaling, args.time_scale_factor, args.testing_examples, args.dt)

    args.result_dir += folder_details

    # run(args, 0)
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