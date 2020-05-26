# import keras dataset to deal with our common use cases
from keras.datasets import mnist
from retry import retry

from pynn_object_serialisation.OutputDataProcessor import OutputDataProcessor
from pynn_object_serialisation.experiments.mobilenet_testing.utils import \
    retrieve_git_commit
from pynn_object_serialisation.experiments.post_run_analysis import post_run_analysis
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
import pylab as plt
import traceback

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
        (args.model, args.t_stim, args.rate_scaling, args.time_scale_factor, args.testing_examples, args.timestep)
    return folder_details


@retry(tries=5, jitter=(0, 10), logger=logger)
def run(args, start_index):
    # Record SCRIPT start time (wall clock)
    start_time = plt.datetime.datetime.now()

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

    N_layer = 28 ** 2  # number of neurons in input population
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

    print("Setting number of neurons per core...")

    sim.set_number_of_neurons_per_core(SpikeSourcePoissonVariable, 16)
    sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 16)
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 64)
    sim.set_number_of_neurons_per_core(sim.IF_cond_exp, 64)

    print("Restoring populations and projections...")
    populations, projections, extra_params = restore_simulator_from_file(
        sim, args.model,
        input_type='vrpss',
        vrpss_cellparams=input_params,
        replace_params=replace)

    old_runtime = extra_params['runtime'] if 'runtime' in extra_params else None
    print("Setting i_offsets...")
    set_i_offsets(populations, runtime, old_runtime=old_runtime)

    spikes_dict = {}
    neo_spikes_dict = {}
    current_error = None
    final_connectivity = {}

    def reset_membrane_voltage():
        for population in populations[1:]:
            population.set_initial_value(variable="v", value=0)
        return

    for pop in populations[:]:
        pop.record("spikes")
    if args.record_v:
        populations[-1].record("v")

    sim_start_time = plt.datetime.datetime.now()
    if not args.reset_v:
        sim.run(runtime)

    for i in range(args.chunk_size):
        # TODO add a lower level of retry that doesn't need to reload model.
        # TODO find out why the following print doesn't print.
        print('Presenting example {}/{}'.format(start_index + i, testing_examples))
        sim.run(t_stim)
        reset_membrane_voltage()

    # Compute time taken to reach this point
    end_time = plt.datetime.datetime.now()
    total_time = end_time - start_time
    sim_total_time = end_time - sim_start_time

    for pop in populations[:]:
        spikes_dict[pop.label] = pop.spinnaker_get_data('spikes')
    if args.record_v:
        output_v = populations[-1].spinnaker_get_data('v')

    try:
        for proj in projections:
            try:
                final_connectivity[proj.label] = \
                    np.array(proj.get(('weight', 'delay'), format="list")._get_data_items())
            except AttributeError as ae:
                print("Careful! Something happened when retrieving the "
                      "connectivity:", ae,
                      "\nRetrying using standard PyNN syntax...")
                final_connectivity[proj.label] = \
                    np.array(proj.get(('weight', 'delay'), format="list"))
            except TypeError as te:
                print("Connectivity is None (", te,
                      ") for connection", proj.label)
                print("Connectivity as empty array.")
                final_connectivity[proj.label] = np.array([])
    except:
        traceback.print_exc()
        print("Couldn't retrieve connectivity.")


    if args.result_filename:
        results_filename = args.result_filename
    else:
        results_filename = "mnist_results"
        if args.suffix:
            results_filename += args.suffix
        else:
            pass

    # Retrieve simulation parameters for provenance tracking and debugging purposes
    sim_params = {
        "argparser": vars(args),
        "git_hash": retrieve_git_commit(),
        "run_end_time": end_time.strftime("%H:%M:%S_%d/%m/%Y"),
        "wall_clock_script_run_time": str(total_time),
        "wall_clock_sim_run_time": str(sim_total_time),
    }
    results_file = os.path.join(os.path.join(args.result_dir, results_filename + "_" + str(start_index)))

    np.savez_compressed(results_file,
                        output_v=output_v,
                        neo_spikes_dict=neo_spikes_dict,
                        all_spikes=spikes_dict,
                        all_neurons=extra_params['all_neurons'],
                        testing_examples=testing_examples,
                        no_testing_examples=testing_examples,
                        num_classes=10,
                        y_test=y_test,
                        input_params=input_params,
                        input_size=N_layer,
                        simtime=runtime,
                        sim_params=sim_params,
                        final_connectivity=final_connectivity,
                        init_connectivity=extra_params['all_connections'],
                        extra_params=extra_params,
                        current_error=current_error)
    sim.end()

    # TODO move analysis into another parallel loop
    # Analysis time!
    post_run_analysis(filename=results_file, fig_folder=args.result_dir+args.figures_dir)

    # Report time taken
    print("Results stored in  -- " + results_filename)

    # Report time taken
    print("Total time elapsed -- " + str(total_time))
    return current_error


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
    p = Pool(args.number_of_processes)
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
