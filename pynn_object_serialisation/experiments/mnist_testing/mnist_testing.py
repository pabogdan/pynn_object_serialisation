# import keras dataset to deal with our common use cases
from keras.datasets import mnist
# usual sPyNNaker imports
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
import numpy as np
import pylab as plt
import os
import traceback




def test_converted_network(path_to_network, t_stim, rate_scaling=1000,
                           no_slices=None, curr_slice=None,
                           testing_examples=None, result_filename=None,
                           result_dir="results",
                           figures_dir="figures", suffix=None,
                           timestep=1.0,
                           timescale=None, reset_v=False, record_v=False):
    # Check parameters passed in from argparser
    if testing_examples and (no_slices or curr_slice):
        raise AttributeError("Can't received both number of testing examples and "
                             "slice information.")

    # Record SCRIPT start time (wall clock)
    start_time = plt.datetime.datetime.now()

    # Checking directory structure exists
    if not os.path.isdir(result_dir) and not os.path.exists(result_dir):
        os.mkdir(result_dir)

    N_layer = 28 ** 2  # number of neurons in each population
    t_stim = t_stim
    (x_train, y_train), (full_x_test, full_y_test) = mnist.load_data()
    # reshape input to flatten data
    # x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
    full_x_test = full_x_test.reshape(full_x_test.shape[0], np.prod(full_x_test.shape[1:]))

    if not (no_slices or curr_slice):
        if testing_examples:
            no_testing_examples = testing_examples
        else:
            no_testing_examples = full_x_test.shape[0]
        testing_examples = np.arange(no_testing_examples)
    else:
        no_testing_examples = full_x_test.shape[0] // no_slices
        testing_examples = \
            np.arange(full_x_test.shape[0])[curr_slice * no_testing_examples:
                                       (curr_slice + 1) * no_testing_examples]

    runtime = no_testing_examples * t_stim
    number_of_slots = int(runtime / t_stim)
    range_of_slots = np.arange(number_of_slots)
    starts = np.ones((N_layer, number_of_slots)) * (range_of_slots * t_stim)
    durations = np.ones((N_layer, number_of_slots)) * t_stim
    rates = full_x_test[testing_examples, :].T
    y_test = full_y_test[testing_examples]


    # scaling rates
    _0_to_1_rates = rates / float(np.max(rates))
    rates = _0_to_1_rates * rate_scaling

    input_params = {
        "rates": rates,
        "durations": durations,
        "starts": starts
    }

    print("Number of testing examples to use:", no_testing_examples)
    print("Min rate", np.min(rates))
    print("Max rate", np.max(rates))
    print("Mean rate", np.mean(rates))

    replace = None
    # produce parameter replacement dict
    output_v = []
    populations, projections, custom_params = restore_simulator_from_file(
    sim, args.model,
    input_type='vrpss',
    vrpss_cellparams=input_params,
    replace_params=replace)
    dt = sim.get_time_step()
    min_delay = sim.get_min_delay()
    max_delay = sim.get_max_delay()
    sim.set_number_of_neurons_per_core(SpikeSourcePoissonVariable, 16)
    sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 16)
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 64)
    old_runtime = custom_params['runtime']
    set_i_offsets(populations, runtime, old_runtime=old_runtime)
    # if args.test_with_pss:
    #     pss_params = {
    #         'rate'
    #     }
    #     populations.append(sim.Population(sim.SpikeSourcePoisson, ))
    # set up recordings for other layers if necessary
    spikes_dict = {}
    neo_spikes_dict = {}
    
    def record_output(populations,offset,output):
        spikes = populations[-1].spinnaker_get_data('spikes')
        spikes = spikes + [0,offset]
        name = populations[-1].label
        if np.shape(spikes)[0]>0:
            if name in list(output.keys()):
                output[name] = np.concatenate((output,spikes))
            else:
                output[name] = spikes 
        return output

    for pop in populations[:]:
        pop.record("spikes")
    if record_v:
        populations[-1].record("v")
    spikes_dict = {}
    neo_spikes_dict = {}
    current_error = None
    final_connectivity = {}

    sim_start_time = plt.datetime.datetime.now()
    try:
        if not reset_v:
            sim.run(runtime)
        else:
            run_duration = t_stim  # ms
            no_runs = runtime // run_duration
            for curr_run_number in range(no_runs):
                print("RUN NUMBER", curr_run_number, "STARTED")
                sim.run(run_duration)  # ms
                for pop in populations[1:]:
                    pop.set_initial_value("v", 0)
                print("RUN NUMBER", curr_run_number, "COMPLETED")
    except Exception as e:
        print("An exception occurred during execution!")
        traceback.print_exc()
        current_error = e

    def reset_membrane_voltage():        
        for population in populations[1:]:
            population.set_initial_value(variable="v", value=0)
        return

    for population in populations[1:]:
        pop.set_initial_value(variable="v", value=0)
    for presentation in range(testing_examples):
        sim.run(t_stim)
        reset_membrane_voltage()
    for pop in populations[:]:
        spikes_dict[pop.label] = pop.spinnaker_get_data('spikes')
    for pop in populations:
        # neo_spikes_dict[pop.label] = pop.get_data('spikes')
        neo_spikes_dict[pop.label] = []
    # the following takes more time than spinnaker_get_data
    # for pop in populations[:]:
    #     neo_spikes_dict[pop.label] = pop.get_data('spikes')
    if record_v:
        output_v = populations[-1].spinnaker_get_data('v')
    # save results


    try:
        for proj in projections:
            try:
                final_connectivity[proj.label] = \
                    np.array(proj.get(('weight', 'delay'),
                                      format="list")._get_data_items())
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


    if result_filename:
        results_filename = result_filename
    else:
        results_filename = "mnist_results"
        if suffix:
            results_filename += suffix
        else:
            now = plt.datetime.datetime.now()
            results_filename += "_" + now.strftime("_%H%M%S_%d%m%Y")


    # Retrieve simulation parameters for provenance tracking and debugging purposes
    sim_params = {
        "argparser": vars(args),
        "git_hash": retrieve_git_commit(),
        "run_end_time": end_time.strftime("%H:%M:%S_%d/%m/%Y"),
        "wall_clock_script_run_time": str(total_time),
        "wall_clock_sim_run_time": str(sim_total_time),
    }

    # TODO Retrieve original connectivity and JSON and store here.
    results_file = os.path.join(result_dir, results_filename)
    np.savez_compressed(results_file,
                        output_v=output_v,
                        neo_spikes_dict=neo_spikes_dict,
                        all_spikes=spikes_dict,
                        all_neurons=extra_params['all_neurons'],
                        testing_examples=testing_examples,
                        no_testing_examples=no_testing_examples,
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
    # Analysis time!
    post_run_analysis(filename=results_file, fig_folder=figures_dir)

    # Report time taken
    print("Results stored in  -- " + results_filename)

    # Report time taken
    print("Total time elapsed -- " + str(total_time))
    return current_error


if __name__ == "__main__":
    from pynn_object_serialisation.experiments.mnist_testing.mnist_argparser import args
    test_converted_network(path_to_network=args.model,
                           t_stim=args.t_stim,
                           rate_scaling=args.rate_scaling,
                           # Data slicing
                           no_slices=args.no_slices,
                           curr_slice=args.curr_slice,
                           testing_examples=args.testing_examples,
                           # Output names and folders
                           figures_dir=args.figures_dir,
                           result_filename=args.result_filename,
                           result_dir=args.result_dir,
                           suffix=args.suffix,
                           # Simualtion parameters
                           timescale=args.timescale,
                           timestep=args.timestep,
                           # Mode selection
                           reset_v=args.reset_v,
                           # Recordings
                           record_v=args.record_v
                           )