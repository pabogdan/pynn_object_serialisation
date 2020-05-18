# import keras dataset to deal with our common use cases
from keras.datasets import mnist, cifar10, cifar100
import keras
# usual sPyNNaker imports
from pynn_object_serialisation.experiments.mnist_testing.mnist_argparser import args
from pynn_object_serialisation.experiments.mobilenet_testing.utils import \
    retrieve_git_commit, compute_input_spikes
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

# Check parameters passed in from argparser
if args.testing_examples and (args.no_slices or args.curr_slice):
    raise AttributeError("Can't received both number of testing examples and "
                         "slice information.")



# Record SCRIPT start time (wall clock)
start_time = plt.datetime.datetime.now()

# Checking directory structure exists
if not os.path.isdir(args.result_dir) and not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)

N_layer = 28 ** 2  # number of neurons in each population
t_stim = args.t_stim
(x_train, y_train), (full_x_test, full_y_test) = mnist.load_data()
# reshape input to flatten data
# x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
full_x_test = full_x_test.reshape(full_x_test.shape[0], np.prod(full_x_test.shape[1:]))

if not (args.no_slices or args.curr_slice):
    if args.testing_examples:
        no_testing_examples = args.testing_examples
    else:
        no_testing_examples = full_x_test.shape[0]
    testing_examples = np.arange(no_testing_examples)
else:
    no_testing_examples = full_x_test.shape[0] // args.no_slices
    testing_examples = \
        np.arange(full_x_test.shape[0])[args.curr_slice * no_testing_examples:
                                   (args.curr_slice + 1) * no_testing_examples]

runtime = no_testing_examples * t_stim
number_of_slots = int(runtime / t_stim)
range_of_slots = np.arange(number_of_slots)
starts = np.ones((N_layer, number_of_slots)) * (range_of_slots * t_stim)
durations = np.ones((N_layer, number_of_slots)) * t_stim
rates = full_x_test[testing_examples, :].T
y_test = full_y_test[testing_examples]


# scaling rates
_0_to_1_rates = rates / float(np.max(rates))
rates = _0_to_1_rates * args.rate_scaling

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

sim.setup(args.timestep,
          args.timestep,
          args.timestep,
          time_scale_factor=args.timescale)

sim.set_number_of_neurons_per_core(SpikeSourcePoissonVariable, 64)
# sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 64)
# sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 64)

populations, projections, extra_params = restore_simulator_from_file(
    sim, args.model,
    is_input_vrpss=True,
    vrpss_cellparams=input_params,
    replace_params=replace,
)
# set_i_offsets(populations, runtime)

for pop in populations[:]:
    pop.record("spikes")
if args.record_v:
    populations[-1].record("v")
spikes_dict = {}
neo_spikes_dict = {}
current_error = None
final_connectivity = {}

sim_start_time = plt.datetime.datetime.now()
try:
    if not args.reset_v:
        sim.run(runtime)
    else:
        run_duration = args.t_stim  # ms
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

# Compute time taken to reach this point
end_time = plt.datetime.datetime.now()
total_time = end_time - start_time
sim_total_time = end_time - sim_start_time



for pop in populations[:]:
    spikes_dict[pop.label] = pop.spinnaker_get_data('spikes')
for pop in populations:
    # neo_spikes_dict[pop.label] = pop.get_data('spikes')
    neo_spikes_dict[pop.label] = []
# the following takes more time than spinnaker_get_data
# for pop in populations[:]:
#     neo_spikes_dict[pop.label] = pop.get_data('spikes')
if args.record_v:
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


if args.result_filename:
    results_filename = args.result_filename
else:
    results_filename = "mnist_results"
    if args.suffix:
        results_filename += args.suffix
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
results_file = os.path.join(args.result_dir, results_filename)
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
                    extra_params=extra_params)
sim.end()
# Analysis time!
post_run_analysis(filename=results_file, fig_folder=args.figures_dir)

# Report time taken
print("Results stored in  -- " + results_filename)

# Report time taken
print("Total time elapsed -- " + str(total_time))
