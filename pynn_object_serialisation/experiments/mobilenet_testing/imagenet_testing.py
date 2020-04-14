# usual sPyNNaker imports
from pynn_object_serialisation.experiments.mobilenet_testing.imagenet_argparser import args

try:
    import spynnaker8 as sim
except Exception:
    import pyNN.spiNNaker as sim
from pynn_object_serialisation.functions import \
    restore_simulator_from_file, get_rescaled_i_offset, \
    set_i_offsets, get_input_size
import pylab as plt
import numpy as np
import os
import traceback
import sys
from pynn_object_serialisation.experiments.mobilenet_testing.utils import \
    retrieve_git_commit, compute_input_spikes
# Record SCRIPT start time (wall clock)
start_time = plt.datetime.datetime.now()

# Checking directory structure exists
if not os.path.isdir(args.result_dir) and not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)

N_layer = get_input_size(args.model)

input_params, y_test = compute_input_spikes(
    no_tests=args.testing_examples,
    data_path=args.data_dir,
    input_size=N_layer,
    t_stim=args.t_stim,
    rate_scaling=args.rate_scaling,
)

runtime = args.testing_examples * args.t_stim

# produce parameter replacement dict
replace = {
    "tau_syn_E": 0.2,
    "tau_syn_I": 0.2,
    "v_thresh": 1.,
}
output_v = []
populations, projections, custom_params = restore_simulator_from_file(
    sim, args.model,
    is_input_vrpss=True,
    vrpss_cellparams=input_params,
    replace_params=replace,
    prune_level=args.conn_level,
    n_boards_required=args.number_of_boards,
    time_scale_factor=args.timescale,
    first_n_layers=args.first_n_layers)
set_i_offsets(populations, runtime)

# set up recordings for other layers if necessary
for pop in populations[:]:
    pop.record("spikes")
if args.record_v:
    populations[-1].record("v")
spikes_dict = {}
neo_spikes_dict = {}

# Record simulation start time (wall clock)
sim_start_time = plt.datetime.datetime.now()
current_error = None
final_connectivity = {}
# Run the simulation
try:
    sim.run(runtime)
except Exception as e:
    print("An exception occurred during execution!")
    traceback.print_exc()
    current_error = e
# Compute time taken to reach this point
end_time = plt.datetime.datetime.now()
total_time = end_time - start_time
sim_total_time = end_time - sim_start_time

pop_labels = [pop.label for pop in populations]


for pop in populations:
    spikes_dict[pop.label] = pop.spinnaker_get_data('spikes')
# the following takes more time than spinnaker_get_data
for pop in populations:
    # neo_spikes_dict[pop.label] = pop.get_data('spikes')
    neo_spikes_dict[pop.label] = []

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
if args.record_v:
    output_v = populations[-1].spinnaker_get_data('v')
# save results

if args.result_filename:
    results_filename = args.result_filename
else:
    results_filename = "imagenet_results"
    if args.suffix:
        results_filename += args.suffix
    else:
        import pylab

        now = pylab.datetime.datetime.now()
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
np.savez_compressed(os.path.join(args.result_dir, results_filename),
                    output_v=output_v,
                    neo_spikes_dict=neo_spikes_dict,
                    all_spikes=spikes_dict,
                    y_test=input_params['y_test'],
                    input_params=input_params,
                    input_size=N_layer,
                    sim_time=runtime,
                    sim_params=sim_params,
                    final_connectivity=final_connectivity,
                    **spikes_dict)
sim.end()
