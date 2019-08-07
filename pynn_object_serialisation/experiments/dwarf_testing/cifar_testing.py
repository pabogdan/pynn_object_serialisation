# import keras dataset to deal with our common use cases
from keras.datasets import mnist, cifar10, cifar100
import keras
# usual sPyNNaker imports
from cifar_argparser import args

try:
    import spynnaker8 as sim
except:
    import pyNN.spiNNaker as sim
import matplotlib.pyplot as plt
from pynn_object_serialisation.functions import \
    restore_simulator_from_file
from spynnaker8.extra_models import SpikeSourcePoissonVariable
import numpy as np
import os

# Checking directory structure exists
if not os.path.isdir(args.result_dir) and not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)

N_layer = 3 * 32 ** 2  # number of neurons in each population
t_stim = args.t_stim
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# reshape input to flatten data
x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))

if args.testing_examples:
    testing_examples = args.testing_examples
else:
    testing_examples = x_test.shape[0]

runtime = testing_examples * t_stim
number_of_slots = int(runtime / t_stim)
range_of_slots = np.arange(number_of_slots)
starts = np.ones((N_layer, number_of_slots)) * (range_of_slots * t_stim)
durations = np.ones((N_layer, number_of_slots)) * t_stim
# rates = np.random.randint(1, 5, size=(N_layer, number_of_slots))
rates = x_test[:testing_examples, :].T

# scaling rates
_0_to_1_rates = rates / float(np.max(rates))
rates = _0_to_1_rates * args.rate_scaling

input_params = {
    "rates": rates,
    "durations": durations,
    "starts": starts
}
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
    replace_params=replace)
sim.set_number_of_neurons_per_core(SpikeSourcePoissonVariable, 16)
sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 16)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 64)
# if args.test_with_pss:
#     pss_params = {
#         'rate'
#     }
#     populations.append(sim.Population(sim.SpikeSourcePoisson, ))
# set up recordings for other layers if necessary
for pop in populations[:]:
    pop.record("spikes")
if args.record_v:
    populations[-1].record("v")
spikes_dict = {}
neo_spikes_dict = {}
sim.run(runtime)
for pop in populations[:]:
    spikes_dict[pop.label] = pop.spinnaker_get_data('spikes')
# the following takes more time than spinnaker_get_data
# for pop in populations[:]:
#     neo_spikes_dict[pop.label] = pop.get_data('spikes')
if args.record_v:
    output_v = populations[-1].spinnaker_get_data('v')
# save results

if args.result_filename:
    results_filename = args.result_filename
else:
    results_filename = "cifar10_results"
    if args.suffix:
        results_filename += args.suffix
    else:
        import pylab

        now = pylab.datetime.datetime.now()
        results_filename += "_" + now.strftime("_%H%M%S_%d%m%Y")

np.savez_compressed(os.path.join(args.result_dir, results_filename),
                    output_v=output_v,
                    neo_spikes_dict=neo_spikes_dict,
                    y_test=y_test,
                    N_layer=N_layer,
                    t_stim=t_stim,
                    runtime=runtime,
                    sim_time=runtime, 
                    **spikes_dict)
sim.end()
