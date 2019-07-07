# import keras dataset to deal with our common use cases
from keras.datasets import mnist, cifar10, cifar100
import keras
# usual sPyNNaker imports
from mnist_argparser import args

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

N_layer = 28 ** 2  # number of neurons in each population
t_stim = 200
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# reshape input to flatten data
x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))

runtime = x_test.shape[0] * t_stim
number_of_slots = int(runtime / t_stim)
range_of_slots = np.arange(number_of_slots)
starts = np.ones((N_layer, number_of_slots)) * (range_of_slots * t_stim)
durations = np.ones((N_layer, number_of_slots)) * t_stim
# rates = np.random.randint(1, 5, size=(N_layer, number_of_slots))
rates = x_test.T
input_params = {
    "rates": rates,
    "durations": durations,
    "starts": starts
}
# scaling rates
# TODO parameterise this
rates = rates / 2

populations, projections = restore_simulator_from_file(
    sim, args.model,
    is_input_vrpss=True,
    vrpss_cellparams=input_params)
sim.set_number_of_neurons_per_core(SpikeSourcePoissonVariable, 16)
sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 16)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 64)
# set up recordings for other layers if necessary
populations[1].record("spikes")
populations[2].record("spikes")
populations[3].record("spikes")
spikes_dict = {}
neo_spikes_dict = {}
sim.run(runtime)
for pop in populations[1:]:
    spikes_dict[pop.label] = pop.spinnaker_get_data('spikes')
for pop in populations[1:]:
    neo_spikes_dict[pop.label] = pop.get_data('spikes')

# save results

if args.result_filename:
    results_filename = args.result_filename
else:
    results_filename = "mnist_results"
    if args.suffix:
        results_filename += args.suffix
    else:
        import pylab
        now = pylab.datetime.datetime.now()
        results_filename += "_"+now.strftime("_%H%M%S_%d%m%Y")

np.savez_compressed(os.path.join(args.result_dir, results_filename),
                    spikes_dict=spikes_dict,
                    neo_spikes_dict=neo_spikes_dict,
                    y_test=y_test,
                    N_layer=N_layer,
                    t_stim=t_stim)
sim.end()
