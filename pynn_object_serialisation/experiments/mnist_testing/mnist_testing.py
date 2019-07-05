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

N_layer = 28**2  # number of neurons in each population
t_stim = 200
(x_train, y_train), (x_test, y_test) = mnist.load_data()
runtime = x_test.shape[0] * t_stim
number_of_slots = int(runtime / t_stim)
range_of_slots = np.arange(number_of_slots)
starts = np.ones((N_layer, number_of_slots)) * (range_of_slots * t_stim)
durations = np.ones((N_layer, number_of_slots)) * t_stim
rates = np.random.randint(1, 5, size=(N_layer, number_of_slots))
input_params = {
    "rates":rates,
    "durations":durations,
    "starts":starts
}

populations, projections = restore_simulator_from_file(sim, args.model,
                                                       is_input_vrpss=True,
                                                       vrpss_cellparams=input_params)
sim.set_number_of_neurons_per_core(SpikeSourcePoissonVariable, 256 // 16)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 128)
sim.run(runtime)
sim.end()

