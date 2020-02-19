# import keras dataset to deal with our common use cases
from keras.datasets import mnist, cifar10, cifar100
import keras
from pynn_object_serialisation.OutputDataProcessor import OutputDataProcessor
# usual sPyNNaker imports

try:
    import spynnaker8 as sim
except:
    import pyNN.spiNNaker as sim
import matplotlib.pyplot as plt
from pynn_object_serialisation.functions import \
    restore_simulator_from_file, set_i_offsets
from spynnaker8.extra_models import SpikeSourcePoissonVariable
from multiprocessing.pool import Pool
import multiprocessing
import itertools
import numpy as np
import os
import sys

def run(args, start_index):
    
    current = multiprocessing.current_process()
    print('Started {}'.format(current))
    if not os.path.exists('errorlog'):
        os.makedirs('errorlog')
    
    f_name = "errorlog/" + current.name +"_stdout.txt"
    g_name = "errorlog/" + current.name + "_stderror.txt"
    f = open(f_name, 'w')
    g = open(g_name, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = f
    sys.stderr = g

    # Checking directory structure exists
    if not os.path.isdir(args.result_dir) and not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

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

    rates = x_test[start_index:start_index+args.chunk_size, :].T
    y_test = y_test[start_index:start_index+args.chunk_size]
    # scaling rates
    _0_to_1_rates = rates / float(np.max(rates))
    rates = _0_to_1_rates * args.rate_scaling

    input_params = {
    "rates": rates,
    "durations": durations,
    "starts": starts
    }

    replace = None
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
    sim.set_number_of_neurons_per_core(sim.IF_cond_exp, 64)
    old_runtime = custom_params['runtime']
    set_i_offsets(populations, runtime, old_runtime=old_runtime)

    spikes_dict = {}
    neo_spikes_dict = {}
    
    def reset_membrane_voltage():        
        for population in populations[1:]:
            population.set(v=0)
        return

    for pop in populations[:]:
        pop.record("spikes")
    if args.record_v:
        populations[-1].record("v")
    for i in range(args.chunk_size):
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
         import pylab

        now = pylab.datetime.datetime.now()
        results_filename += "_" + now.strftime("_%H%M%S_%d%m%Y")

    np.savez_compressed(os.path.join(args.result_dir, results_filename+"_" + str(start_index)),
            output_v=output_v,
            neo_spikes_dict=neo_spikes_dict,
            y_test=y_test,
            N_layer=N_layer,
            t_stim=t_stim,
            runtime=runtime,
            sim_time=runtime,
            dt = dt,
            **spikes_dict)
    sim.end()

if __name__ == "__main__":
    import mnist_argparser
    args = mnist_argparser.main()
    #Make a pool
    p = Pool(args.number_of_threads)
    #Run the pool
    p.starmap(run, zip(itertools.repeat(args), list(range(0, args.testing_examples, args.chunk_size))))
    print("Simulations complete. Gathering data...")

    accuracies = []
    for filename in os.listdir(args.result_dir):
        data_processor = OutputDataProcessor(os.path.join(args.result_dir,filename))
        data_processor.plot_bin(0, -1)
        plt.show()
        accuracy = data_processor.get_accuracy()
        print("File: {} Accuracy: {}".format(str(filename), str(accuracy)))
        accuracies.append(accuracy)
    print("Accuracy = {}".format(str(sum(accuracies)/len(accuracies))))
