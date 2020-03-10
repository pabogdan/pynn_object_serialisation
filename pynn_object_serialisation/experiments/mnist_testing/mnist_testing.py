# import keras dataset to deal with our common use cases
from keras.datasets import mnist, cifar10, cifar100
import keras
# usual sPyNNaker imports

try:
    import spynnaker8 as sim
except:
    import pyNN.spiNNaker as sim
import matplotlib.pyplot as plt
from pynn_object_serialisation.functions import \
    restore_simulator_from_file, set_i_offsets
from spynnaker8.extra_models import SpikeSourcePoissonVariable
import numpy as np
import os

def run(args):

    # Checking directory structure exists
    if not os.path.isdir(args.result_dir) and not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    N_layer = 28 ** 2  # number of neurons in each population
    t_stim = args.t_stim
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
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

    #number of ms to simulate in a chunk
    chunk_time = t_stim 
    number_chunks = runtime // chunk_time
    remainder_chunk_time = runtime % chunk_time
    '''
    for i in range(number_chunks):
        for pop in populations[:]:
            pop.record("spikes")
        if args.record_v:
            populations[-1].record("v")
        sim.run(chunk_time)
        spikes_dict = record_output(populations, i*chunk_time, spikes_dict)
        sim.reset()
    if remainder_chunk_time != 0:
        #After a sim reset the vrpss needs to have its inputs readded 
        for pop in populations[:]:
            pop.record("spikes")
        if args.record_v:
            populations[-1].record("v")

        sim.run(remainder_chunk_time)
        spikes_dict = record_output(populations, i*chunk_time, spikes_dict)
    '''
    for pop in populations[:]:
        pop.record("spikes")
    if args.record_v:
        populations[-1].record("v")

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
    # the following takes more time than spinnaker_get_data
    # for pop in populations[:]:
    #     neo_spikes_dict[pop.label] = pop.get_data('spikes')
    if args.record_v:
        output_v = populations[-1].spinnaker_get_data('v')
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
        results_filename += "_" + now.strftime("_%H%M%S_%d%m%Y")

    np.savez_compressed(os.path.join(args.result_dir, results_filename),
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
    run(args)
