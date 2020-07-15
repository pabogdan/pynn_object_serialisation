# import keras dataset to deal with our common use cases
from keras.datasets import mnist, cifar10, cifar100
import keras
# usual sPyNNaker imports

try:
    import spynnaker8 as sim
except Exception:
    import pyNN.spiNNaker as sim
import matplotlib.pyplot as plt
from pynn_object_serialisation.functions import \
    restore_simulator_from_file, set_i_offsets, set_zero_i_offsets
from spynnaker8.extra_models import SpikeSourcePoissonVariable
from radioisotopedatatoolbox.DataGenerator import RandomIsotopeFlyBys
import numpy as np
import os

def convert_rate_array_to_VRPSS(input_rates: np.array, max_rate=1000, duration=1000):
    print("Generating VRPSS...")
    if len(input_rates.shape) < 3:
        input_rates = np.expand_dims(input_rates, -1)
    number_of_examples = input_rates.shape[0]
    number_of_input_neurons = input_rates.shape[1]
    input_rates = max_rate*(input_rates[:,...,0])
    input_rates = np.transpose(input_rates)
    run_duration = number_of_examples * duration
    start_values = np.array(
        [range(0, run_duration, duration)] * number_of_input_neurons)
    durations = np.repeat(duration,
                          number_of_examples * number_of_input_neurons)
    durations = durations.reshape(start_values.shape)
    return {
        'rates': input_rates,
        'starts': start_values,
        'durations': durations}

def get_result_filename(args):
    if args.result_filename:
        result_filename = args.result_filename
    else:
        result_filename = "isotope_results"
        if args.suffix:
            result_filename += args.suffix
        else:
            import pylab

    result_filename += "_" + str(args.start_index)
    return result_filename

def run(args):

    # Checking directory structure exists
    if not os.path.isdir(
            args.result_dir) and not os.path.exists(
            args.result_dir):
        os.mkdir(args.result_dir)

    # Load data from file
    
    x_test = np.load(args.data_dir +"x_test.npz")['arr_0']
    y_test = np.load(args.data_dir +"y_test.npz")['arr_0']

    replace = None       
    runtime = args.t_stim * args.testing_examples
    
    from pathlib import Path
    from os import getcwd
    path = str(Path(getcwd()).parent) + '/'
    
    path = "/home/edwardjones/git/RadioisotopeDataToolbox/"
    input_params = convert_rate_array_to_VRPSS(x_test[args.start_index:args.start_index+args.testing_examples], max_rate=args.rate_scaling, duration=args.t_stim)

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
    sim.set_number_of_neurons_per_core(sim.IF_curr_delta, 64)

    v_reset = 0

    replace= {'v_thresh': args.v_thresh,
              'tau_refrac': 0,
              'v_reset': v_reset,
              'v_rest': 0,
              'v':0,
              'cm': 1,
              'tau_m': 1000,
              'tau_syn_E': 0.02,
              'tau_syn_I': 0.02,
              'delay': 0}


    populations, projections, extra_params = restore_simulator_from_file(
        sim, args.model,
        input_type='vrpss',
        vrpss_cellparams=input_params,
        replace_params=replace,
        delta_input=args.delta)
    
    dt = sim.get_time_step()
    simtime = args.testing_examples * args.t_stim
    N_layer = len(populations)
    min_delay = sim.get_min_delay()
    max_delay = sim.get_max_delay()
    sim.set_number_of_neurons_per_core(SpikeSourcePoissonVariable, 16)
    sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 16)
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 128)
    sim.set_number_of_neurons_per_core(sim.IF_cond_exp, 128)
    old_runtime = extra_params['simtime'] if 'simtime' in extra_params else None
    #set_i_offsets(populations, runtime, old_runtime=old_runtime)
    set_zero_i_offsets(populations)
    spikes_dict = {}
    output_v = []
    neo_spikes_dict = {}

    def record_output(populations, offset, output):
        spikes = populations[-1].spinnaker_get_data('spikes')
        spikes = spikes + [0, offset]
        name = populations[-1].label
        if np.shape(spikes)[0] > 0:
            if name in list(output.keys()):
                output[name] = np.concatenate((output, spikes))
            else:
                output[name] = spikes
        return output
        
    for pop in populations[:]:
        pop.record("spikes")
    if args.record_v:
        populations[-1].record("v")
    
    def reset_membrane_voltage(v_reset):
        for population in populations[1:]:
            population.set_initial_value(variable="v", value=v_reset)
        return
    
    for presentation in range(args.testing_examples):
        print("Presenting test example {}".format(presentation))
        sim.run(args.t_stim)
        reset_membrane_voltage(v_reset)
    for pop in populations[:]:
        spikes_dict[pop.label] = pop.spinnaker_get_data('spikes')
    if args.record_v:
        output_v = populations[-1].spinnaker_get_data('v')

    sim_params = {
        "argparser": vars(args)
    }

    # save results
    result_filename=get_result_filename(args)
    np.savez_compressed(result_filename,
                        output_v=output_v,
                        neo_spikes_dict=neo_spikes_dict,
                        all_spikes=spikes_dict,
                        all_neurons=extra_params['all_neurons'],
                        testing_examples=args.testing_examples,
                        N_layer=N_layer,
                        no_testing_examples=args.testing_examples,
                        num_classes=10,
                        y_test=y_test,
                        input_params=input_params,
                        input_size=N_layer,
                        simtime=simtime,
                        t_stim=args.t_stim,
                        timestep=timestep,
                        time_scale_factor=timescale,
                        sim_params=sim_params,
                        init_connectivity=extra_params['all_connections'],
                        extra_params=extra_params)
    sim.end()

if __name__ == "__main__":
    import isotope_argparser
    args = isotope_argparser.main()
    run(args)

    from pynn_object_serialisation.OutputDataProcessor import OutputDataProcessor
    
    proc = OutputDataProcessor(args.result_dir +'/'+ get_result_filename(args)+'.npz')
    print(proc.get_accuracy())
