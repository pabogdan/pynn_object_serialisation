import keras
from keras.datasets import mnist, cifar10, cifar100
import keras
# usual sPyNNaker imports

try:
    import spynnaker8 as sim
except Exception:
    import pyNN.spiNNaker as sim
import matplotlib.pyplot as plt
from pynn_object_serialisation.functions import \
    restore_simulator_from_file, set_i_offsets
from spynnaker8.extra_models import SpikeSourcePoissonVariable
from radioisotopedatatoolbox.DataGenerator import RandomIsotopeFlyBys
import numpy as np
import os


#This is a hacky script that is designed to get the accuracy relatively quickly

def convert_rate_array_to_PSS(input_rates: np.array, max_rate=1000, duration=1000):
    print("Generating PSS...")
    print("Rate scaling: {}".format(max_rate))
    number_of_examples = 1
    number_of_input_neurons = input_rates.shape[0]
    input_rates = input_rates*max_rate
    start_values = np.zeros(input_rates.shape)
    durations = np.repeat(duration*1000, number_of_input_neurons)
    durations = np.expand_dims(durations, axis=1)
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
    
    x_train = np.load("dataset/x_train.npz")['arr_0']
    y_train = np.load("dataset/y_train.npz")['arr_0']
    x_test = np.load("dataset/x_test.npz")['arr_0']
    y_test = np.load("dataset/y_test.npz")['arr_0']

    
    labels = np.load("dataset/labels.npz", allow_pickle=True)['arr_0']
    v_rest = 0.0    
    # Produce parameter replacement dict
#    replace = {'e_rev_E': 1.5,
#                'tau_m': 1000.0,
#                'cm': 1.0,
#                'v_thresh': 1.0,
#                'v_rest': v_rest,
#                'i_offset': 0.0,
#                'tau_syn_I': 0.01,
#                'tau_syn_E': 0.01,
#                'tau_refrac': 0.0,
#                'v_reset': 0.0,
#                'e_rev_I': -0.5}
     
    replace={}       
    t_stim = args.t_stim
    runtime = t_stim * args.testing_examples
    
    from radioisotopedatatoolbox.DataGenerator import IsotopeRateFetcher, BackgroundRateFetcher, LinearMovementIsotope
    
    from pathlib import Path
    from os import getcwd
    path = str(Path(getcwd()).parent) + '/'
    path = "/home/edwardjones/git/RadioisotopeDataToolbox/"
    input_params = convert_rate_array_to_PSS(x_test[0], duration=args.t_stim)    
    output_v = []
    populations, projections, custom_params = restore_simulator_from_file(
        sim, args.model,
        input_type='vrpss',
        vrpss_cellparams=input_params,
        replace_params=replace,
        time_scale_factor=args.time_scale_factor)
    dt = sim.get_time_step()
    N_layer = len(populations)
    min_delay = sim.get_min_delay()
    max_delay = sim.get_max_delay()
    sim.set_number_of_neurons_per_core(SpikeSourcePoissonVariable, 32)
    sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 32)
    sim.set_number_of_neurons_per_core(sim.IF_cond_exp, 128)
    old_runtime = custom_params['runtime']
    set_i_offsets(populations, runtime, old_runtime=old_runtime)
    spikes_dict = {}
    neo_spikes_dict = {}

#    def record_output(populations, offset, output):
#        spikes = populations[-1].spinnaker_get_data('spikes')
#        spikes = spikes + [0, offset]
#        name = populations[-1].label
#        if np.shape(spikes)[0] > 0:
#            if name in list(output.keys()):
#                output[name] = np.concatenate((output, spikes))
#            else:
#                output[name] = spikes
#        return output
        
    for pop in populations[:]:
        pop.record("spikes")
#    if args.record_v:
#        populations[-1].record("v")
    
#    def reset_membrane_voltage(v_rest):        
#        for population in populations[1:]:
#            population.set_initial_value(variable="v", value=v_rest)
#        return
#    
    def load_pss(index, args):
        params = convert_rate_array_to_PSS(x_test[index],max_rate=args.rate_scaling)
        populations[0].set(rates=params['rates'])

    for presentation in range(args.testing_examples):
        print("Presenting test example {}".format(presentation))
        load_pss(presentation, args)
        sim.run(t_stim)
        reset_membrane_voltage(v_rest)
    for pop in populations[:]:
        spikes_dict[pop.label] = pop.spinnaker_get_data('spikes')
    if args.record_v:
        output_v = populations[-1].spinnaker_get_data('v')
    
    # save results
    result_filename = get_result_filename(args)
    np.savez_compressed(os.path.join(args.result_dir, result_filename),
                        output_v=output_v,
                        neo_spikes_dict=neo_spikes_dict,
                        y_test=y_test,
                        N_layer=N_layer,
                        t_stim=t_stim,
                        runtime=runtime,
                        sim_time=runtime,
                        dt=dt,
                        **spikes_dict)
    sim.end()

if __name__ == "__main__":
    import isotope_argparser
    args = isotope_argparser.main()
    run(args)

    from pynn_object_serialisation.OutputDataProcessor import OutputDataProcessor
    
    proc = OutputDataProcessor("results/"+get_result_filename(args)+'.npz')
    import pdb; pdb.set_trace()
    print(proc.get_accuracy())
