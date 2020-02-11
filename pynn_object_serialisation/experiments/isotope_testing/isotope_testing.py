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
    restore_simulator_from_file, set_i_offsets
from spynnaker8.extra_models import SpikeSourcePoissonVariable
from radioisotopedatatoolbox.DataGenerator import RandomIsotopeFlyBys
import numpy as np
import os

def convert_rate_array_to_VRPSS(input_rates: np.array, duration=1000):
    number_of_examples = input_rates.shape[0]
    number_of_input_neurons = input_rates.shape[1]
    input_rates = np.ravel(input_rates)
    run_duration = number_of_examples * duration
    start_values = np.array(
        [range(0, run_duration, duration)] * number_of_input_neurons)
    start_values = np.ravel(start_values.T)
    durations = np.repeat(duration,
                          number_of_examples * number_of_input_neurons)

    return {
        'rates': input_rates,
        'starts': start_values,
        'durations': durations}


def run(args):

    # Checking directory structure exists
    if not os.path.isdir(
            args.result_dir) and not os.path.exists(
            args.result_dir):
        os.mkdir(args.result_dir)

    # Load data from file
    
    x_train = np.load("dataset/x_train.npz")['arr_0']
    y_train = np.load("dataset/y_train.npz")['arr_0']
    #These have no background class
    x_test = np.load("dataset/x_test.npz")['arr_0']
    y_test = np.load("dataset/y_test.npz")['arr_0']
    #These have background class
    x_test_full = np.load("dataset/x_test_full.npz")['arr_0']
    y_test_full = np.load("dataset/y_test_full.npz")['arr_0']
    
    labels = np.load("dataset/labels.npz", allow_pickle=True)['arr_0']
    
    # Produce parameter replacement dict
    replace = None
     
       
    t_stim = args.t_stim
    runtime = t_stim * args.testing_examples
    example = x_test[1] # Just doing the first one
    # Generate input params from data
    input_params = convert_rate_array_to_VRPSS(example, runtime)

    output_v = []
    populations, projections, custom_params = restore_simulator_from_file(
        sim, args.model,
        is_input_vrpss=True,
        vrpss_cellparams=input_params,
        replace_params=replace)
    dt = sim.get_time_step()
    N_layer = len(populations)
    min_delay = sim.get_min_delay()
    max_delay = sim.get_max_delay()
    sim.set_number_of_neurons_per_core(SpikeSourcePoissonVariable, 16)
    sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 16)
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 64)
    old_runtime = custom_params['runtime']
    set_i_offsets(populations, runtime, old_runtime=old_runtime)
    spikes_dict = {}
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
        results_filename = "isotope_results"
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
                        dt=dt,
                        **spikes_dict)
    sim.end()


if __name__ == "__main__":
    import isotope_argparser
    args = isotope_argparser.main()
    run(args)
