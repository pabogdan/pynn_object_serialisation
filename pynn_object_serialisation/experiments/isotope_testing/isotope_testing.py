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


def run(args):

    # Checking directory structure exists
    if not os.path.isdir(
            args.result_dir) and not os.path.exists(
            args.result_dir):
        os.mkdir(args.result_dir)

    data_generator = RandomIsotopeFlyBys(
        args.testing_examples,
        data_path='/home/edwardjones/git/RadioisotopeDataToolbox/')
    x_test = data_generator.flybys
    y_labels = data_generator.selected_sources_labels
    reference_labels = np.load('labels.npz', allow_pickle=True)[
        'arr_0'].astype('U')
    from radioisotopedatatoolbox.DataGenerator import encode_labels
    y_test = encode_labels(y_labels, reference_labels)
    N_layer = x_test['rates'].shape[0]  # number of neurons in each population
    t_stim = args.t_stim

    runtime = args.testing_examples * t_stim
    number_of_slots = int(runtime / t_stim)
    range_of_slots = np.arange(number_of_slots)

    input_params = {
        "rates": x_test['rates'],
        "durations": x_test['durations'],
        "starts": x_test['starts']
    }

    replace = None
    # produce parameter replacement dict
    output_v = []
    populations, projections, custom_params = restore_simulator_from_file(
        sim, args.model,
        is_input_vrpss=True,
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

    def record_output(populations, offset, output):
        import pdb
        pdb.set_trace()
        spikes = populations[-1].spinnaker_get_data('spikes')
        spikes = spikes + [0, offset]
        name = populations[-1].label
        if np.shape(spikes)[0] > 0:
            if name in list(output.keys()):
                output[name] = np.concatenate((output, spikes))
            else:
                output[name] = spikes
        return output

    # number of ms to simulate in a chunk
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
                        dt=dt,
                        **spikes_dict)
    sim.end()


if __name__ == "__main__":
    import isotope_argparser
    args = isotope_argparser.main()
    run(args)