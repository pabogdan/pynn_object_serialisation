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

class serialised_snn (object)
    """ This class represented a serialised snn model and gives the possibility to 
    run this model serially and in parallel """

    def __init__():
        self.model_path = model_path
        self.testing_examples = testing_examples
        self.parallel_processes = parallel_processes
        self.t_stim = t_stim
        self.dataset = self.load_dataset()
        self.results_dir = results_dir
        self.base_filename_results = base_filename_results
        self.model_for_size = self.load_model()
        self.record_v = False
        self.start_index = False
        self.results = []

    def in_parallel(self, func):
        
        def init_worker():
            current = multiprocessing.current_process()
            print('Started {}'.format(current))
            if not os.path.exists('errorlog'):
                os.makedirs('errorlog')
            
            f_name = "errorlog/" + current.name +"_stdout.txt"
            g_name = "errorlog/" + current.name + "_stderror.txt"
            f = open(f_name, 'w')
            g = open(g_name, 'w')
            #old_stdout = sys.stdout
            #old_stderr = sys.stderr
            sys.stdout = f
            sys.stderr = g
        
        def inner(*args, **kwargs):
            #Make a pool
            p = Pool(initializer=init_worker, self.parallel_processes)
            #Run the pool
            p.starmap(func, zip(itertools.repeat(*args), itertools.repeat(**kwargs), list(range(0, args.testing_examples, args.chunk_size))))
        
        return inner

    def load_model(self):
        """Loads model from model path"""
        replace = None
        populations, projections, custom_params = restore_simulator_from_file(
        sim, self.model_path,
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
    
    def load_dataset(self):
        """ Loads in mnist dataset """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # reshape input to flatten data
        self.y_train = y_train
        self.y_test = y_test
        self.x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
        self.x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))

    def generate_VRPSS(self):
        """Generates variable rate poisson spike source"""
        
        N_layer = len(self.model_for_size.populations[0]) # number of neurons in input population
        testing_examples = args.chunk_size
        runtime = testing_examples * self.t_stim
        number_of_slots = int(runtime / t_stim)
        range_of_slots = np.arange(number_of_slots)
        starts = np.ones((N_layer, number_of_slots)) * (range_of_slots * t_stim)
        durations = np.ones((N_layer, number_of_slots)) * t_stim

        rates = x_test[start_index:start_index+args.chunk_size, :].T

        # scaling rates
        _0_to_1_rates = rates / float(np.max(rates))
        rates = _0_to_1_rates * args.rate_scaling

        input_params = {
        "rates": rates,
        "durations": durations,
        "starts": starts
        }

    def set_record_output(self):
        output_v = []        
        spikes_dict = {}
        neo_spikes_dict = {}
        
        for pop in populations[:]:
            pop.record("spikes")
        if self.record_v:
            populations[-1].record("v")

    def get_output(self):
        """Gets the desired outputs from SpiNNaker at end of sim and saves it"""

        for pop in populations[:]:
            spikes_dict[pop.label] = pop.spinnaker_get_data('spikes')
        if args.record_v:
            output_v = populations[-1].spinnaker_get_data('v')

        if self.start_index:
            filename = self.base_filename_results + '_' + self.start_index
        else:
            import pylab
            filename = self.base_filename_results + "_" + now.strftime("_%H%M%S_%d%m%Y")

        np.savez_compressed(os.path.join(self.result_dir, filename)),
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

    def simulate(self):
    """A wrapper around sim.run to do reset between presentations"""
        def reset_membrane_voltage():        
            for population in populations[1:]:
                population.set_initial_value(variable="v", value=0)
        
        for i in range (self.testing_examples):
            sim.run(self.t_stim)
            reset_membrane_voltage()

    
    def run(self):
        #setup results directory
        if not os.path.isdir(self.result_dir) and not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

        #restore sim from file
        self.load_model()
        
        #set variables recording
        self.record_output()
        
        self.simulate()

        #collect results
        self.get_output()

        sim.end() 

    def get_results(self):
        #check for result dir and results files
        #if they're there, run OutputDataProcessor on them
        self.results = list of OutputDataProcessors
        else:
            self.run()
            self.results = #loop over the files

    def get_total_accuracy(self):
        self.total_accuracy = np.mean([proc.get_accuracy for proc in self.results])
        
    def main(self)
        self.get_results()
        self.get_total_accuracy()
        print(self.total_accuracy)

if __name__ == "__main__":
    import mnist_argparser
    args = mnist_argparser.main()
