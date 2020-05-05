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

class serialised_snn:
    """ This class represented a serialised snn model and gives the possibility to 
    run this model serially and in parallel """

    def __init__(self, args):
        self.model_path = args.model
        self.testing_examples = args.testing_examples
        self.parallel_processes = args.number_of_threads
        self.t_stim = args.t_stim
        self.runtime = self.t_stim*self.testing_examples
        self.load_dataset()
        self.result_dir = args.result_dir
        self.base_filename_results = args.result_filename
        self.input_params = {}
        self.record_v = args.record_v
        self.start_index = False
        self.time_scale_factor = args.time_scale_factor
        self.results = []
        self.N_layer = self.x_test.shape[1] # number of neurons in input population
        self.chunk_size = self.testing_examples // self.parallel_processes
        #TODO fix for case when self.testing_examples%self.parallel_processes != 0
 
    def load_model(self):
        """Loads model from model path"""
        
        self.generate_VRPSS()
        replace = None
  
        self.populations, self.projections, self.custom_params = restore_simulator_from_file(
        sim, self.model_path,
        input_type='vrpss',
        vrpss_cellparams=self.input_params,
        replace_params=replace,
        time_scale_factor=args.time_scale_factor)
        self.dt = sim.get_time_step()
        old_runtime = self.custom_params['runtime'] if 'runtime' in self.custom_params else None
        min_delay = sim.get_min_delay()
        max_delay = sim.get_max_delay()
        sim.set_number_of_neurons_per_core(SpikeSourcePoissonVariable, 16)
        sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 16)
        sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 64)
        set_i_offsets(self.populations, self.runtime, old_runtime=old_runtime)
    
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

        runtime = self.chunk_size * self.t_stim
        range_of_slots = np.arange(self.chunk_size)
        starts = np.ones((self.N_layer, self.chunk_size)) * (range_of_slots * self.t_stim)
        durations = np.ones((self.N_layer, self.chunk_size)) * self.t_stim

        rates = self.x_test[self.start_index:self.start_index+self.chunk_size, :].T

        # scaling rates
        _0_to_1_rates = rates / float(np.max(rates))
        rates = _0_to_1_rates * args.rate_scaling

        self.input_params = {
        "rates": rates,
        "durations": durations,
        "starts": starts
        }

    def set_record_output(self):
        self.output_v = []        
        self.spikes_dict = {}
        self.neo_spikes_dict = {}
        
        for pop in self.populations[:]:
            pop.record("spikes")
        if self.record_v:
            self.populations[-1].record("v")

    def get_output(self):
        """Gets the desired outputs from SpiNNaker at end of sim and saves it"""

        for pop in self.populations[:]:
            self.spikes_dict[pop.label] = pop.spinnaker_get_data('spikes')
        if args.record_v:
            self.output_v = self.populations[-1].spinnaker_get_data('v')

        if self.start_index:
            filename = self.base_filename_results + '_' + self.start_index
        else:
            from datetime import datetime
            now = datetime.now()
            filename = self.base_filename_results + "_" + now.strftime("_%H%M%S_%d%m%Y")

        np.savez_compressed(os.path.join(self.result_dir, filename),\
            output_v=self.output_v,\
            neo_spikes_dict=self.neo_spikes_dict,\
            y_test=self.y_test,\
            N_layer=self.N_layer,\
            t_stim=self.t_stim,\
            runtime=self.runtime,\
            sim_time=self.runtime,\
            dt = self.dt,\
            **self.spikes_dict)
        sim.end()

    def simulate(self):
        """A wrapper around sim.run to do reset between presentations"""
        def reset_membrane_voltage():        
            for population in self.populations[1:]:
                population.set_initial_value(variable="v", value=0)
        
        for i in range (self.testing_examples):
            sim.run(self.t_stim, start_index = self.start_index)
            reset_membrane_voltage()

    def in_parallel(self, func):
        """ Decorator to run function in parallel """
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
 
        def inner(self):
            #Make a pool
            p = Pool(initializer=init_worker, processes=self.parallel_processes)
            #Run the pool
            p.starmap(func, zip(self, list(range(0, self.testing_examples, self.chunk_size))))
            return
 
        return inner
    
    def run(self, start_index = 0):
        
        self.start_index = start_index
        #setup results directory
        if not os.path.isdir(self.result_dir) and not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

        #restore sim from file
        self.load_model()
        
        #set variables 
        self.set_record_output()
        
        self.simulate()

        #collect results
        self.get_output()

        sim.end() 

    def parallel_run(self):
        return self.in_parallel(self.run)()

    def get_results(self):
        #check for result dir and results files
        if os.path.isdir(self.result_dir) and len(os.listdir(self.result_dir))>0  and not args.force_resim:
            pass
        elif self.parallel_processes >1:
            self.parallel_run()
        else:
            self.run()
        for filename in os.listdir(self.result_dir):
             print(filename)
             self.results.append(OutputDataProcessor(self.result_dir+'/'+filename))
    
    def get_total_accuracy(self):
        import pdb; pdb.set_trace()
        self.total_accuracy = np.mean([proc.get_accuracy() for proc in self.results])
        
    def main(self):
        self.get_results()
        self.get_total_accuracy()
        print(self.total_accuracy)

if __name__ == "__main__":
    import mnist_argparser
    args = mnist_argparser.main()
    this = serialised_snn(args)
    this.main()
