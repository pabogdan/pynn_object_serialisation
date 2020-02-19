import numpy as np
import matplotlib.pyplot as plt

class OutputDataProcessor():
    ''' A class to represent the output of a serialised model and to
    alllow for easier processing.
    '''

    
    def __init__(self, path):
        self.data = np.load(path, allow_pickle=True)
        self.spikes_dict = self.reconstruct_spikes_dict()
        self.layer_names = list(self.spikes_dict.keys())
        self.order_layer_names()
        self.input_layer_name = 'InputLayer'
        self.output_layer_name = self.layer_names[-1]
        self.y_test = self.data['y_test']
        self.t_stim = self.data['t_stim']
        self.runtime = int(self.data['runtime'])
        self.N_layer = int(self.data['N_layer'])
        self.dt = 1
        #self.neo_object = self.data['neo_spikes_dict']
        self.delay = self.get_delay()
        self.input_layer_name = self.layer_names[0]
        self.output_layer_name = self.layer_names[-1]
        self.input_layer_shape = (3238,1)
        self.input_spikes = self.spikes_dict[self.input_layer_name]
        self.output_spikes = self.spikes_dict[self.output_layer_name]
        self.last_but_one_layer = self.layer_names[-2]
        self.number_of_examples = self.runtime // self.t_stim
        self.y_pred = self.get_batch_predictions() 
        self.layer_shapes = self.get_layer_shapes()

    def order_layer_names(self):
        self.layer_names.sort()
        self.layer_names.insert(0,self.layer_names.pop(-1))
        
    def get_layer_shapes(self):
        from snntoolbox.simulation.utils import get_shape_from_label
        return [get_shape_from_label(label) if label != "InputLayer" else self.input_layer_shape for label in self.layer_names]
        
    def get_shape_from_name(name):
        shape_string = name.split('_')
        if len(shape_string) < 2:
            return
        else:
            shape_string = shape_string[1]
        shape_list = shape_string.split('x')
        return tuple([int(item) for item in shape_list])

    def reconstruct_spikes_dict(self):
        '''This is necessary due to serialisation problems is spikes_dict is packaged into an array.'''
        expected_files = [
            'N_layer',
            'y_test',
            'output_v',
            'runtime',
            't_stim',
            'neo_spikes_dict',
            'sim_time',
            'dt']
        unexpected_files = [
            file for file in self.data.files if file not in expected_files]
        return {file_name: self.data[file_name] for file_name in unexpected_files}

    def get_delay(self):
        #Cannnot currently calculate model with delays other than 1 ms between layers

        return (len(self.layer_names) - 1) * self.dt

    def get_bounds(self, bin_number):
        lower_end_bin_time = bin_number * self.t_stim + self.delay
        higher_end_bin_time = (bin_number + 1) * self.t_stim + self.delay
        if higher_end_bin_time > self.runtime:
            higher_end_bin_time = self.runtime
            print('Final bin cut off.')
        return lower_end_bin_time, higher_end_bin_time

    def get_bin_spikes(self, bin_number, layer_index):
        layer_name = self.layer_names[layer_index]
        '''Returns the spike train data for a given layer and bin'''
        lower_end_bin_time, higher_end_bin_time = self.get_bounds(bin_number)
        spikes = self.spikes_dict[layer_name]
        output = spikes[np.where((spikes[:, 1] >= lower_end_bin_time) & (
            spikes[:, 1] < higher_end_bin_time))]
        output = np.asarray(output).astype(int)
        return output
    
    def get_spikes_event_format(self, bin_number, layer_index):
        shape = self.layer_shapes[layer_index]
        layer_name = self.layer_names[layer_index]
        bin_spikes = self.get_bin_spikes(bin_number, layer_name)
        spikes = [list() for _ in range(shape[0])]
        for spike in bin_spikes:
            spikes[spike[0]].append(spike[1])
        return spikes
        
    def get_counts(self, bin_number, layer_index, minlength= 3*32**2):
        '''Returns the counts per neuron per bin in a given layer'''
        spikes = self.get_bin_spikes(bin_number, layer_index)
        just_spikes = spikes.reshape((-1, 2))[:, 0]
        counts = np.bincount(just_spikes, minlength=minlength)
        return counts

    def get_rates(self, bin_number, layer_index, shape):
        return self.get_counts(bin_number, layer_index, shape) / self.t_stim

    def plot_rates(self, rates, shape = (32, 32, 3)):
        rates /= rates.max()
        plt.imshow(rates.reshape(shape))
        plt.colorbar()
        plt.show()

    def plot_bin(self, bin_number, layer_index, shape = (10,1)):
        self.plot_rates(self.get_rates(bin_number, layer_index, np.product(shape)), shape)
        
    def plot_spikes(self, bin_number, layer_index):
        
        spikes = self.get_spikes_event_format(bin_number, layer_index)
        plt.eventplot(spikes, orientation='vertical')
        plt.show()

    def get_prediction(self, bin_number, layer_index):
        output_size = 10
        counts = self.get_counts(bin_number, layer_index, output_size)
        if counts.max() > 0:
            return int(np.argmax(counts))
        else:
            return -1

    def get_batch_predictions(self):
        y_pred = np.ones(self.number_of_examples) * (-1)
        for bin_number in range(self.number_of_examples):
            y_pred[bin_number] = self.get_prediction(
                bin_number, -1)
        return y_pred

    def plot_output(self, bin_number):
        if bin_number > self.number_of_examples: 
            raise Exception('bin_number greater than number_of_examples')
            bin_number = self.number_of_examples-1
        output_spikes = self.get_counts(bin_number, -1, 10)
        if hasattr(self, 'label_names'):
            label_names = [name.decode('utf-8') for name in self.label_names]
            plt.bar(label_names, output_spikes)
        plt.xticks(rotation=90)

    def get_accuracy(self):
        actual_test_labels = self.y_test[:self.number_of_examples].ravel()
        y_pred = self.get_batch_predictions()
        return np.count_nonzero(y_pred==actual_test_labels)/float(self.number_of_examples)

    def bin_summary(self, bin_number):
        self.plot_output(bin_number)
        self.plot_bin(bin_number, -1)
        
    def animate(self, bin_number, layer_index):
        ''' Animates a bin '''
        # TODO inplement this
        from snntoolbox.simulation.utils import get_shape_from_label
        
        bin_spikes = self.get_bin_spikes(bin, layer_index)
        layer_shape = get_shape_from_label(layer_index)
        
        
        
        
        
