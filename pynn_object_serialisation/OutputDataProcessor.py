import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix

class ModelParameters(object):
    '''A class representing the parameters of an SNN model run'''

    def __init__(self, data):
        self.data = data
        self.simtime = int(self.data['simtime'])
        self.dt = self.set_dt()
        self.neo_object = self.data['neo_spikes_dict']
        self.testing_examples = self.data['testing_examples']
        self.t_stim = self.data['t_stim']
        self.duration = self.t_stim
        self.chunk_size = self.data['chunk_size'] if "chunk_size" in self.data.keys() else self.testing_examples
        self.start_index = self.data['start_index'] if "start_index" in self.data.keys() else 0
        self.y_test = np.array(self.data['y_test'][:self.testing_examples], dtype=np.int8)
        self.delay = 0

    def set_dt(self):
        try:
            self.dt = self.data['timestep']
        except:
            self.dt = 0.1

    def set_delay(self, delay):
        self.delay = delay

    def reshape_y_test(self):
        self.y_test = np.argmax(self.y_test, axis=1)


class SpikeData(object):
    ''' A class representing a number of spiketrains from an SNN run'''

    def __init__(self, data, input_shape):
        self.data = data
        self.spikes_dict = self.data['all_spikes'][()]
        self.layer_names = list(self.spikes_dict.keys())
        self.order_layer_names()
        self.input_layer_name = 'InputLayer'
        self.output_layer_name = self.layer_names[-1]
        self.input_layer_name = self.layer_names[0]
        self.output_layer_name = self.layer_names[-1]
        self.input_layer_shape = input_shape
        self.layer_shapes = self.get_layer_shapes()
        self.input_spikes = self.spikes_dict[self.input_layer_name]
        self.output_spikes = self.spikes_dict[self.output_layer_name]

    def get_batch_predictions(self, chunk_size, duration, delay):
        y_pred = np.ones(chunk_size) * (-1)
        for bin_number in range(chunk_size):
            spikes = Spiketrain(self.spikes_dict[self.output_layer_name])
            y_pred[bin_number] = spikes.get_prediction(bin_number, duration, delay)
        return y_pred

    def get_layer_shapes(self):
        from snntoolbox.simulation.utils import get_shape_from_label
        return [get_shape_from_label(label) if label not in ["InputLayer", "corr_pop"] else self.input_layer_shape for label in self.layer_names]

    def order_layer_names(self):
        self.layer_names.sort()
        self.layer_names.insert(0, self.layer_names.pop(-1))

    def get_shape_from_name(self, name):
        shape_string = name.split('_')
        if len(shape_string) < 2:
            return
        else:
            shape_string = shape_string[1]
        shape_list = shape_string.split('x')
        return tuple([int(item) for item in shape_list])

    def get_spikes_event_format(self, bin_number, layer_index, parameters):
        shape = self.layer_shapes[layer_index]
        layer_name = self.layer_names[layer_index]
        spikes = Spiketrain(self.spikes_dict[layer_name])
        bin_spikes = spikes.get_spikes(bin_number, parameters.duration, parameters.delay)
        spikes = [list() for _ in range(shape[0])]
        for spike in bin_spikes:
            spikes[spike[0]].append(spike[1])
        return spikes





    def save_spiketrain(self, chunk_size, duration, delay, start_index, output_folder='spiketrain_csvs'):

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for i in range(chunk_size):
            input = Spiketrain(self.spikes_dict(self.layer_names[0])).get_spikes(i, duration, delay)
            output = Spiketrain(self.spikes_dict(self.layer_names[-1])).get_spikes(i, duration, delay)
            offset = i*duration
            input[:,1] = input[:,1]-offset
            output[:, 1] = output[:, 1] - offset

            np.savetxt(output_folder+"/example_{}_input.csv".format(start_index+i), input, delimiter=',')
            np.savetxt(output_folder+"/example_{}_output.csv".format(start_index+i), output, delimiter=',')

#TODO make this inherit from Neo object Spiketrain and write the required functions to get the summary data out
class Spiketrain(object):
    ''' A class that represents a spiketrain with associated processing functions'''

    def __init__(self, spikes):
        self.spikes = spikes # spiketrains for a layer over all time

    def get_bounds(self, bin_number, duration, delay):
        lower_end_bin_time = bin_number * duration + delay
        higher_end_bin_time = (bin_number + 1) * duration + delay
        return lower_end_bin_time, higher_end_bin_time

    def get_spikes(self, bin_number, duration, delay):
        '''Returns the spike train data for a given bin'''
        lower_end_bin_time, higher_end_bin_time = self.get_bounds(bin_number, duration, delay)
        output = self.spikes[np.where((self.spikes[:, 1] >= lower_end_bin_time) \
                                      & (self.spikes[:, 1] < higher_end_bin_time))]
        output = np.asarray(output).astype(int)
        return output

    def get_counts(self, bin_number, duration, delay, minlength=10):
        '''Returns the counts per neuron per bin in a given layer'''
        spikes = self.get_spikes(bin_number, duration, delay)
        just_spikes = spikes.reshape((-1, 2))[:, 0]
        counts = np.bincount(just_spikes, minlength=minlength)
        return counts

    def get_prediction(self, bin_number, duration, delay):
        counts = self.get_counts(bin_number, duration, delay)
        if counts.max() > 0:
            return int(np.argmax(counts))
        else:
            return -1

    def get_rates(self, bin_number, layer_name, shape, duration):
        return self.get_counts(bin_number, layer_name, shape) / duration




class OutputDataProcessor():
    ''' A class to represent the output of a serialised model and to
    alllow for easier processing.
    '''

    def __init__(self, path, input_shape=(1024,1)):
        self.data = np.load(path, allow_pickle=True) #Load the data
        self.spikedata = SpikeData(self.data, input_shape) #Extract the spiketrains (SpikeData object)
        self.parameters = ModelParameters(self.data)  #Extracts the parameters (ModelParameters object)


        # Do what is necessary to exchange information between the two helper objects
        if len(self.parameters.y_test.shape) >1 and (self.parameters.y_test.shape[-1] == self.spikedata.layer_shapes[-1][0] or self.parameters.y_test.shape[0] == self.spikedata.layer_shapes[-1][0]):
                self.parameters.reshape_y_test()
        self.parameters.set_delay((len(self.spikedata.layer_names) - 1) * self.parameters.dt)

    def summary(self):
        print(self.spikedata.layer_names)
        print(self.parameters.testing_examples)

    def get_accuracy(self):
        correct_count = np.count_nonzero(self.parameters.y_test[:self.parameters.testing_examples] == self.spikedata.get_batch_predictions(self.parameters.chunk_size, self.parameters.duration, self.parameters.delay))
        return correct_count / self.parameters.testing_examples

    #TODO write an object for plotting

    # def plot_rates(self, rates, shape = (32, 32, 3)):
    #     rates /= rates.max()
    #     plt.imshow(rates.reshape(shape))
    #     plt.colorbar()
    #     plt.show()
    #
    # def plot_bin(self, bin_number, layer_name, shape = (10,1)):
    #     self.plot_rates(self.get_rates(bin_number, layer_name, np.product(shape)), shape)
    #
    # def plot_histogram(self, bin_mumber, layer_index):
    #     spikes = self.get_counts(bin_mumber, self.layer_names[layer_index])
    #     plt.hist(spikes)
    #     plt.show()
    #
    # def plot_spikes(self, bin_number, layer_index):
    #
    #     spikes = self.get_spikes_event_format(bin_number, layer_index)
    #     plt.eventplot(spikes, orientation='vertical')
    #     plt.show()
    #
    # def plot_output(self, bin_number):
    #     if bin_number > self.chunk_size:
    #         raise Exception('bin_number greater than number_of_examples')
    #         bin_number = self.chunk_size - 1
    #     output_spikes = self.get_counts(bin_number, self.output_layer_name, 8)
    #     if hasattr(self, 'label_names'):
    #         label_names = [name.decode('utf-8') for name in self.label_names]
    #         plt.xlabel(label_names)
    #     plt.bar(range(len(output_spikes)), output_spikes)
    #     plt.xticks(rotation=90)
    #     plt.show()
    #
    # def plot_confusion_matrix(self):
    #
    #
    #     cm = confusion_matrix(self.y_test, self.y_pred)
    #     plt.imshow(cm)
    #     plt.xlabel("True label")
    #     plt.xticks(range(np.argmax(self.y_test)))
    #     plt.ylabel("Predicted label")
    #     plt.yticks(range(np.argmax(self.y_test)))
    #     plt.show()
    #     print(cm)

    # def bin_summary(self, bin_number):
    #     self.plot_output(bin_number)
    #     self.plot_bin(bin_number, self.spikedata.output_layer_name)




if __name__ == "__main__":
    import OutputDataProcessor_argparser
    args = OutputDataProcessor_argparser.main()
    proc = OutputDataProcessor(args.data_file)
    print(proc.get_accuracy())
    if args.return_spiketrains:
        proc.spikedata.save_spiketrain(start_index=args.start_index, duration=proc.parameters.duration, chunk_size=proc.parameters.chunk_size, delay=proc.parameters.delay)

