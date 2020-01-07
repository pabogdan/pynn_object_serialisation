import numpy as np
import matplotlib.pyplot as plt
from builtins import Exception
from Onboard.KeyboardPopups import LayoutPopup

class OutputDataProcessor():
    ''' A class to represent the output of a serialised model and to
    alllow for easier processing.
    '''

    def __init__(self, data):
        if type(data) == str:
            self.data = np.load(data, allow_pickle=True)
        if type(data) == np.array :
            self.data = data
             
        self.raw_data = self.data['raw_data'][0]
        self.duration = self.data['duration']
        self.dt = self.data['dt']
        self.batch_size = self.data['batch_size']
        self.runtime = self.batch_size*self.duration
        self.spikes_dict = self.reconstruct_spikes_dict()
        self.layer_names = list(self.spikes_dict.keys())

        self.order_layer_names()
        

        self.delay = self.get_delay()
        self.input_layer_name = self.layer_names[0]
        self.output_layer_name = self.layer_names[-1]
        self.input_layer_shape = (32, 32, 3)
        self.input_spikes = self.spikes_dict[self.input_layer_name]
        self.output_spikes = self.spikes_dict[self.output_layer_name]
#         self.y_test = np.array(self.data['y_test'][:self.batch_size], dtype=np.int8)
#         self.y_pred = np.array(self.get_batch_predictions(), dtype=np.int8) 
#         labels = np.load(
#             '/mnt/snntoolbox/snn_toolbox_private/examples/models/05-mobilenet_dwarf_v1/label_names.npz')
#         self.label_names = labels['arr_0']
#         labels.close()
# 
#         x_test_file = np.load(
#             '/mnt/snntoolbox/snn_toolbox_private/examples/models/05-mobilenet_dwarf_v1/x_test.npz')
#         self.x_test = x_test_file['arr_0']
#         x_test_file.close()
#         
#         self.accuracy = self.get_accuracy()

    def get_accuracy(self):
        import numpy as np 
        correct_count = np.count_nonzero(self.y_test[:self.batch_size] == self.y_pred)
        return correct_count /self.batch_size

    def order_layer_names(self):
        self.layer_names.sort()
        self.layer_names.insert(0, self.layer_names.pop(-1))
        
    def get_shape_from_name(name):
        shape_string = name.split('_')
        if len(shape_string) < 2:
            return
        else:
            shape_string = shape_string[1]
        shape_list = shape_string.split('x')
        return tuple([int(item) for item in shape_list])

    def reconstruct_spikes_dict(self):
        output = {}
        #for batch in range(self.raw_data[0].shape[0]):
        for layer in self.raw_data.keys():
            output[layer] = self.raw_data[layer][0]['spikes'] 
        return output

    def get_delay(self):
        #Cannnot currently calculate model with delays other than 1 ms between layers

        return (len(self.layer_names) - 1) #* self.dt

    def get_bounds(self, bin_number):
        lower_end_bin_time = bin_number * self.duration+ self.delay
        higher_end_bin_time = (bin_number + 1) * self.duration + self.delay
        if higher_end_bin_time > self.runtime:
            higher_end_bin_time = self.runtime
            #print('Final bin cut off.')
        if higher_end_bin_time < lower_end_bin_time:
            print('Bin out of range of runtime')
            return self.runtime, self.runtime
        return lower_end_bin_time, higher_end_bin_time
    
    def get_spikes(self, lower_end_bin_time=0, higher_end_bin_time=None, layer=0):
        if higher_end_bin_time is None:
            higher_end_bin_time = self.runtime        
        if type(layer) == int:
            layer_name = self.layer_names[layer]
        elif type(layer) == str:
            layer_name = layer
        else:
            raise Exception("Layer given is neither an int nor a string")
        spikes = self.spikes_dict[layer_name]
        output = spikes[np.where((spikes[:, 1] >= lower_end_bin_time) & (
            spikes[:, 1] < higher_end_bin_time))]
        output = np.asarray(output).astype(int)
        return output
    
    def get_all_spikes(self, layer=0):
        return self.get_spikes(0, self.runtime, layer)

    def get_bin_spikes(self, bin_number, layer_name):
        '''Returns the spike train data for a given layer and bin'''
        bounds = self.get_bounds(bin_number)
        return self.get_spikes(*bounds, layer_name)

    def get_counts(self, bin_number, layer_name, minlength= 28**2):
        '''Returns the counts per neuron per bin in a given layer'''
        spikes = self.get_bin_spikes(bin_number, layer_name)
        just_spikes = spikes.reshape((-1, 2))[:, 0]
        counts = np.bincount(just_spikes, minlength=minlength)
        return counts

    def get_rates(self, bin_number, layer_name, shape):
        return self.get_counts(bin_number, layer_name, shape) / self.duration

    def plot_rates(self, rates, shape = (32, 32, 3)):
        plt.imshow(rates.reshape(shape))
        plt.colorbar()
        plt.show()

    def plot_bin(self, bin_number, layer_name, shape = (10,1)):
        self.plot_rates(self.get_rates(bin_number, layer_name, np.product(shape)), shape)

    def get_prediction(self, bin_number, layer_name):
        output_size = 10
        counts = self.get_counts(bin_number, layer_name, output_size)
        if counts.max() > 0:
            return int(np.argmax(counts))
        else:
            return -1

    def get_batch_predictions(self):
        y_pred = np.ones(self.batch_size) * (-1)
        for bin_number in range(self.batch_size):
            y_pred[bin_number] = self.get_prediction(
                bin_number, self.output_layer_name)
        return y_pred

    def plot_output(self, bin_number):
        if bin_number > self.batch_size: 
            raise Exception('bin_number greater than batch_size')
            bin_number = self.batch_size-1
        output_spikes = self.get_counts(bin_number, self.output_layer_name, 10)
        #label_names = [name.decode('utf-8') for name in self.label_names]
        
        plt.bar([range(10)], output_spikes)
        plt.xticks(rotation=90)

    def get_accuracy(self):
        actual_test_labels = self.y_test[:self.batch_size].ravel()
        y_pred = self.get_batch_predictions()
        return np.count_nonzero(y_pred==actual_test_labels)/float(self.batch_size)

    def bin_summary(self, bin_number):
        self.plot_output(bin_number)
        self.plot_bin(bin_number, self.output_layer_name)
        
def main():
    p = OutputDataProcessor('/home/edwardjones/git/snn_toolbox_private/examples/raw_data_file.npz')
    print(p.get_batch_predictions())
    print(p.get_all_spikes(p.layer_names[0]))
    p.plot_bin(1,p.layer_names[0], (28,28))
    
if __name__ == "__main__":
    main()
