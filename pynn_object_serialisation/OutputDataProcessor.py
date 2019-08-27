class OutputDataProcessor():
    ''' A class to represent the output of a serialised model and to
    alllow for easier processing.
    '''
    path = 'results/output_t_stim_2000_testing_examples_1.npz'

    def __init__(self, path):
        self.data = np.load(path)
        self.input_layer_name = 'InputLayer'
        self.output_layer_name = self.get_ordered_layer_names_from_data(
            self.data)[-1]
        self.y_test = self.data['y_test']
        self.spikes_dict = self.reconstruct_spikes_dict()
        self.t_stim = data['t_stim']
        self.runtime = int(self.data['runtime'])
        self.N_layer = int(self.data['N_layer'])
        self.neo_object = self.data['neo_spikes_dict']
        self.delay = self.get_delay()
        self.input_layer_name = self.get_ordered_layer_names(self.spikes_dict)[
            0]
        self.output_layer_name = self.get_ordered_layer_names(
            self.spikes_dict)[-1]
        self.input_layer_shape = (32, 32, 3)
        self.input_spikes = self.spikes_dict[self.input_layer_name]
        self.output_spikes = self.spikes_dict[self.output_layer_name]
        self.last_but_one_layer = self.spikes_dict[self.get_ordered_layer_names(
            self.spikes_dict)[-2]]
        self.number_of_examples = self.runtime // self.t_stim

        labels = np.load(
            '/mnt/snntoolbox/snn_toolbox_private/examples/models/05-mobilenet_dwarf_v1/label_names.npz')
        self.label_names = labels['arr_0']
        labels.close()

        x_test_file = np.load(
            '/mnt/snntoolbox/snn_toolbox_private/examples/models/05-mobilenet_dwarf_v1/x_test.npz')
        self.x_test = x_test_file['arr_0']
        x_test_file.close()

    def get_ordered_layer_names(dictionary):
        self.layer_names = list(dictionary)
        layer_names.sort()
        layer_names[0] = layer_names.pop(-1)
        return layer_names

    def get_ordered_layer_names_from_data(self):
        if self.spikes_dict is None:
            spikes_dict = self.reconstruct_spikes_dict()
        dictionary = spikes_dict.keys()
        return self.get_ordered_layer_names(dictionary)

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
            'sim_time']
        unexpected_files = [
            file for file in self.data.files if file not in expected_files]
        self.spike_dict = {
            file_name: self.data[file_name] for file_name in unexpected_files}

    def get_delay(self):
        'Cannnot currently calculate model with delays other than 1 ms between layers'

        return (self.N_layers - 1) * self.dt

    def get_bounds(self, bin_number):
        #delay = get_propagation_delay(t_stim, N_layers)
        delay = 0
        lower_end_bin_time = bin_number * self.t_stim + delay
        higher_end_bin_time = (bin_number + 1) * self.t_stim + delay
        '''maximum_time = t_stim * maximum_bins
        if higher_end_bin_time > maximum_time:
            higher_end_bin_time = maximum_time
            print('Final bin cutoff.')'''
        return lower_end_bin_time, higher_end_bin_time

    def get_bin_spikes(self, bin_number):
        lower_end_bin_time, higher_end_bin_time = self.get_bounds(bin_number)
        output = spikes[np.where((spikes[:, 1] >= lower_end_bin_time) & (
            spikes[:, 1] < higher_end_bin_time)), :]
        output = np.asarray(output).astype(int)
        return output

    def get_counts(self, spikes, bin_number):
        min_length = 3 * 32**2
        spikes = get_bin_spikes(spikes, bin_number)
        just_spikes = spikes.reshape((-1, 2))[:, 0]
        counts = np.bincount(just_spikes, minlength=minlength)
        return counts

    def get_rates(self, spikes, bin_number):
        return get_counts(spikes, bin_number) / self.t_stim

    def plot_rates(self, rates):
        shape = shape = (32, 32, 3)
        rates /= rates.max()
        plt.imshow(rates.reshape(shape))
        plt.show()

    def plot_bin(spikes, bin_number):
        self.plot_rates(self.get_rates(spikes, bin_number))

    def get_prediction(self, spikes, bin_number):
        output_size = 10
        counts = get_counts(spikes, bin_number, t_stim, 10)
        if counts.max() > 0:
            return np.argmax(counts)
        else:
            return -1

    def get_batch_predictions(self):
        actual_test_labels = self.y_test[:self.number_of_examples].ravel()
        self.y_pred = np.ones(self.number_of_examples) * (-1)
        for bin_number in range(self.number_of_examples):
            self.y_pred[bin_number] = get_prediction(
                self.output_spikes, bin_number)
