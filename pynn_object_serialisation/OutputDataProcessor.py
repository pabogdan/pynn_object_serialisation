import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy import int

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
        try:
            self.custom_params = self.data['custom_params']
        except:
            pass


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
        bin_spikes = self.get_bin_spikes(bin_number, layer_index)
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
        
    def plot_spikes(self, bin_number, layer_index, labels=None):
        
        spikes = self.get_spikes_event_format(bin_number, layer_index)
        plt.eventplot(spikes, orientation='horizontal')
        if labels is not None:
            plt.xticks(range(len(labels)), labels, rotation='vertical')


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
        
    def get_spikes_in_timestep(self, spikes, timestep):
        output_spikes = np.zeros((1, len(spikes))).astype(np.bool)
        for i, neuron in enumerate(spikes):
            if timestep in neuron:
                output_spikes[0,i] = True
        return output_spikes.astype(bool)
        
        
        
    def animate_bin(self, bin_number, layer_index):
        ''' Animates a bin '''
        path = "/home/edwardjones/git/RadioisotopeDataToolbox/"
        
        from snntoolbox.simulation.utils import get_shape_from_label
        
        bin_spikes = self.get_spikes_event_format(bin_number, layer_index)
        try:
            layer_shape = get_shape_from_label(self.layer_names[layer_index])
        except IndexError:
            layer_shape = self.input_layer_shape

        fig, (ax1, ax2, ax3) = plt.subplots(3,1)
        
        fig.set_figheight(15)
        fig.set_figwidth(15)
        
        try:
            distances = self.custom_params['distances']
            labels = self.custom_params['isotope_labels']
        except:
            labels = ['Am-241', 'Ba-133', 'Background', 'Co-60', 'Cs-137', 'Eu-152']
            from radioisotopedatatoolbox.DataGenerator import IsotopeRateFetcher, BackgroundRateFetcher, LinearMovementIsotope
            myisotope = IsotopeRateFetcher('Co-60', data_path=path)
            background = BackgroundRateFetcher(intensity=1, data_path=path)
        
        
            moving_isotope = LinearMovementIsotope(
            myisotope, background=background, path_limits=[-2, 2],
            duration=5000, min_distance=0.1)
            distances = moving_isotope.distances
            
        if labels is not None:
            labels = np.insert(labels,0, '')
            print(labels)
            ax3.set_xticklabels(labels, rotation='vertical')
        
        ax3.yaxis.set_visible(False)
        ax3.set_xlabel("Isotope classification (spikes in final layer)")
        ax1.plot(range(self.runtime), distances)
        ax1.set_ylabel('Distance between source and detector/m')
        ax1.set_xlabel('Time/ms')  
        
        ax2.eventplot(self.get_spikes_event_format(0, 0), orientation='horizontal')
        ax2.set_ylim(0, 4000)
        ax2.yaxis.set_ticks(np.arange(0, 4000, 500))
        ax2.set_ylabel("Gamma photon frequency/keV")
        ax2.set_xlabel('Time/ms')  
        
        redline_1 = ax1.axvline(x=0, c='red', alpha=0.5)
        redline_2 = ax2.axvline(x=0, c='red', alpha=0.5)
        
        a = np.zeros((1, len(bin_spikes)))
        im = ax3.imshow(a,interpolation='none', vmin=0, vmax=1)
        title = ax1.text(0,1,"")
        
        def init():
            a = np.zeros((1,len(bin_spikes)))
            im.set_array(a)
            redline_1.set_xdata(0)
            redline_2.set_xdata(0)
            title.set_text("Timestep: 0")
            return [im, redline_1, redline_2, title]
        
        # animation function.  This is called sequentially
        def animate(i):
            title.set_text("Timestep: {}".format(i))
            #a = im.get_array()
            #a=a*np.exp(-0.01*i)    # exponential decay of the values
            a = self.get_spikes_in_timestep(bin_spikes, i)
            im.set_array(a)
            redline_1.set_xdata(i)
            redline_2.set_xdata(i)
            print(i)
            return [im, redline_1, redline_2,title]

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=5000, interval=1, blit=True)
        
        # save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # installed.  The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html
        #anim.save('flyby_test.mp4', fps=50, extra_args=['-vcodec', 'libx264'])
        
        plt.show()
        
        
        
        
        
        
        
        
        
