import matplotlib as plt
from pynn_object_serialisation.OutputDataProcessor import OutputDataProcessor

output_file= "results/model_name_trained_model_of_lenet_300_100_relu_crossent_dense_adam_run_serialised_t_stim_1000_rate_scaling_100.0_tsf_1.0_testing_examples_1_dt_1.0/mnist_results_0.npz"

data = OutputDataProcessor(output_file)

data.plot_bin(0, 'corr_pop', (28,28))

plt.show()