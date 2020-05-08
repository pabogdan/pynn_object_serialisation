go batch_runner.py \
../../mnist_testing/networks/trained_model_of_lenet_300_100_relu_crossent_sparse_hard_adam_run_serialised \
../../mnist_testing/networks/trained_model_of_lenet_300_100_relu_crossent_sparse_hard_adam_no_rew_run_serialised \
../../mnist_testing/networks/trained_model_of_lenet_300_100_relu_crossent_sparse_soft_adam_run_serialised \
../../mnist_testing/networks/trained_model_of_lenet_300_100_relu_crossent_sparse_decay_adam_run_serialised \
../../mnist_testing/networks/trained_model_of_lenet_300_100_relu_crossent_dense_adam_run_serialised \
--model_script ../../mnist_testing/mnist_testing.py \
--max_processes 10