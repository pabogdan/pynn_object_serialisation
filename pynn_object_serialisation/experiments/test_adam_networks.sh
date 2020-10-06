nohup python batch_runner.py \
./mnist_testing/networks/trained_model_of_lenet_300_100_relu_crossent_dense_adam_non_neg_run_serialised \
./mnist_testing/networks/trained_model_of_lenet_300_100_relu_crossent_dense_adam_run_serialised \
./mnist_testing/networks/trained_model_of_lenet_300_100_relu_crossent_sparse_hard_adam_run_serialised \
./mnist_testing/networks/trained_model_of_lenet_300_100_relu_crossent_sparse_soft_adam_run_serialised \
./mnist_testing/networks/trained_model_of_lenet_300_100_relu_crossent_sparse_decay_adam_run_serialised \
./mnist_testing/networks/trained_model_of_lenet_300_100_relu_crossent_sparse_hard_adam_no_rew_run_serialised \
--model_script mnist_testing/mnist_testing.py \
--t_stim 2000 --no_slices 10 --timestep 1.0 --rate_scaling 100 \
--max_processes 10 > parallel_run_2000_dt_1.0.out 2>&1 &

nohup python batch_runner.py \
./mnist_testing/networks/trained_model_of_lenet_300_100_relu_crossent_dense_adam_non_neg_run_serialised \
--model_script mnist_testing/mnist_testing.py \
--t_stim 2000 --no_slices 10 --timestep 1.0 --rate_scaling 100 \
--max_processes 10 > parallel_run_2000_dt_1.0_2x_slowdown.out 2>&1 &
