#!/bin/bash
nohup python ../imagenet_testing.py ../networks/trained_model_of_mobilenet_relu_crossent_sparse_soft_adam_SOFT_lr_schedule_from_mnet_serialised \
      --number_of_boards 90 --data_dir /spinnaker_dev/ILSVRC --testing_examples 10 --timescale 1000 > soft.out 2>&1 &

