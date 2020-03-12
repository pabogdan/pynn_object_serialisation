#!/bin/bash
nohup python ../imagenet_testing.py ../networks/trained_model_of_mobilenet_relu_crossent_sparse_hard_adam_HARD_10_percent_lr_schedule_from_mnet_serialised \
      --number_of_boards 90 --data_dir /spinnaker_dev/ILSVRC --testing_examples 10 --timescale 1000 > hard_10.out 2>&1 &

