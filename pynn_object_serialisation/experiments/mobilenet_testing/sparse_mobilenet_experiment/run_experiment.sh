#!/bin/bash
nohup python ../imagenet_testing.py ../sparse_mobilenet_hard_r --number_of_boards 90 --data_dir /spinnaker_dev/ILSVRC --testing_examples 10 2>&1 &
