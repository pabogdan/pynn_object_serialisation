#!/bin/bash
nohup python ../imagenet_testing.py ../keras_mobilenet --number_of_boards 90 --testing_examples 10 --data_dir /spinnaker_dev/ILSVRC 2>&1 &
