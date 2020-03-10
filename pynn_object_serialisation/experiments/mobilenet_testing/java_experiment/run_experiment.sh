#!/bin/bash
nohup python ../imagenet_testing.py ../keras_mobilenet --testing_examples 10 --number_of_boards 90 --data_dir /spinnaker_dev/ILSVRC 2>&1 &
