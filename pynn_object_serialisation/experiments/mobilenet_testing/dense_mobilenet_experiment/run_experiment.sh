#!/bin/bash
nohup python ../imagenet_testing.py ../keras_mobilenet --number_of_boards 90 --data_dir /spinnaker_dev/ILSVRC --testing_examples 10 2>&1 &
