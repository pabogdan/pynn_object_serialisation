#!/bin/bash
nohup python ../imagenet_testing.py ../networks/keras_mobilenet \
      --number_of_boards 90 --data_dir /spinnaker_dev/ILSVRC --testing_examples 10 --timescale 1000 > dense.out 2>&1 &
