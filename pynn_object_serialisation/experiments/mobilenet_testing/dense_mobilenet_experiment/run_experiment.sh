#!/bin/bash
nohup python ../imagenet_testing.py ../keras_mobilenet --data_dir /spinnaker_dev/ILSVRC --testing_examples 10 2>&1 &
