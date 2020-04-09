import argparse
import os
import glob
import numpy as np
from pynn_object_serialisation import OutputDataProcessor
parser = argparse.ArgumentParser()

parser.add_argument("dir")

args = parser.parse_args()

accuracies = []

for file in glob.glob(args.dir+'/*'):
    proc=OutputDataProcessor.OutputDataProcessor(file)
    accuracies.append(proc.get_accuracy())

print(accuracies)

print(np.mean(np.array(accuracies)))
