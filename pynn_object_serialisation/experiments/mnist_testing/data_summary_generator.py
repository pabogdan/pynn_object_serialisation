#This script processes the run directory at the end of the run and generates a summary data file
import argparse
from pynn_object_serialisation.OutputDataProcessor import OutputDataProcessor
import csv
import os


#Parse args
parser = argparse.ArgumentParser(
    description='data summary generator',
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('directory', type=str,
                    help='output folder')

args = parser.parse_args()
#Initialise output file name

summary_filename = os.path.join(args.directory, 'summary.csv')
accuracies = []

import fnmatch

for filename in os.listdir(args.directory):
    if fnmatch.fnmatch(filename, '*.csv'):
        continue
    data_processor = OutputDataProcessor(os.path.join(args.directory, filename))
    accuracy = data_processor.get_accuracy()
    print("File: {} Accuracy: {}".format(str(filename), str(accuracy)))
    accuracies.append(accuracy)

mean_accuracy = sum(accuracies) / len(accuracies)
output = [os.path.basename(args.directory), mean_accuracy]

with open(summary_filename, 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(output)
