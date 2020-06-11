#Plots everything in a master_summary file

import argparse
import matplotlib.pyplot as plt
import csv

#Parse args
parser = argparse.ArgumentParser(
    description='data summary plotter',
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('summary_file', type=str,
                    help='summary_file')

args = parser.parse_args()

data = {}



with open(args.summary_file, mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = dict(csv_reader)

for key, value in data.items():
    data[key] = float(value)

plt.bar(range(len(data)), list(data.values()), align='center', color='red')
plt.xticks(range(len(data)), list(data.keys()))
plt.show()