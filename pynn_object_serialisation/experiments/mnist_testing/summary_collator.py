#Collates the summary files in the from each run into a master list

import argparse
import os
import csv

#Parse args
parser = argparse.ArgumentParser(
    description='data summary collator',
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('root_directory', type=str,
                    help='output folder')

args = parser.parse_args()

#Output file

master_summary_filename = 'master_summary.csv'
data_summary_file = "summary.csv"



output = []


for root, dirs, files in os.walk(args.root_directory):
    if data_summary_file in files:
        with open(os.path.join(root, data_summary_file), 'r') as myfile:
            data = myfile.read().replace('\n', '')
            print(data)
        output.append(data)

with open(master_summary_filename, 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(output)