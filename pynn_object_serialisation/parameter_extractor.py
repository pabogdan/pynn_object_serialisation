import argparse
from pynn_object_serialisation.functions import extract_parameters
import os

#Parses filename from arg
parser = argparse.ArgumentParser(
    description='parameter extractor argparser',
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('model', type=str,
            help='model to extract parameters from')

parser.add_argument('output_dir', type=str,
            help='the root directory for the output')

parser.add_argument('output_type', type=str,
            help='the root directory for the output',
            default="csv")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

#runs the parameter extractor on them
try:
    extract_parameters(args.model, args.output_dir, args.output_type)
except Exception as e:
    print("Something went wrong.\n")
    print(e)
