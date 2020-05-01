import argparse
from pynn_object_serialisation.functions import extract_parameters

#Parses filename from arg
parser = argparse.ArgumentParser(
    description='parameter extractor argparser',
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('model', type=str,
            help='model to extract parameters from')

parser.add_argument('output_dir', type=str,
            help='the root directory for the output')

args = parser.parse_args()


#runs the parameter extractor on them
try:
    extract_parameters(args.model, args.output_dir)
except Exception as e:
    print("Something went wrong.\n")
    print(e)
