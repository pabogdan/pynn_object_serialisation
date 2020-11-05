import argparse

parser = argparse.ArgumentParser(
    description='OutputDataPreprocessor argparser',
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('data_file', type=str,
                    help='An output .npz from a run')

parser.add_argument('--return_spiketrains',
                    help='An output .npz from a run',
                    action="store_true")

def main():
    return parser.parse_args()


if __name__ == "__main__":
    main()


