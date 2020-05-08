import json
import pynn_object_serialisation.serialisation_utils as utils
from pynn_object_serialisation.functions import DEFAULT_RECEPTOR_TYPES
from pynn_object_serialisation.experiments.analysis_common import *


def post_run_analysis(filename, fig_folder, dark_background=False):
    pass


if __name__ == "__main__":
    from pynn_object_serialisation.experiments.analysis_argparser import *

    if analysis_args.input and len(analysis_args.input) > 0:
        for in_file in analysis_args.input:
            post_run_analysis(in_file, analysis_args.figures_dir,
                              dark_background=analysis_args.dark_background)
