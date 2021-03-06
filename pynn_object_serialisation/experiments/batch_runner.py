"""
Batch runner adapted from '
'https://github.com/pabogdan/neurogenesis/blob/master/synaptogenesis/batch_argparser.py'
'and '
'https://github.com/pabogdan/neurogenesis/blob/master/synaptogenesis/batch_runner.py
"""

import subprocess
import os
import numpy as np
import sys
import hashlib
import pylab as plt
from pynn_object_serialisation.experiments.batch_argparser import *
import shutil
import ntpath

if not args.model_script:
    raise AttributeError("You need to specify the path to the script to run ("
                         "e.g. mnist_testing.py or imagenet_testing.py)")

currrent_time = plt.datetime.datetime.now()
string_time = currrent_time.strftime("%H%M%S_%d%m%Y")

if args.suffix:
    suffix = args.suffix
else:
    suffix = hashlib.md5(string_time.encode('utf-8')).hexdigest()

# Some constants
NO_CPUS = args.no_cpus
MAX_CONCURRENT_PROCESSES = args.max_processes

POISSON_PHASE = 0
PERIODIC_PHASE = 1
PHASES = [POISSON_PHASE, PERIODIC_PHASE]
PHASES_NAMES = ["poisson", "periodic"]
PHASES_ARGS = [None, "--periodic_stimulus"]

concurrently_active_processes = 0

# Compute total number of runs
total_runs = len(args.models) * args.no_slices

parameters_of_interest = {
}

log_calls = []

# making a directory for this experiment
dir_name = "{}_@{}".format("parallel_run", suffix)
print("=" * 80)
print("TOTAL RUNS", total_runs)
if not os.path.isdir(dir_name):
    print("MKDIR", dir_name)
    os.mkdir(dir_name)
else:
    print("FOLDER ALREADY EXISTS. RE-RUNNING INCOMPLETE JOBS.")
print("CHDIR", dir_name)
os.chdir(dir_name)
print("GETCWD", os.getcwd())

for network in args.models:
    # making a directory for this experiment
    dir_name = "{}_@{}".format(ntpath.basename(network), suffix)
    print("=" * 80)
    print("TOTAL RUNS", total_runs)
    if not os.path.isdir(dir_name):
        print("MKDIR", dir_name)
        os.mkdir(dir_name)
    else:
        print("FOLDER ALREADY EXISTS. RE-RUNNING INCOMPLETE JOBS.")
    print("CHDIR", dir_name)
    os.chdir(dir_name)
    print("GETCWD", os.getcwd())
    print("-" * 80)
    for slice in range(args.no_slices):
        curr_params = {
            'network': network,
            'slice': slice
        }
        filename = "{}_slice_{}".format(ntpath.basename(network), slice)
        # making a directory for this individual experiment
        prev_run = True
        if os.path.isdir(filename) and os.path.isfile(
                os.path.join(filename, "structured_provenance.csv")):
            print("Skipping", filename)
            continue
        elif not os.path.isdir(filename):
            os.mkdir(filename)
            prev_run = False
        os.chdir(filename)
        print("GETCWD", os.getcwd())
        shutil.copyfile("../../../spynnaker.cfg", "spynnaker.cfg")

        concurrently_active_processes += 1
        null = open(os.devnull, 'w')
        print("Run ", concurrently_active_processes, "...")

        call = [sys.executable,
                os.path.join("../../../", args.model_script),
                os.path.join("../../../", network),
                '-o', filename,
                '--no_slices', str(args.no_slices),
                '--curr_slice', str(slice),
                '--timestep', str(args.timestep),
                '--t_stim', str(args.t_stim),
                '--rate_scaling', str(args.rate_scaling)
        ]

        if args.reset_v:
            call.append('--reset_v')
        print("CALL", call)
        log_calls.append((call, filename, curr_params))
        if (concurrently_active_processes % MAX_CONCURRENT_PROCESSES == 0
                or concurrently_active_processes == total_runs):
            # Blocking
            with open("results.out", "wb") as out, open("results.err", "wb") as err:
                subprocess.call(call, stdout=out, stderr=err)
            print("{} sims done".format(concurrently_active_processes))
        else:
            # Non-blocking
            with open("results.out", "wb") as out, open("results.err", "wb") as err:
                subprocess.Popen(call, stdout=out, stderr=err)
        os.chdir("..")
        print("=" * 80)
        # TODO block if re-running simulations and not yet done (how would I know?)
    os.chdir("..")
print("All done!")

end_time = plt.datetime.datetime.now()
total_time = end_time - currrent_time
np.savez_compressed("batch_{}".format(suffix),
                    parameters_of_interest=parameters_of_interest,
                    total_time=total_time,
                    log_calls=log_calls,
                    argparser=vars(args))
