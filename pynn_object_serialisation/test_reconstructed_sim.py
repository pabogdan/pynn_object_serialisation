
"""
Synfirechain-like example
"""
from argparser import *
try:
    import spynnaker8 as sim
except:
    import pyNN.spinnaker as sim
from pynn_object_serialisation.functions import \
    restore_simulator_from_file
import os
runtime = args.sim_time
model_file_path = os.path.join(args.dir, args.model)
populations, projections = restore_simulator_from_file(sim, model_file_path)
sim.run(runtime)
sim.end()

if __name__ == "__main__":
    pass
