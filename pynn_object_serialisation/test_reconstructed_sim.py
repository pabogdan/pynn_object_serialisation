
"""
Synfirechain-like example
"""
from argparser import *
try:
    import spynnaker8 as sim
except:
    import pyNN.spinnaker as sim
import pylab as plt
from pynn_object_serialisation.functions import \
    restore_simulator_from_file
import os
start_time = plt.datetime.datetime.now()
runtime = args.sim_time
model_file_path = os.path.join(args.dir, args.model)
populations, projections = restore_simulator_from_file(
    sim, model_file_path, prune_level=args.prune_level)
sim.run(runtime)
sim.end()
end_time = plt.datetime.datetime.now()
total_time = end_time - start_time
print("Total time elapsed -- " + str(total_time))
if __name__ == "__main__":
    pass
