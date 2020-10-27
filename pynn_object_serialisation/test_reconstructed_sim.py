from argparser import *
try:
    import spynnaker8 as sim
except:
    import pyNN.spinnaker as sim
import pylab as plt
from pynn_object_serialisation.functions import \
    restore_simulator_from_file
import os
import numpy as np
import traceback

start_time = plt.datetime.datetime.now()
runtime = args.sim_time
current_error = "NO ERROR"
model_file_path = os.path.join(args.dir, args.model)
populations, projections = restore_simulator_from_file(
    sim, model_file_path, prune_level=args.prune_level)
try:
    sim.run(runtime)
    sim.end()
except Exception as e:
    current_error = e
    traceback.print_exc()
end_time = plt.datetime.datetime.now()
total_time = end_time - start_time
print("Total time elapsed -- " + str(total_time))
suffix = end_time.strftime("_%H%M%S_%d%m%Y")
filename = "reconstruction_results"
filename += "_pruned_at_" + str(args.prune_level)
np.savez_compressed(filename,
                    total_time=total_time,
                    model_file_path=model_file_path,
                    prune_level=args.prune_level,
                    exception=str(current_error))

if __name__ == "__main__":
    pass
