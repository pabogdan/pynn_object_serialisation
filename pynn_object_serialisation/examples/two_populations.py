import spynnaker8 as sim
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
from pynn_object_serialisation.functions import intercept_simulator


runtime = 5000
sim.setup(timestep=1.0, min_delay=1.0, max_delay=15.0)