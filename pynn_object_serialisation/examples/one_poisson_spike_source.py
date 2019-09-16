"""
Synfirechain-like example
"""
import spynnaker8 as sim
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
from pynn_object_serialisation.functions import intercept_simulator, restore_simulator_from_file

runtime = 5000
sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0)
nNeurons = 200  # number of neurons in each population

# Spike sources
poisson_spike_source = sim.Population(nNeurons, sim.SpikeSourcePoisson(
    rate=[[50]]*nNeurons, duration=runtime), label='poisson_source')

lif_pop = sim.Population(nNeurons, sim.IF_curr_exp, label='pop_1')

loopConnections = list()
for i in range(0, nNeurons):
    singleConnection = ((i, (i + 1) % nNeurons, 1, 1))

    loopConnections.append(singleConnection)
sim.Projection(
    poisson_spike_source, lif_pop, sim.FromListConnector(loopConnections))

intercept_simulator(sim, "one_pss")

from importlib import reload

sim = reload(sim)
populations, projections = restore_simulator_from_file(sim, "one_pss")
intercept_simulator(sim, "comparison_one_pss")
sim.run(runtime)
sim.end()
