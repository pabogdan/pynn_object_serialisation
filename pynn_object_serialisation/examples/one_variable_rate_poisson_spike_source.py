import spynnaker8 as sim
from pynn_object_serialisation.functions import intercept_simulator, restore_simulator_from_file
from spynnaker8.extra_models import SpikeSourcePoissonVariable
import numpy as np
from brian2.units import *

runtime = 5000
sim.setup(timestep=1.0, min_delay=1.0, max_delay=10)
sim.set_number_of_neurons_per_core(SpikeSourcePoissonVariable, 256 // 16)
N_layer = 200  # number of neurons in each population

t_stim = 250
number_of_slots = int(runtime / t_stim)
range_of_slots = np.arange(number_of_slots)
slots_starts = np.ones((N_layer, number_of_slots)) * (range_of_slots * t_stim)
durations = np.ones((N_layer, number_of_slots)) * t_stim
rates = np.random.randint(1, 5, size=(N_layer, number_of_slots))

# Spike sources
poisson_spike_source = sim.Population(N_layer, SpikeSourcePoissonVariable(
    starts=slots_starts, rates=rates, durations=durations), label='poisson_source')

lif_pop = sim.Population(N_layer, sim.IF_curr_exp, label='pop_1')

conns = [[x, x] for x in range(N_layer)]
sim.Projection(
    poisson_spike_source, lif_pop, sim.FromListConnector(conns),
    sim.StaticSynapse(weight=2, delay=1))

poisson_spike_source.record(['spikes'])
lif_pop.record(['spikes'])

intercept_simulator(sim, "variable_rate_pss")
sim.run(runtime)
pss_spikes = poisson_spike_source.spinnaker_get_data('spikes')
lif_spikes = lif_pop.spinnaker_get_data('spikes')
sim.end()


from importlib import reload

sim = reload(sim)
populations, projections = restore_simulator_from_file(sim, "variable_rate_pss")
intercept_simulator(sim, "comparison_variable_rate_pss")
sim.set_number_of_neurons_per_core(SpikeSourcePoissonVariable, 256 // 16)
# PSS recording is not picked up by interception!
populations[0].record(['spikes'])
sim.run(runtime)
reconstructed_pss_spikes = populations[0].spinnaker_get_data('spikes')
reconstructed_lif_spikes = populations[1].spinnaker_get_data('spikes')
sim.end()

# compute instantenous rates
def get_inst_rate(N_layer, runtime, t_stim, post_spikes):
    per_neuron_instaneous_rates = np.empty((N_layer, int(runtime / t_stim)))
    chunk_size = t_stim
    for neuron_index in np.arange(N_layer):
        firings_for_neuron = post_spikes[
            post_spikes[:, 0] == neuron_index]
        for chunk_index in np.arange(per_neuron_instaneous_rates.shape[
                                         1]):
            per_neuron_instaneous_rates[neuron_index, chunk_index] = \
                np.count_nonzero(
                    np.logical_and(
                        firings_for_neuron[:, 1] >= (
                                chunk_index * chunk_size),
                        firings_for_neuron[:, 1] < (
                                (chunk_index + 1) * chunk_size)
                    )
                ) / (1 * (chunk_size * ms))
    # instaneous_rates = np.sum(per_neuron_instaneous_rates,
    #                           axis=0) / N_layer
    # return instaneous_rates
    return per_neuron_instaneous_rates


inst_rates_pss = get_inst_rate(N_layer, runtime, t_stim,
                               pss_spikes)
inst_rates_rec_pss = get_inst_rate(N_layer, runtime, t_stim,
                                   reconstructed_pss_spikes)
# plot spikes
import matplotlib.pyplot as plt
import matplotlib as mlib
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
# ensure we use viridis as the default cmap
plt.viridis()

# ensure we use the same rc parameters for all matplotlib outputs
mlib.rcParams.update({'font.size': 24})
mlib.rcParams.update({'errorbar.capsize': 5})
mlib.rcParams.update({'figure.autolayout': True})

minimus = np.min([np.min(inst_rates_pss), np.min(inst_rates_rec_pss)])
maximus = np.max([np.max(inst_rates_pss), np.max(inst_rates_rec_pss)])

np.savez_compressed("vrpss_results",
                    pss_spikes=pss_spikes,
                    reconstructed_pss_spikes=reconstructed_pss_spikes,
                    lif_spikes=lif_spikes,
                    reconstructed_lif_spikes=reconstructed_lif_spikes,
                    inst_rates_pss=inst_rates_pss,
                    inst_rates_rec_pss=inst_rates_rec_pss,
                    rates=rates)

print("Minimum firing rate", minimus)
print("Maximum firing rate", maximus)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(20, 8), dpi=600)
ax1.imshow(inst_rates_pss, vmin=minimus, vmax=maximus)
ax2.imshow(inst_rates_pss - inst_rates_rec_pss, vmin=minimus, vmax=maximus)
i = ax3.imshow(inst_rates_rec_pss, vmin=minimus, vmax=maximus)
ax1.set_xlabel("Pattern #")
ax2.set_xlabel("Pattern #")
ax3.set_xlabel("Pattern #")
ax1.set_ylabel("Neuron ID")
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
cbar = plt.colorbar(i, cax=cax)
plt.savefig("firing_rate_original_vs_reconstructed.png")
plt.close(fig)
