import pyNN
from spinn_front_end_common.utilities import globals_variables
import json  # used for saving and loading json description of PyNN network
import pydoc  # used to retrieve Class from string
import numpy as np
import pynn_object_serialisation.serialisation_utils as utils

DEFAULT_RECEPTOR_TYPES=["excitatory", "inhibitory"]



def intercept_simulator(sim, output_filename=None, cellparams=None,
                        post_abort=False):
    # intercept object and pickle
    current_simulator = globals_variables.get_simulator()
    projections = current_simulator.projections
    populations = current_simulator.populations

    network_dict = {}
    network_dict['populations'] = {}
    network_dict['projections'] = {}
    network_dict['setup'] = {}
    network_dict['recordings'] = {}
    network_dict['connectivity_file'] = ''
    network_dict['front_end_versions'] = current_simulator._front_end_versions

    # save setup info
    network_dict['setup']['machine_time_step'] = current_simulator.machine_time_step
    network_dict['setup']['min_delay'] = current_simulator.min_delay
    network_dict['setup']['max_delay'] = current_simulator.max_delay

    # Linking dicts
    _id_to_count = {}
    _projection_id_to_connectivity = {}
    _spike_source_to_activity = {}

    # save population info
    for count, pop in enumerate(populations):
        network_dict['populations'][count] = {}
        network_dict['populations'][count]['id'] = id(pop)
        network_dict['populations'][count]['label'] = pop.label
        network_dict['populations'][count]['n_neurons'] = pop.size
        network_dict['populations'][count]['cellclass'] = \
            utils._type_string_manipulation(str(type(pop.celltype)))
        _id_to_count[id(pop)] = count
        # TODO extra info for PSS
        utils._trundle_through_neuron_information(pop, network_dict['populations'][count])
        # Implement later
        network_dict['populations'][count]['structure'] = None
        # network_dict['populations'][count]['constraints'] = pop.constraints
        # Recording
        try:
            network_dict['populations'][count]['recording_variables'] = \
                pop._vertex._neuron_recorder.recording_variables
        except:
            network_dict['populations'][count]['recording_variables'] = None


    # save projection info
    for count, proj in enumerate(projections):
        network_dict['projections'][count] = {}
        network_dict['projections'][count]['id'] = id(proj)
        network_dict['projections'][count]['receptor_type'] = \
            proj._synapse_information.synapse_type
        # TODO additional info req for STDP / Structural Plasticity
        network_dict['projections'][count]['synapse_dynamics'] = \
            utils._type_string_manipulation(str(type(proj._synapse_information.synapse_dynamics)))
        network_dict['projections'][count]['synapse_dynamics_constructs'] = {}
        utils._trundle_through_synapse_information(
            proj._synapse_information.synapse_dynamics,
            network_dict['projections'][count]['synapse_dynamics_constructs'])
        network_dict['projections'][count]['connector_id'] = id(proj._synapse_information.connector)
        network_dict['projections'][count]['connector_type'] = \
            utils._type_string_manipulation(str(type(proj._synapse_information.connector)))
        network_dict['projections'][count]['pre_id'] = id(proj.pre)
        network_dict['projections'][count]['pre_number'] = _id_to_count[id(proj.pre)]
        # Help readability
        network_dict['projections'][count]['pre_label'] = \
            network_dict['populations'][_id_to_count[id(proj.pre)]]['label']
        network_dict['projections'][count]['post_id'] = id(proj.post)
        network_dict['projections'][count]['post_number'] = _id_to_count[id(proj.post)]
        # Help readability
        network_dict['projections'][count]['post_label'] = \
            network_dict['populations'][_id_to_count[id(proj.post)]]['label']

        # Implement later
        network_dict['projections'][count]['space'] = None
        network_dict['projections'][count]['source'] = None

        _projection_id_to_connectivity[str(id(proj))] = \
            proj._synapse_information.connector.conn_list



    if output_filename:
        if output_filename[-5:] == ".json":
            output_filename = output_filename[:-5]
        network_dict['connectivity_file'] = output_filename + ".npz"
        with open(output_filename + ".json", 'w') as json_file:
            json.dump(network_dict, json_file)
            json_data = json.dumps(network_dict)
        # save connectivity information
        np.savez_compressed(output_filename,
                            json_data=json_data,
                            **_projection_id_to_connectivity)

    if post_abort:
        import sys; sys.exit()


def restore_simulator_from_file(sim, filename, prune_level=1):
    # Objects and parameters
    projections = []
    populations = []

    # Load the data from disk
    with open(filename + ".json", "r") as read_file:
        json_data = json.load(read_file)
    # Load connectivity data from disk
    connectivity_data = np.load(filename + ".npz")

    no_pops = len(json_data['populations'].keys())
    no_proj = len(json_data['projections'].keys())

    # setup
    setup_params = json_data['setup']
    sim.setup(setup_params['machine_time_step']/1000.,
              setup_params['min_delay'],
              setup_params['max_delay'])
    # could set global constraints TODO

    # set up populations
    for pop_no in range(no_pops):
        pop_info = json_data['populations'][str(pop_no)]
        populations.append(
            sim.Population(
                pop_info['n_neurons'],
                pydoc.locate(pop_info['cellclass']),
                pop_info['cellparams'],
                label=pop_info['label']
            )
        )
        # set up recordings
        recording_variables = pop_info['recording_variables']
        if recording_variables:
            populations[pop_no].record(recording_variables)

    # set up projections
    for proj_no in range(no_proj):
        # temporary utility variable
        proj_info = json_data['projections'][str(proj_no)]
        # id of projection used to retrieve from list connectivity
        _conn = utils._prune_connector(connectivity_data[str(proj_info['id'])],
                                       prune_level=prune_level)

        # build synapse dynamics
        synapse_dynamics = utils._build_synapse_info(sim, proj_info)
        # create the projection
        projections.append(
            sim.Projection(
                populations[proj_info['pre_number']],  # pre population
                populations[proj_info['post_number']],  # post population
                pydoc.locate(proj_info['connector_type'])(_conn),  # connector
                synapse_type=synapse_dynamics,
                source=proj_info['source'],
                receptor_type=DEFAULT_RECEPTOR_TYPES[proj_info['receptor_type']],
                space=proj_info['space']
            )
        )

    connectivity_data.close()
    return populations, projections

