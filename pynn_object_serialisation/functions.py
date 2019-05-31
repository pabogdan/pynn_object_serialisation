import pyNN
from spinn_front_end_common.utilities import globals_variables
import json  # used for saving and loading json description of PyNN network
import pydoc  # used to retrieve Class from string
import numpy as np


def _type_string_manipulation(class_string):
    return class_string.split("'")[1]


def intercept_simulator(sim, output_filename=None, cellparams=None):
    # intercept object and pickle
    current_simulator = globals_variables.get_simulator()
    # Todo retrieve spike sources as well
    projections = current_simulator.projections
    populations = current_simulator.populations
    # TODO retrieve textual description of:
    # populations_types =
    # TODO save textual description
    x = projections[0]

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
            _type_string_manipulation(str(type(pop.celltype)))
        _id_to_count[id(pop)] = count
        network_dict['populations'][count]['cellparams'] = pop._cellparams
        # Implement later
        network_dict['populations'][count]['structure'] = None
        # network_dict['populations'][count]['constraints'] = pop.constraints

    # save projection info
    for count, proj in enumerate(projections):
        network_dict['projections'][count] = {}
        network_dict['projections'][count]['id'] = id(proj)
        network_dict['projections'][count]['receptor_type'] = \
            proj._synapse_information.synapse_type
        network_dict['projections'][count]['synapse_dynamics'] = \
            _type_string_manipulation(str(type(proj._synapse_information.synapse_dynamics)))
        network_dict['projections'][count]['connector_id'] = id(proj._synapse_information.connector)
        network_dict['projections'][count]['connector_type'] = \
            _type_string_manipulation(str(type(proj._synapse_information.connector)))
        network_dict['projections'][count]['pre_id'] = id(proj.pre)
        network_dict['projections'][count]['pre_number'] = _id_to_count[id(proj.pre)]
        network_dict['projections'][count]['post_id'] = id(proj.post)
        network_dict['projections'][count]['post_number'] = _id_to_count[id(proj.post)]
        # Implement later
        network_dict['projections'][count]['space'] = None
        network_dict['projections'][count]['source'] = None
        _projection_id_to_connectivity[str(id(proj))] = \
            proj._synapse_information._connector.conn_list

    if output_filename:
        if output_filename[-5:] == ".json":
            output_filename = output_filename[:-5]
        with open(output_filename + ".json", 'w') as json_file:
            json.dump(network_dict, json_file)
            json_data = json.dumps(network_dict)
        # save connectivity information
        np.savez_compressed(output_filename,
                            json_data=json_data,
                            **_projection_id_to_connectivity)



def restore_simulator_from_file(filename):
    pass
