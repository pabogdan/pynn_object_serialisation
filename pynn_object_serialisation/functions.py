from spinn_front_end_common.utilities import globals_variables
import json  # used for saving and loading json description of PyNN network
import pydoc  # used to retrieve Class from string
import numpy as np
import pynn_object_serialisation.serialisation_utils as utils
from spynnaker8.extra_models import SpikeSourcePoissonVariable
from spynnaker8 import SpikeSourceArray, SpikeSourcePoisson

DEFAULT_RECEPTOR_TYPES = ["excitatory", "inhibitory"]


def intercept_simulator(sim, output_filename=None, cellparams=None,
                        post_abort=False, custom_params=None):
    # intercept object and pickle
    current_simulator = globals_variables.get_simulator()
    sim = current_simulator
    projections = current_simulator.projections
    populations = current_simulator.populations
    custom_params = custom_params or {}
    network_dict = {}
    network_dict['populations'] = {}
    network_dict['projections'] = {}
    network_dict['setup'] = {}
    network_dict['recordings'] = {}
    network_dict['connectivity_file'] = ''
    network_dict['front_end_versions'] = current_simulator._front_end_versions
    network_dict['custom_params'] = custom_params
    # save setup info
    network_dict['setup']['machine_time_step'] = current_simulator.machine_time_step
    network_dict['setup']['min_delay'] = current_simulator.min_delay
    network_dict['setup']['max_delay'] = current_simulator.max_delay

    # Linking dicts
    _id_to_count = {}
    _projection_id_to_connectivity = {}
    _population_id_to_parameters = {}
    _spike_source_to_activity = {}

    # save population info
    for count, pop in enumerate(populations):
        network_dict['populations'][count] = {}
        p_id = id(pop)
        network_dict['populations'][count]['id'] = p_id
        network_dict['populations'][count]['label'] = pop.label
        network_dict['populations'][count]['n_neurons'] = pop.size
        network_dict['populations'][count]['cellclass'] = \
            utils._type_string_manipulation(str(type(pop.celltype)))
        if isinstance(pop.celltype, SpikeSourcePoissonVariable):
            network_dict['populations'][count]['cellparams'] = str(id(pop))
            _population_id_to_parameters[str(p_id)] = \
                utils._trundle_through_neuron_information(pop)
        else:
            _population_id_to_parameters[str(p_id)] = \
                utils._trundle_through_neuron_information(
                    pop)
        _id_to_count[id(pop)] = count
        # TODO extra info for PSS

        # Implement later
        network_dict['populations'][count]['structure'] = None
        # network_dict['populations'][count]['constraints'] = pop.constraints
        # Recording
        try:
            network_dict['populations'][count]['recording_variables'] = \
                pop._vertex._neuron_recorder.recording_variables
        except Exception:
            network_dict['populations'][count]['recording_variables'] = None

    # save projection info
    for count, proj in enumerate(projections):
        network_dict['projections'][count] = {}
        network_dict['projections'][count]['id'] = id(proj)
        network_dict['projections'][count]['receptor_type'] = \
            proj._synapse_information.synapse_type
        # TODO additional info req for STDP / Structural Plasticity
        network_dict['projections'][count]['synapse_dynamics'] = \
            utils._type_string_manipulation(
                str(type(proj._synapse_information.synapse_dynamics)))
        network_dict['projections'][count]['synapse_dynamics_constructs'] = {}
        utils._trundle_through_synapse_information(
            proj._synapse_information.synapse_dynamics,
            network_dict['projections'][count]['synapse_dynamics_constructs'])
        network_dict['projections'][count]['connector_id'] = id(
            proj._synapse_information.connector)
        network_dict['projections'][count]['connector_type'] = \
            utils._type_string_manipulation(
                str(type(proj._synapse_information.connector)))
        network_dict['projections'][count]['pre_id'] = id(proj.pre)
        network_dict['projections'][count]['pre_number'] = _id_to_count[id(
            proj.pre)]
        # Help readability
        network_dict['projections'][count]['pre_label'] = \
            network_dict['populations'][_id_to_count[id(proj.pre)]]['label']
        network_dict['projections'][count]['post_id'] = id(proj.post)
        network_dict['projections'][count]['post_number'] = _id_to_count[id(
            proj.post)]
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
                            **_projection_id_to_connectivity,
                            **_population_id_to_parameters)

    if post_abort:
        import sys
        sys.exit()


def restore_simulator_from_file(sim, filename, prune_level=1.,
                                is_input_vrpss=False,
                                vrpss_cellparams=None,
                                replace_params=None):
    replace_params = replace_params or {}

    if not is_input_vrpss and vrpss_cellparams:
        raise AttributeError("Undefined configuration. You are passing in "
                             "parameters for a Variable Rate Poisson Spike "
                             "Source, but you are not setting "
                             "is_input_vrpss=True")
    # Objects and parameters
    projections = []
    populations = []

    # Load the data from disk
    with open(filename + ".json", "r") as read_file:
        json_data = json.load(read_file)
    # Load connectivity data from disk
    connectivity_data = np.load(filename + ".npz", allow_pickle=True)

    no_pops = len(json_data['populations'].keys())
    no_proj = len(json_data['projections'].keys())
    # setup
    setup_params = json_data['setup']
    sim.setup(setup_params['machine_time_step'] / 1000.,
              setup_params['min_delay'],
              setup_params['max_delay'])
    # could set global constraints TODO

    try:
        custom_params = json_data['custom_params']
    except KeyError:
        custom_params = {}
    # set up populations
    print("Population reconstruction begins...")
    for pop_no in range(no_pops):
        pop_info = json_data['populations'][str(pop_no)]
        p_id = pop_info['id']
        pop_cellclass = pydoc.locate(pop_info['cellclass'])
        print("Reconstructing pop", pop_info['label'], "containing",  pop_info['n_neurons'], "neurons")
        if is_input_vrpss and (
                pop_cellclass is SpikeSourcePoissonVariable or
                pop_cellclass is SpikeSourceArray or
                pop_cellclass is SpikeSourcePoisson):

            print("--Going to use a VRPSS for this reconstruction ...")
            print("--VRPSS is set to have", pop_info['n_neurons'], "neurons")
            print("--and is labeled as ", pop_info['label'])

            pop_cellclass = SpikeSourcePoissonVariable
            pop_cellparams = vrpss_cellparams
        elif pop_cellclass is SpikeSourcePoissonVariable:
            pop_cellparams = connectivity_data[pop_info['cellparams']].ravel()[0]
        else:
            pop_cellparams = \
                connectivity_data[str(p_id)].ravel()[0]

        for k in replace_params.keys():
            if k in pop_cellparams.keys():
                pop_cellparams[k] = replace_params[k]

        populations.append(
            sim.Population(
                pop_info['n_neurons'],
                pop_cellclass,
                pop_cellparams,
                label=pop_info['label']
            )
        )
        # set up recordings
        recording_variables = pop_info['recording_variables']
        if recording_variables:
            populations[pop_no].record(recording_variables)

    # set up projections
    print("\n\n\nProjection reconstruction begins...")
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
    print("Reconstruction complete!")
    return populations, projections, custom_params


def get_input_size(sim):
    if isinstance(sim, str):
        # Load the data from disk
        with open(sim + ".json", "r") as read_file:
            json_data = json.load(read_file)
        # Load connectivity data from disk
        connectivity_data = np.load(sim + ".npz", allow_pickle=True)
        no_pops = len(json_data['populations'].keys())
    else:
        print("sim should be file")

    input_layer = json_data['populations']['0']

    assert input_layer['label'] == 'InputLayer', "First layer is not input layer"

    return input_layer['n_neurons']


def get_params_from_serialisation(sim, key):
    """Gets a given parameter from an intercepted sim.
    """

    if isinstance(sim, str):
        # Load the data from disk
        with open(sim + ".json", "r") as read_file:
            json_data = json.load(read_file)
        # Load connectivity data from disk
        connectivity_data = np.load(sim + ".npz", allow_pickle=True)
        no_pops = len(json_data['populations'].keys())
    else:
        print("sim should be file")

    for pop_no in range(no_pops):
        pop_info = json_data['populations'][str(pop_no)]
        p_id = pop_info['id']
        pop_cellclass = pydoc.locate(pop_info['cellclass'])
        if str(p_id) not in connectivity_data.files:
            continue
        else:
            pop_cellparams = connectivity_data[str(p_id)].ravel()[0]
        try:
            print(pop_cellparams[key])
        except Exception:
            print("Param {} not found".format(key))
    return params


def get_rescaled_i_offset(i_offset, new_runtime, old_runtime=1000):
    """Adjusts biases to the runtime of the simulation. This assumes no leak and no changes.
    """

    scaling_factor = old_runtime / new_runtime

    return i_offset * scaling_factor


def set_i_offsets(populations, new_runtime, old_runtime=1000):
    for population in populations:
        try:
            i_offset = population.get('i_offset')
            population.set(i_offset=get_rescaled_biases(i_offset, new_runtime))
        except Exception:
            pass
