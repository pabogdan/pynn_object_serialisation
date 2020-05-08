from spinn_front_end_common.utilities import globals_variables
import json  # used for saving and loading json description of PyNN network
import pydoc  # used to retrieve Class from string
import numpy as np
import pynn_object_serialisation.serialisation_utils as utils
from spynnaker8.extra_models import SpikeSourcePoissonVariable
from spynnaker8 import SpikeSourceArray, SpikeSourcePoisson
from pprint import pprint as pp
from colorama import Fore, Style, init as color_init
import scipy

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
                                replace_params=None, n_boards_required=None,
                                time_scale_factor=None, first_n_layers=None,
                                timestep=1.0
                                ):
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

    # the number of populations to be reconstructed is either passed in
    # (first_n_layers) or the total number of available populations
    no_pops = first_n_layers or len(json_data['populations'].keys())
    no_proj = len(json_data['projections'].keys())
    # setup
    setup_params = json_data['setup']
    # TODO move setup outside into whatever experiment is run

    sim.setup(timestep,
              timestep,
              timestep,
              n_boards_required=n_boards_required,
              time_scale_factor=time_scale_factor)
    extra_params = {}
    try:
        extra_params['json_custom_params'] = json_data['custom_params']
    except:
        pass
    # set up populations
    # Add reports here
    total_no_neurons = 0
    total_no_synapses = 0
    max_synapses_per_neuron = 0
    no_afferents = {}
    all_neurons = {}
    all_connections = {}
    print("Population reconstruction begins...")
    for pop_no in range(no_pops):
        pop_info = json_data['populations'][str(pop_no)]
        p_id = pop_info['id']
        pop_cellclass = pydoc.locate(pop_info['cellclass'])
        print("Reconstructing pop {:35}".format(pop_info['label']), "containing",
              format(pop_info['n_neurons'], ","), "neurons")
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

        no_afferents[pop_info['label']] = 0
        all_neurons[pop_info['label']] = pop_info['n_neurons']
        total_no_neurons += pop_info['n_neurons']
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
        receptor_type = DEFAULT_RECEPTOR_TYPES[proj_info['receptor_type']]
        _type = "_exc" if receptor_type == "excitatory" else "_inh"
        conn_label = proj_info['pre_label'] + "_to_" + proj_info['post_label'] + _type
        if proj_info['post_label'] not in all_neurons.keys():
            print("Aborting the creation of proj", conn_label)
            continue

        # id of projection used to retrieve from list connectivity
        _conn = utils._prune_connector(connectivity_data[str(proj_info['id'])],
                                       prune_level=prune_level)

        # build synapse dynamics
        synapse_dynamics = utils._build_synapse_info(sim, proj_info)
        total_no_synapses += _conn.shape[0]

        post_n_neurons = \
            json_data['populations'][str(proj_info['post_number'])]['n_neurons']

        number_of_synapses = _conn.shape[0]
        max_synapses_per_neuron = max(max_synapses_per_neuron,
                                      number_of_synapses / post_n_neurons)
        # create the projection
        print("Reconstructing proj", conn_label)
        _c = Fore.GREEN if receptor_type == "excitatory" else Fore.RED
        print("\t{:20}".format(format(number_of_synapses, ",")),
              _c, DEFAULT_RECEPTOR_TYPES[proj_info['receptor_type']],
              Style.RESET_ALL,
              "synapses")
        # Storing projection info based on name of post-synaptic population
        no_afferents[proj_info['post_label']] += number_of_synapses
        all_connections[conn_label] = _conn
        if len(_conn) > 0:
            projections.append(
                sim.Projection(
                    populations[proj_info['pre_number']],  # pre population
                    populations[proj_info['post_number']],  # post population
                    pydoc.locate(proj_info['connector_type'])(_conn),  # connector
                    synapse_type=synapse_dynamics,
                    source=proj_info['source'],
                    receptor_type=DEFAULT_RECEPTOR_TYPES[proj_info['receptor_type']],
                    space=proj_info['space'],
                    label=conn_label
                )
            )

    connectivity_data.close()
    print("Reconstruction complete!")

    print("=" * 80)
    print("Reports")
    print("-" * 80)
    write_report("Total number of neurons", total_no_neurons)
    write_report("Total number of synapses", total_no_synapses)
    if total_no_synapses > 0:
        write_report("Avg fan in", total_no_synapses / total_no_neurons)
    else:
        write_report("Avg fan in", "NaN")
    print("-" * 80)
    print("Number of afferents (exc + inh)")
    for k in no_afferents.keys():
        print("Total afferents for {:35} : {:25}".format(
            k,
            format(int(no_afferents[k]), ",")))
        print("\tthat is {:15.2f} synapses / neuron".format(
            no_afferents[k] / all_neurons[k]))
    print("=" * 80)

    extra_params['all_neurons'] = all_neurons
    extra_params['all_connections'] = all_connections
    extra_params['json_dict'] = json_data

    return populations, projections, extra_params


def write_report(msg, value):
    print("{:<50}:{:>14}".format(msg, format(int(value), ",")))


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

def extract_parameters(filename, output_dir):
    """Takes parameters from the serialised model and outputs a more human-readable directory structure.
    """
    import os
    
    #Grab the files
    
    # Objects and parameters
    projections = []
    populations = []

    no_neurons = {}
    total_no_neurons = 0
    
    # Load the data from disk
    with open(filename + ".json", "r") as read_file:
        json_data = json.load(read_file)
    # Load connectivity data from disk
    connectivity_data = np.load(filename + ".npz", allow_pickle=True)
    # the number of populations to be reconstructed is either passed in
    # (first_n_layers) or the total number of available populations
    no_pops = len(json_data['populations'].keys())
    no_proj = len(json_data['projections'].keys())

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    os.chdir(output_dir)

    #Loop over layers
    for pop_no in range(no_pops):
        pop_info = json_data['populations'][str(pop_no)]
        p_id = pop_info['id']
        pop_cellclass = pydoc.locate(pop_info['cellclass'])
        folderName = output_dir + '/' + pop_info['label']
        no_neurons[pop_info['label']] = pop_info['n_neurons']
        total_no_neurons += pop_info['n_neurons']
        
        if not os.path.exists(folderName):
            os.mkdir(folderName)
        os.chdir(folderName)
        
        print("Found pop {:35}".format(pop_info['label']), "containing",
              format(pop_info['n_neurons'], ","), "neurons")
        #Generate txt file that defines population (number of neurons, neuron model etc.)
        f= open("Population_description.txt","w+")
        for key in pop_info.keys():
            f.write(str(key) + " : " + str(pop_info[key]) +'\n')
        
        #Make a directory for inhibitory and excitatory projections
        if not os.path.exists("exc_projections"):
            os.mkdir("exc_projections")
    
    
        if not os.path.exists("inh_projections"):
            os.mkdir("inh_projections")
    
    os.chdir(output_dir)
    
    for proj_no in range(no_proj):
        # temporary utility variable
        proj_info = json_data['projections'][str(proj_no)]
        receptor_type = DEFAULT_RECEPTOR_TYPES[proj_info['receptor_type']]
        _type = "_exc" if receptor_type == "excitatory" else "_inh"
        conn_label = proj_info['pre_label'] + "_to_" + proj_info['post_label'] + _type
        if proj_info['post_label'] not in no_neurons.keys():
            print("Aborting the creation of proj", conn_label)
            continue
        print("Outputing {}".format(conn_label)) 
        #Go to the correct directory
        os.chdir(proj_info['post_label'])
        if _type == "_exc":
            os.chdir("exc_projections")
        elif _type == "_inh":
            os.chdir("inh_projections")
        else:
            print("How did you manage to get a receptor type that isn't exc or inh?")
        
        #Convert from_list to matrix
        weight_matrix = convert_from_list_to_matrix(connectivity_data[str(proj_info['id'])])
        #Write a csv
        scipy.sparse.save_npz("connections" + _type, weight_matrix)
        #Leave
        os.chdir(output_dir)

    connectivity_data.close()
    #Put connectivity .npz in these directories

    #Is there some way to spot convolution? Probably not
    #Actually, could you use something akin to dictionary coding:
    #For every pre weight assign a letter to a (relative_pre_neuron_index, weight) pair
    #This should be repeated
    #Sounds like a lot of work

def convert_from_list_to_matrix(from_list):
    """Converts a from_list connector to an ANN-like connectivity matrix.
    """
    from scipy import sparse
    #from_list (pre_index, post_index, weight, delay)
    from_list = np.array(from_list)
    mat_coo = sparse.coo_matrix((from_list[:,2], (from_list[:,0].astype(int), from_list[:,1].astype(int))))
    return mat_coo

def main():
    extract_parameters("/mnt/snntoolbox/pynn_object_serialisation/pynn_object_serialisation/experiments/isotope_testing/isotope_model_dense_normalised_input_production_serialised",\
        "/mnt/snntoolbox/pynn_object_serialisation/pynn_object_serialisation/experiments/isotope_testing/test_out_dir")

if __name__ == "__main__":
   main()

