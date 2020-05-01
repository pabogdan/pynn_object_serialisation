import json
import pynn_object_serialisation.serialisation_utils as utils
from pynn_object_serialisation.functions import DEFAULT_RECEPTOR_TYPES
from pynn_object_serialisation.experiments.mobilenet_testing.analysis_common import *


def network_statistics(filename, fig_folder, dark_background=False):
    if dark_background:
        plt.style.use('dark_background')
    print("=" * 80)
    print("LOOKING AT", filename)
    print("-" * 80)
    # Checking fig directory structure exists
    if not os.path.isdir(fig_folder) and not os.path.exists(fig_folder):
        os.mkdir(fig_folder)
    current_fig_folder = os.path.join(fig_folder, filename.split('/')[-1])
    # Make folder for current figures
    if not os.path.isdir(current_fig_folder) and not os.path.exists(current_fig_folder):
        os.mkdir(current_fig_folder)
    # Load the data from disk
    with open(filename + ".json", "r") as read_file:
        json_data = json.load(read_file)
    # Load the data from disk
    with open(filename + ".json", "r") as read_file:
        json_data = json.load(read_file)
    # Load connectivity data from disk
    connectivity_data = np.load(filename + ".npz", allow_pickle=True)

    # the number of populations to be reconstructed is either passed in
    # (first_n_layers) or the total number of available populations
    no_pops = len(json_data['populations'].keys())
    no_proj = len(json_data['projections'].keys())
    # Add reports here
    total_no_neurons = 0
    total_no_synapses = 0
    max_synapses_per_neuron = 0
    no_afferents = {}
    all_neurons = {}
    all_connections = {}
    conn_py_post = {}
    layer_order = []
    projection_order = []
    for pop_no in range(no_pops):
        pop_info = json_data['populations'][str(pop_no)]
        p_id = pop_info['id']
        label = pop_info['label']
        no_afferents[label] = 0
        layer_order.append(label)
        all_neurons[label] = pop_info['n_neurons']
        conn_py_post[label] = [[], []]
        total_no_neurons += pop_info['n_neurons']

    for proj_no in range(no_proj):
        # temporary utility variable
        proj_info = json_data['projections'][str(proj_no)]
        conn_label = proj_info['pre_label'] + "_to_" + proj_info['post_label']
        if proj_info['post_label'] not in all_neurons.keys():
            print("ROGUE CONNECTION!!!!!", conn_label)
            continue
        # id of projection used to retrieve from list connectivity
        _conn = utils._prune_connector(connectivity_data[str(proj_info['id'])],
                                       prune_level=1)
        total_no_synapses += _conn.shape[0]

        post_n_neurons = \
            json_data['populations'][str(proj_info['post_number'])]['n_neurons']

        number_of_synapses = _conn.shape[0]
        max_synapses_per_neuron = max(max_synapses_per_neuron,
                                      number_of_synapses / post_n_neurons)
        receptor_type = DEFAULT_RECEPTOR_TYPES[proj_info['receptor_type']]
        syn_type = "_exc" if receptor_type == "excitatory" else "_inh"
        all_connections[conn_label + syn_type] = _conn
        conn_py_post[proj_info['post_label']][proj_info['receptor_type']] = _conn
        projection_order.append(conn_label + syn_type)
        no_afferents[proj_info['post_label']] += number_of_synapses
    for k, v in no_afferents.items():
        if v == 0:
            del conn_py_post[k]
    print("=" * 80)
    print("Reports")
    print("-" * 80)
    write_report("Total number of neurons", total_no_neurons)
    write_report("Total number of synapses", total_no_synapses)
    write_report("Total number of layers", no_pops)
    if total_no_synapses > 0:
        write_report("Avg fan in", total_no_synapses / total_no_neurons)
    else:
        write_report("Avg fan in", "NaN")
    print("-" * 80)
    print("Number of afferents (exc + inh)")
    for k in layer_order:
        print("Total afferents for {:35} : {:25}".format(
            k,
            format(int(no_afferents[k]), ",")))
        print("\tthat is {:15.2f} synapses / neuron".format(
            no_afferents[k] / all_neurons[k]))
    # Write in LaTeX mode
    print("<LaTeX mode>")
    for k in layer_order:
        print("{:35} & {:25} & {:15.2f} synapses / neuron".format(
            k,
            format(int(no_afferents[k]), ","),
            no_afferents[k] / all_neurons[k]))
    print("</LaTeX mode>")
    # report mean and std for each connection
    for proj_k in projection_order:
        # Report range of source, target, weight, delay
        _conn = all_connections[proj_k]
        sources = _conn[:, 0]
        targets = _conn[:, 1]
        weights = _conn[:, 2]
        delays = _conn[:, 3]

        print(("Proj {:60} has "
               "source [{:8d}, {:8d}], "
               "target [{:8d}, {:8d}], "
               "weight [{:8.4f}, {:8.4f}], "
               "delay [{:4}, {:4}]").format(
            proj_k,
            int(sources.min()), int(sources.max()),
            int(targets.min()), int(targets.max()),
            weights.min(), weights.max(),
            delays.min(), delays.max()
        ))
        # Report mean, mode and std of weight
        print("\t weights stats: mean {:8.4f}, "
              "mode {:12.8f}, "
              "std {:8.4f}".format(
            np.mean(weights), scipy.stats.mode(weights).mode[0], np.std(weights)
        ))
    print("=" * 80)
    print("Plots")
    print("-" * 80)
    # TODO add plots
    plot_weight_barplot(all_neurons, conn_py_post,
                        layer_order=layer_order,
                        current_fig_folder=current_fig_folder)


if __name__ == "__main__":
    from pynn_object_serialisation.experiments. \
        mobilenet_testing. \
        analysis_argparser import *

    if analysis_args.input and len(analysis_args.input) > 0:
        for in_file in analysis_args.input:
            network_statistics(in_file, analysis_args.figures_dir,
                               dark_background=analysis_args.dark_background)
