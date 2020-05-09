import json
import pynn_object_serialisation.serialisation_utils as utils
from pynn_object_serialisation.functions import DEFAULT_RECEPTOR_TYPES
from pynn_object_serialisation.experiments.analysis_common import *
from sklearn.metrics import classification_report, confusion_matrix


def what_network_thinks(spikes, t_stim, simtime, num_classes):
    instaneous_counts = np.empty((num_classes, int(simtime / t_stim)))
    for index, value in np.ndenumerate(instaneous_counts):
        number_index, t_stim_index = index
        spikes_for_curr_nid = spikes[spikes[:, 0].astype(int) == number_index][:, 1] * ms
        instaneous_counts[number_index, t_stim_index] = np.count_nonzero(
            np.logical_and(
                spikes_for_curr_nid >= (t_stim_index * t_stim),
                spikes_for_curr_nid < ((t_stim_index + 1) * t_stim)
            )
        )

    argmax_result = np.empty(instaneous_counts.shape[1])
    for i in range(argmax_result.shape[0]):
        if np.all(instaneous_counts[:, i] == 0):
            argmax_result[i] = -1
        else:
            ir_max = np.max(instaneous_counts[:, i])
            argmax_result[i] = np.random.choice(np.flatnonzero(instaneous_counts[:, i] == ir_max))
            # argmax_result[i] = np.argmax(instaneous_counts[:, i])

    return argmax_result


def single_run_post_analysis(data):
    # Retrieve information from results file
    all_spikes = data['all_spikes'].ravel()[0]
    try:
        final_connectivity = data['final_connectivity'].ravel()[0]
    except:
        final_connectivity = []
        traceback.print_exc()
    try:
        init_connectivity = data['init_connectivity'].ravel()[0]
    except:
        init_connectivity = []
        traceback.print_exc()
    all_neurons = data['all_neurons'].ravel()[0]
    num_classes = data['num_classes']
    sim_params = data['sim_params'].ravel()[0]
    testing_examples = data['testing_examples']
    no_testing_examples = data['no_testing_examples']
    test_labels = data['y_test']
    extra_params = data['extra_params'].ravel()[0]
    t_stim = sim_params['argparser']['t_stim'] * ms
    timestep = sim_params['argparser']['timestep'] * ms
    simtime = data['simtime'] * ms
    # Pre-compute conversions
    time_to_bin_conversion = 1. / (timestep / ms)
    no_timesteps = int(simtime / ms * time_to_bin_conversion)

    # Report useful parameters
    print("=" * 80)
    print("Simulation parameters")
    print("-" * 80)
    pp(sim_params)
    # Report useful parameters
    print("=" * 80)
    print("Analysis report")
    print("-" * 80)
    print("Current time",
          plt.datetime.datetime.now().strftime("%H:%M:%S on %d.%m.%Y"))

    # Compute plot order
    plot_order = list(all_spikes.keys())
    # Report number of neurons
    print("=" * 80)
    print("Number of neurons in each population")
    print("-" * 80)
    for pop in plot_order:
        print("\t{:30} -> {:10} neurons".format(pop, all_neurons[pop]))

    # Report weights values
    print("Average weight per projection")
    print("-" * 80)
    conn_dict = {}
    for key in final_connectivity:
        # Connection holder annoyance here:
        conn = np.asarray(final_connectivity[key])
        init_conn = np.asarray(init_connectivity[key])
        if final_connectivity[key] is None or conn.size == 0:
            print("Skipping analysing connection", key)
            continue
        if len(conn.shape) == 1 or conn.shape[1] != 4:
            try:
                x = np.concatenate(conn)
                conn = x
            except:
                pass
            names = [('source', 'int_'),
                     ('target', 'int_'),
                     ('weight', 'float_'),
                     ('delay', 'float_')]
            useful_conn = np.zeros((conn.shape[0], 4), dtype=np.float)
            for i, (n, _) in enumerate(names):
                useful_conn[:, i] = conn[n].astype(np.float)
            final_connectivity[key] = useful_conn.astype(np.float)
            conn = useful_conn.astype(np.float)
        conn_dict[key] = conn
        original_conn = np.mean(init_conn[:, 2])
        orig_conn_sign = np.sign(original_conn)
        mean = np.mean(conn[:, 2]) * orig_conn_sign
        # replace with percentage of difference
        diff = original_conn - mean
        prop_diff = (diff / original_conn)
        # assert (0 <= proportion <= 1), proportion
        is_close = np.abs(prop_diff) <= .05
        _c = Fore.GREEN if is_close else Fore.RED

        print("{:45} -> {}{:4.6f}{} uS".format(
            key, _c, mean, Style.RESET_ALL),
            "c.f. {: 4.6f} uS ({:>7.2%})".format(
                original_conn, prop_diff))

    # Report delay values
    print("=" * 80)
    print("Average Delay per projection")
    print("-" * 80)
    for key in final_connectivity:
        conn = conn_dict[key]
        init_conn = init_connectivity[key]
        mean = np.mean(conn[:, 3])
        # replace with percentage of difference
        original_conn = np.mean(init_conn[:, 3])
        orig_conn_sign = np.sign(original_conn)
        if mean < original_conn:
            proportion = mean / original_conn
        else:
            proportion = original_conn / mean
        diff = original_conn - mean
        prop_diff = orig_conn_sign * (diff / original_conn)
        # assert (0 <= proportion <= 1), proportion
        is_close = proportion >= .95
        _c = Fore.GREEN if is_close else Fore.RED

        print("{:45} -> {}{:4.2f}{} ms".format(
            key, _c, mean, Style.RESET_ALL),
            "c.f. {: 4.2f} ms ({:>7.2%})".format(
                original_conn, prop_diff))

    # Compute accuracy
    output_nid_argmax = what_network_thinks(all_spikes[plot_order[-1]],
                                            t_stim, simtime,
                                            num_classes=num_classes)

    results = {
        'all_spikes': all_spikes,
        'true_labels': test_labels,
        'predicated_labels': output_nid_argmax,
        'num_classes': num_classes,
        'plot_order': plot_order
    }
    return results


def post_run_analysis(filename, fig_folder, dark_background=False):
    if dark_background:
        plt.style.use('dark_background')
    # Retrieve results file
    try:
        data = np.load(filename, allow_pickle=True)
    except FileNotFoundError:
        data = np.load(filename + ".npz", allow_pickle=True)
        filename += ".npz"

    # Check if the folders exist
    if not os.path.isdir(fig_folder) and not os.path.exists(fig_folder):
        os.mkdir(fig_folder)

    # Create figures folder for this results_file
    sim_fig_folder = os.path.join(fig_folder,
                                  str(ntpath.basename(filename))[:-4])
    if not os.path.isdir(sim_fig_folder) and not os.path.exists(sim_fig_folder):
        os.mkdir(sim_fig_folder)
    # Set up colours
    color_init(strip=False)

    # Plotting results for ...
    print("=" * 80)
    print("Analysing results for", filename)
    print("-" * 80)

    # TODO Loop over all available result files
    results = single_run_post_analysis(data)

    # TODO Aggregate results (should be safe on SANDS)

    # Use the results
    all_spikes = results['all_spikes']
    test_labels = results['true_labels']
    output_nid_argmax = results['predicated_labels']
    num_classes = results['num_classes']
    plot_order = results['plot_order']
    n_plots = float(len(plot_order))

    conf_mat = confusion_matrix(test_labels, output_nid_argmax, labels=range(num_classes))
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)
    print(classification_report(test_labels, output_nid_argmax))

    print("=" * 80)
    print("Plotting figures...")
    print("-" * 80)

    # Plot confusion matrix
    fig_conn, ax1 = plt.subplots(1, 1, figsize=(9, 9), dpi=800)

    ff_conn_ax = ax1.matshow(conf_mat, vmin=0, vmax=1)

    ax1.set_title("Confusion matrix\n")
    ax1.set_xlabel("Predicted label")
    ax1.set_ylabel("True label")

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    cbar = plt.colorbar(ff_conn_ax, cax=cax)
    cbar.set_label("Percentage")

    plt.tight_layout()
    save_figure(plt, os.path.join(sim_fig_folder, "confusion_matrix"),
                extensions=['.png', '.pdf'])
    plt.close(fig_conn)

    # raster plot including ALL populations
    print("Plotting spiking raster plot for all populations")
    f, axes = plt.subplots(len(all_spikes.keys()), 1,
                           figsize=(14, 20), sharex=True, dpi=400)
    for index, pop in enumerate(plot_order):
        curr_ax = axes[index]
        # spike raster
        curr_spikes = all_spikes[pop]
        curr_filtered_spikes = curr_spikes[curr_spikes[:, 1] < 10000]
        _times = curr_filtered_spikes[:, 1]
        _ids = curr_filtered_spikes[:, 0]
        curr_ax.scatter(_times,
                        _ids,
                        color=viridis_cmap(index / (n_plots + 1)),
                        s=.5, rasterized=True)
        curr_ax.set_title(use_display_name(pop))
    plt.xlabel("Time (ms)")
    # plt.suptitle((use_display_name(simulator)+"\n")
    f.tight_layout()
    save_figure(plt, os.path.join(sim_fig_folder, "raster_plots"),
                extensions=['.png', '.pdf'])
    plt.close(f)


if __name__ == "__main__":
    from pynn_object_serialisation.experiments.analysis_argparser import *

    if analysis_args.input and len(analysis_args.input) > 0:
        for in_file in analysis_args.input:
            post_run_analysis(in_file, analysis_args.figures_dir,
                              dark_background=analysis_args.dark_background)
