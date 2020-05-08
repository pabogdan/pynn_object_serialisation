import json
import pynn_object_serialisation.serialisation_utils as utils
from pynn_object_serialisation.functions import DEFAULT_RECEPTOR_TYPES
from pynn_object_serialisation.experiments.analysis_common import *


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
    print("Plotting results for", filename)
    print("-" * 80)
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
    sim_params = data['sim_params'].ravel()[0]
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
    plot_order = all_spikes.keys()
    n_plots = float(len(plot_order))
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
        conn_exists = True
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



    print("=" * 80)
    print("Plotting figures...")
    print("-" * 80)

    # raster plot including ALL populations
    print("Plotting spiking raster plot for all populations")
    f, axes = plt.subplots(len(all_spikes.keys()), 1,
                           figsize=(14, 20), sharex=True, dpi=400)
    for index, pop in enumerate(plot_order):
        curr_ax = axes[index]
        # spike raster
        _times = all_spikes[pop][:, 1]
        _ids = all_spikes[pop][:, 0]
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
