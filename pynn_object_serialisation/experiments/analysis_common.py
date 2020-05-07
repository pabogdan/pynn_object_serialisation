import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as cm_mlib
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy
from matplotlib import animation, rc, colors
from brian2.units import *
import matplotlib as mlib
from scipy import stats
from pprint import pprint as pp
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import traceback
import os
import copy
import neo
from datetime import datetime
import warnings
import ntpath
from colorama import Fore, Style, init as color_init
import pandas as pd
import string
from matplotlib.ticker import MultipleLocator


mlib.use('Agg')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ensure we use viridis as the default cmap
plt.viridis()

# ensure we use the same rc parameters for all matplotlib outputs
mlib.rcParams.update({'font.size': 24})
mlib.rcParams.update({'errorbar.capsize': 5})
mlib.rcParams.update({'figure.autolayout': True})
viridis_cmap = mlib.cm.get_cmap('viridis')


def color_for_index(index, size, cmap=viridis_cmap):
    return cmap(index / (size + 1))


def write_sep():
    print("=" * 80)


def write_line():
    print("-" * 80)


def write_header(msg):
    write_sep()
    print(msg)
    write_line()


def write_short_msg(msg, value):
    print("{:40}:{:39}".format(msg, str(value)))


def write_value(msg, value):
    print("{:60}:{:19}".format(msg, str(value)))


def write_report(msg, value):
    print("{:<50}:{:>14}".format(msg, format(int(value), ",")))


COMMON_DISPLAY_NAMES = {

}


def capitalise(name):
    return string.capwords(
        " ".join(
            str.split(name, "_")
        ))


def use_display_name(name):
    name = name.lower()
    return COMMON_DISPLAY_NAMES[name] \
        if name in COMMON_DISPLAY_NAMES.keys() \
        else capitalise(name)


def save_figure(plt, name, extensions=(".png",), **kwargs):
    for ext in extensions:
        write_short_msg("Plotting", name + ext)
        plt.savefig(name + ext, **kwargs)


def strip_file_extension(name):
    split_by_dot = name.split(".")
    if split_by_dot[-1] in ["npz", "json", "h5"]:
        return ".".join(split_by_dot[:-1])
    else:
        return name


def plot_weight_barplot(all_neurons, conn_py_post, layer_order=None,
                        current_fig_folder="./"):
    """
    Plot histogram of e.g.
    """
    no_pops = len(all_neurons.keys())
    layer_order = layer_order[1:] or list(all_neurons.keys())
    fig, axes = plt.subplots(no_pops, 1, figsize=(8, 3 * no_pops), sharex=True)
    for (index, ax), curr_layer in zip(np.ndenumerate(axes), layer_order):
        i = index[0]
        all_weights_for_post = conn_py_post[curr_layer]

        for i2, curr_conn in enumerate(all_weights_for_post):
            # hist_weights = np.ones_like(current_conns) / float(N_layer)
            if len(curr_conn) != 0:
                _c = viridis_cmap(0) if i2 == 1 else viridis_cmap(0.99)
                ax.hist(curr_conn[:, 2], bins=20, color=_c,
                        edgecolor='k')

        ax.set_title(curr_layer)
        ax.set_ylabel("Count")
    plt.tight_layout()
    save_figure(plt, os.path.join(current_fig_folder, "weights_in_network"),
                extensions=[".png",  ".pdf"])
    plt.close(fig)

def plot_hist(all_neurons, variable_by_post, variable_name,  layer_order=None,
              current_fig_folder="./"):
    """
    Plot histogram of e.g.
    """
    layer_order = layer_order or list(all_neurons.keys())
    no_pops = len(layer_order)
    fig, axes = plt.subplots(no_pops, 1, figsize=(8, 3 * no_pops), sharex=True)
    for (index, ax), curr_layer in zip(np.ndenumerate(axes), layer_order):
        i = index[0]
        curr_values = variable_by_post[curr_layer]

        ax.hist(curr_values, bins=20, color=color_for_index(i, no_pops),
                edgecolor='k')

        ax.set_title(curr_layer)
        ax.set_ylabel("Count")
    plt.tight_layout()
    save_figure(plt, os.path.join(current_fig_folder, variable_name),
                extensions=[".png",  ".pdf"])
    plt.close(fig)


def plot_histogram():
    """
    Plot histogram of e.g.
    """
    pass
