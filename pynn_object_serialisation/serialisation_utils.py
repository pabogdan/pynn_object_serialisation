import pydoc
from spynnaker8.models.synapse_dynamics import SynapseDynamicsStatic
from spynnaker8 import SpikeSourceArray, SpikeSourcePoisson
from spynnaker8.extra_models import SpikeSourcePoissonVariable
import numpy as np


def _type_string_manipulation(class_string):
    return class_string.split("'")[1]


def _trundle_through_synapse_information(syn_info, dict_to_augment):
    dict_to_augment['weight'] = syn_info.weight
    dict_to_augment['delay'] = syn_info.delay
    # TODO continue for other types of synapse dynamics


def _build_synapse_info(sim, construct):
    dyn_type = construct['synapse_dynamics']
    syn_info_class = pydoc.locate(dyn_type)
    constructor_info = construct['synapse_dynamics_constructs']
    if syn_info_class is SynapseDynamicsStatic:
        syn_info = syn_info_class(weight=constructor_info['weight'],
                                  delay=constructor_info['delay'])

    else:
        raise NotImplementedError(
            "Synapse dynamics of type {} are not supported yet".format(
                syn_info_class))
    return syn_info


def _get_init_params_and_svars(cls):
    init = getattr(cls, "__init__")
    params = None
    if hasattr(init, "_parameters"):
        params = getattr(init, "_parameters")
    svars = None
    if hasattr(init, "_state_variables"):
        svars = getattr(init, "_state_variables")
    return init, params, svars


def _trundle_through_neuron_information(neuron_model, dict_to_augment=None):
    parameter_list = neuron_model._celltype.default_parameters.keys()
    retrieved_params = {}
    if (isinstance(neuron_model._celltype, SpikeSourceArray) or
            isinstance(neuron_model._celltype, SpikeSourcePoisson)):
        # model_components = [neuron_model._celltype]
        merged_dict = {'spike_times': neuron_model._vertex._spike_times}
    elif isinstance(neuron_model._celltype, SpikeSourcePoissonVariable):
        __cell_ref = neuron_model._vertex
        merged_dict = {
            'rates':__cell_ref._rates,
            'starts':__cell_ref._starts,
            'durations':__cell_ref._durations}
    else:
        # model_components = neuron_model._celltype._model._components
        # model_components = neuron_model._vertex._parameters
        # merge returned dicts
        merged_dict = neuron_model._vertex._parameters
        # for comp in model_components:
        #     merged_dict.update(comp.get_all_parameters())
    # check that parameters are not numpy arrays
    for p in merged_dict.keys():
        if isinstance(merged_dict[p], np.ndarray):
            merged_dict[p] = merged_dict[p].tolist()
        else:
            merged_dict[p] = list(merged_dict[p])


    for param in parameter_list:
        retrieved_params[param] = merged_dict[param]

    if dict_to_augment:
        dict_to_augment['cellparams'] = retrieved_params
    # TODO continue for other types of synapse dynamics
    return retrieved_params


def _prune_connector(conn, prune_level=1):
    if prune_level == 1:
        return conn
    descending_argsort = conn[:, 2].argsort()[::-1]
    cutoff_number = int(conn.shape[0] * prune_level)
    conn = conn[descending_argsort[:cutoff_number]]
    return conn
