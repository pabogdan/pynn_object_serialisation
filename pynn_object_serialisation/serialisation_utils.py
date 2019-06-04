import pydoc
from spynnaker8.models.synapse_dynamics import SynapseDynamicsStatic


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


def _trundle_through_neuron_information(neuron_model, dict_to_augment):
    parameter_list = neuron_model._celltype.default_parameters.keys()
    retrieved_params = {}
    model_components = neuron_model._celltype._model._components
    for param in parameter_list:
        for comp in model_components:
            try:
                retrieved_params[param] = getattr(comp, '_' + param)
            except:
                pass
    dict_to_augment['cellparams'] = retrieved_params
    # TODO continue for other types of synapse dynamics
