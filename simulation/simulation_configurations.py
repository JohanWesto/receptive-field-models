#!/usr/bin/python
"""
" @section DESCRIPTION
" Configurations for generating stimuli and simulating neural networks
"""

import numpy as np


# Stimulus types
def get_stimulus_params(name='rc',
                        amplitude=1,
                        dimensions=[1, 1],
                        rho=0.5,
                        octave_range=4,
                        n_sweeps=2,
                        block_length=10):
    """
    
    :param name: 'rc'|'drc'|'dmr'|'rn'|'gaussian'|'fm'|'mn'
    :param amplitude: 
    :param dimensions: input dimensions
    :param rho: density parameter for 'rc' and 'drc' stimuli
    :param octave_range: parameter for 'dmr', 'rn', and 'fm' stimuli
    :param n_sweeps: 'fm' parameter
    :param block_length: 'fm' parameter
    :return params: 
    """

    params = {'name': name,
              'params': {'amplitude': amplitude,
                         'dimensions': dimensions,
                         'rho': rho,
                         'octave_range': octave_range,
                         'n_sweeps': n_sweeps,
                         'block_length': block_length
                         }
              }

    return params

nest_type1 = {
    'name': 'nest_type1',
    'type': 'two_layer',
    'simulator': 'nest',
    'params': {'syn_weight': 1400.0,
               'input_filter': np.array([1.0]).reshape(1, 1),
               'output_filter': [0.1, 0.4, 0.7, 1.0, 0.7, 0.4, 0.1],
               'output_rec_filter': [],
               'synapse_type': 'tsodyks',
               'feed_forward_inh': False}
}

nest_type3 = {
    'name': 'nest_type3',
    'type': 'two_layer',
    'simulator': 'nest',
    'params': {'syn_weight': 30.0,
               'input_filter': [1.0, 1.0, 1.0],
               'output_filter': [0.1, 0.4, 0.7, 1.0, 0.7, 0.4, 0.1],
               'output_rec_filter': [],
               'synapse_type': 'static',
               'feed_forward_inh': False}
}

# NEST network types
# Used for generating the network in Westo and May (2017)
w_surround = np.array([-0.15, -0.5, 0.15, 1.0, 0.15, -0.5, -0.15])
w_surround = w_surround.reshape(w_surround.size, 1)
w_surround = np.hstack([0.5*w_surround, w_surround, w_surround, 0.5*w_surround])

nest_pool_net = {
    'name': 'nest_pool',
    'type': 'two_layer',
    'simulator': 'nest',
    'params': {'syn_weight': 600.0,
               'input_filter': w_surround,
               'output_filter': [0.25, 0.75,
                                 1.0, 1.0, 1.0, 1.0, 1.0,
                                 0.75, 0.25],
               'output_rec_filter': [],
               'synapse_type': 'static',
               'feed_forward_inh': False}
}

nest_pool_stp_net = {
    'name': 'nest_pool_stp',
    'type': 'two_layer',
    'simulator': 'nest',
    'params': {'syn_weight': 1300.0,
               'input_filter': w_surround,
               'output_filter': [0.25, 0.75,
                                 1.0, 1.0, 1.0, 1.0, 1.0,
                                 0.75, 0.25],
               'output_rec_filter': [],
               'synapse_type': 'tsodyks',
               'feed_forward_inh': False}
}

nest_pool_rec_stp_net = {
    'name': 'nest_pool_rec_stp',
    'type': 'two_layer',
    'simulator': 'nest',
    'params': {'syn_weight': 1200.0,
               'input_filter': w_surround,
               'output_filter': [1.0, 1.0, 1.0, 1.0, 1.0],
               'output_rec_filter': [1.0, 1.0, 0.0, 1.0, 1.0],
               'synapse_type': 'tsodyks',
               'feed_forward_inh': False}
}


# Rate based simulator model neurons
sim_gabor_filter = {
    'name': 'gabor_filter',
    'type': 'gabor',
    'simulator': 'rate',
    'cf_act_fun': None
}

sim_gabor_sq_filter = {
    'name': 'gabor_sq_filter',
    'type': 'gabor',
    'simulator': 'rate',
    'cf_act_fun': 'square'
}

sim_energy_model = {
    'name': 'energy_model',
    'type': 'energy',
    'simulator': 'rate',
    'cf_act_fun': 'square'
}

sim_energy_model = {
    'name': 'energy_model',
    'type': 'energy',
    'simulator': 'rate',
    'cf_act_fun': 'square'
}

sim_shifted_filters = {
    'name': 'shifted_filters',
    'type': 'shifted',
    'simulator': 'rate',
    'cf_act_fun': 'rectify'
}

sim_shifted_sq_filters = {
    'name': 'shifted_sq_filters',
    'type': 'shifted',
    'simulator': 'rate',
    'cf_act_fun': 'square'
}

sim_context_model = {
    'name': 'context_model',
    'type': 'context',
    'simulator': 'rate',
    'cf_act_fun': None
}
