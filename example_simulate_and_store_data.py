#!/usr/bin/python
"""
" @section DESCRIPTION
" Script that simulates data and stores the resulting x-y data
"""

import os
import cPickle as pickle
from simulation import rate_based_neuron_simulator, stimulus_generator
from simulation.simulation_configurations import *

# Target directory
save_dir = "data/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# 1. Initialization
# 1.1. Stimulus parameters
"""
name:       'rc' random chords              | 'drc' dynamic random chords   |
            'dmr' dynamic moving ripples    | 'rn' ripple noise             |
            'gaussian' white noise
dimensions: stimulus dimensions [spatial x, spatial y] or [frequency, _]
amplitude:  stimulus amplitude
rho:        density parameter for 'rc' and 'drc' stimuli
"""
input_params = get_stimulus_params(name='rc',
                                   dimensions=[11, 1],
                                   amplitude=1.0,
                                   rho=0.5)
# 1.2. Network parameters
"""
sim_time:       total simulation time in ms
dt:             time resolution in ms
stimuli_params: stimulus parameters (defined above)
network_model:  defined in 'simulation_configurations.py'
                sim_gabor_filter: single Gabor filter
                sim_gabor_sq_filter: single squared Gabor filter
                sim_energy_model: two filter energy model
                sim_shifted_filters: five spatially shifted filters
                sim_shifted_sq_filters: five spatially shifted squared filters
                sim_context_model: context model replica
"""
sim_params = {
    'sim_time': 5e5,
    'dt': 10,
    'stimuli_params': input_params,
    'network_model': sim_energy_model,
}

# 2. Generate stimuli
print "\nGenerating stimuli..."
stimuli = stimulus_generator.create_step_current_sequence(sim_params)
print "Done"


# 3. Simulate a network to get spike counts
net = rate_based_neuron_simulator.RateBasedModel()
net.build_model(stimuli, sim_params)
net.view_filters()
net.simulate()
spike_count = net.get_spike_counts()


# 4. Gather data and save
data = {'x': stimuli['values'],
        'x_labels': ['Time (ms)', 'Input dims.'],
        'x_ticks': [[], []],
        'y': spike_count,
        'name': sim_params['network_model']['name'],
        'origin': 'simulation',
        'params': sim_params}
file_name = save_dir + sim_params['network_model']['name'] + \
            '_' + '%1.1e' % (sim_params['sim_time']) + '.dat'

# Use pickle to store the data
pickle.dump(data, open(file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
print "Data saved to: {}".format(file_name)