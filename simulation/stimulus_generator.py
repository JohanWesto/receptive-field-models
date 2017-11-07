#!/usr/bin/python
"""
" @section DESCRIPTION
" Functions for generating various stimuli
"""

import numpy as np
from scipy.special import erf
from scipy.integrate import cumtrapz


def create_step_current_sequence(params):
    """ Generates a step current sequence

    Args:
        sim_time: total simulation time (ms)
        dt: time bin length (ms)
        input_dim: input dimensionality
        input_type: stimulus type
        input_amp: stimulus amplitude
        rho: density parameter for the RC and the DRC stimuli
    Returns:
        step_current: generated current values and time points
    Raises:
        Exception if no the requested model is not implemented
    """

    sim_time = params['sim_time']
    dt = params['dt']
    input_params = params['stimuli_params']

    # Generate a time vector starting from 1 up to time_tot
    # in increments of time_bin_resolution
    time = np.arange(1.0, np.int(sim_time), dt)

    # Generate values for each time point based on the selected model
    # Uniform
    if input_params['name'] == 'uniform':
        values = _uniform_distribution(time, input_params['params'])
    # Gaussian
    elif input_params['name'] == 'gaussian':
        values = _gaussian_distribution(time, input_params['params'])
    # Random chords ('RC')
    elif input_params['name'] == 'rc':
        values = _rc_sequence(time, input_params['params'])
    # Dynamic random chords ('DRC')
    elif input_params['name'] == 'drc':
        values = _drc_sequence(time, input_params['params'])
    # Dynamic moving ripples ('DMR')
    elif input_params['name'] == 'dmr':
        values = _dynamic_moving_ripple(time, input_params['params'])
     # Ripple noise ('RN')
    elif input_params['name'] == 'rn':
        values = _ripple_noise(time, input_params['params'])
    # FM tones ('FM')
    elif input_params['name'] == 'fm':
        values = _fm_sweeps(time, input_params['params'])
    # Modulated noise ('MN')
    elif input_params['name'] == 'mn':
        values = _modulated_noise(time, input_params['params'])
    else:
        raise Exception("No input named: {0}".format(input_params['name']))

    step_current = {'name': 'step_current_generator',
                    'time': time,
                    'values': values}
    return step_current


def _uniform_distribution(time, params):
    """ stimuli generated from a uniform distribution

    :param time:
    :param params:
    :return values:
    """

    # Parameters
    min_val = 0.0
    max_val = params['amplitude']
    range = max_val - min_val
    stimulus_dims = params['dimensions']
    n_stimulus_elements = reduce(lambda i, j: i * j, stimulus_dims)

    # Values
    values = np.random.rand(time.size * n_stimulus_elements) * range + min_val
    values = values.reshape([len(time)] + stimulus_dims)

    return values


def _gaussian_distribution(time, params):
    """ stimuli generated from a gaussian distribution

    :param time:
    :param params:
    :return values:
    """
    stimulus_dims = params['dimensions']
    n_stimulus_elements = reduce(lambda i, j: i * j, stimulus_dims)
    mean = params['amplitude']
    std = mean / 4

    values = mean + np.random.randn(len(time) * n_stimulus_elements) * std
    values = values.reshape([len(time)] + stimulus_dims)

    return values


def _rc_sequence(time, params):
    """ Random chord sequence

    :param time:
    :param params:
    :return values:
    """
    stimulus_dims = params['dimensions']
    n_stimulus_elements = reduce(lambda i, j: i * j, stimulus_dims)
    amplitude = params['amplitude']
    rho = params['rho']

    rand = np.random.rand(time.size * n_stimulus_elements)
    values = amplitude * np.ones(time.size * n_stimulus_elements)
    values[rand > rho] = 0
    values = values.reshape([len(time)] + stimulus_dims)

    return values


def _drc_sequence(time, params):
    """ Dynamic random chord sequence

    :param time:
    :param params:
    :return values:
    """
    stimulus_dims = params['dimensions']
    n_stimulus_elements = reduce(lambda i, j: i * j, stimulus_dims)
    amplitude = params['amplitude']
    rho = params['rho']

    min_val = amplitude / 2
    max_val = amplitude
    range = max_val - min_val

    values = np.random.rand(time.size * n_stimulus_elements) * range + min_val
    rand = np.random.rand(time.size * n_stimulus_elements)
    values[rand > rho] = 0
    values = values.reshape([len(time)] + stimulus_dims)

    return values


def _dynamic_moving_ripple(time, params):
    """Generates a dynamic random ripple stimuli

    See. Escabi & Schreiner (2002)

    Args:
        time: Time vector
        n_freq: Number of frequencies
        octave_range: Octave spacing between min and max frequency
    Returns:
        s_lin: matrix containing the DMR stimuli
    Raises:

    """

    # Only used for estimating spectro-temporal receptive fields
    assert params['dimensions'][1] == 1

    n_freq = params['dimensions'][0]
    octave_range = params['octave_range']
    amplitude = params['amplitude']

    # Parameters
    m = 30
    f_lim = [-350, 350]
    f_rate = 3
    sigma_lim = [0 , 4]
    sigma_rate = 6

    x = np.linspace(0, octave_range, n_freq)
    t_tot = time.max() / 1e3
    n_f = np.int64(f_rate*t_tot)
    n_sigma = np.int64(sigma_rate*t_tot)

    # F
    t_tmp = np.linspace(0, t_tot, n_f+1)
    f_tmp = np.random.randn(n_f+1)
    # fun = interp1d(t_tmp, f_tmp, 'quadratic')
    # f = fun(time/1e3)
    f = np.interp(time/1e3, t_tmp, f_tmp)
    # Sigma
    t_tmp = np.linspace(0, t_tot, n_sigma+1)
    sigma_tmp = np.random.randn(n_sigma+1)
    # fun = interp1d(t_tmp, sigma_tmp, 'quadratic')
    # sigma = fun(time/1e3)
    sigma = np.interp(time/1e3, t_tmp, sigma_tmp)

    # Rescaling
    # First to the range [-1, 1]
    f = erf(f)
    sigma = erf(sigma)
    # Then to provided range
    f += 1
    f *= (f_lim[1]-f_lim[0])/2
    f += f_lim[0]
    sigma += 1
    sigma *= (sigma_lim[1]-sigma_lim[0])/2
    sigma += sigma_lim[0]

    # Finalizing
    f_int = cumtrapz(f, time/1e3)
    f_int = np.insert(f_int, 0, 0)
    s_dmr = m/2 * np.sin(2*np.pi*np.outer(x, sigma) +
                         np.outer(np.ones(n_freq), f_int))

    # s_lin = 10**((s_dmr-m/2)/20)
    s_lin = (s_dmr + m/2) / m
    s_lin *= amplitude

    return s_lin.T


def _ripple_noise(time, params):
    """Generates a radnom ripple stimuli

    See. Escabi & Schreiner (2002)

    Args:
        time: Time vector
        n_freq: Number of frequencies
        octave_range: Octave spacing between min and max frequency
    Returns:
        s_lin: matrix containing the DMR stimuli
    Raises:

    """

    # Only used for estimating spectro-temporal receptive fields
    assert params['dimensions'][1] == 1

    amplitude = params['amplitude']

    # Parameters
    n_dmrs = 16

    s_rn = _dynamic_moving_ripple(time, params)
    for i in range(n_dmrs-1):
        s_rn += _dynamic_moving_ripple(time, params)

    s_rn /= n_dmrs
    s_mean = s_rn.mean()
    s_std = s_rn.std()
    s_rn = 0.5*erf((s_rn-s_mean)/s_std) + 0.5

    scaling = amplitude / s_rn.max()
    s_rn *= scaling

    return s_rn


def _fm_sweeps(time, params):
    """Generates block-design FM tones

    See. Meyer et.a. (2014)

    Args:
        time: Time vector
        n_freq: Number of frequencies
        n_sweeps: Number of frequency sweeps in each block
        block_length: block_length in time bins
    Returns:
        fm_tones: matrix containing the block-design
    Raises:

    """

    # Only used for estimating spectro-temporal receptive fields
    assert params['dimensions'][1] == 1

    n_freq = params['dimensions'][0]
    n_sweeps = params['octave_range']
    block_length = params['block_length']
    amplitude = params['amplitude']

    values = np.zeros([time.size, n_freq])
    n_blocks = int(np.ceil(time.size / block_length))

    for sweep in range(n_sweeps):
        # freq = [np.linspace(2**(8+np.random.rand()*6),
        #                     2**(8+np.random.rand()*6),
        #                     block_length) for i in range(n_blocks)]
        # freq = np.hstack(freq)
        # freq = np.log2(freq) - 8
        # freq = freq / 6 * (n_freq-1)

        freq = [np.linspace(np.random.rand(),
                            np.random.rand(),
                            block_length) for i in range(n_blocks)]
        freq = np.hstack(freq)
        freq = freq / freq.max() * (n_freq-1)

        freq = np.int64(np.round(freq))
        values[np.arange(time.size), freq[0:time.size]] = 1.0

    values *= amplitude

    return values


def _modulated_noise(time, params):
    """Generates a modulated noise stimuli

    See. Woolley et.a. (2005)

    Args:
        time: Time vector
        n_freq: Number of frequencies
    Returns:
        s_lin: matrix containing the MN stimuli
    Raises:

    """

    # Only used for estimating spectro-temporal receptive fields
    assert params['dimensions'][1] == 1

    n_freq = params['dimensions'][0]
    amplitude = params['amplitude']

    # Parameters
    n_ripples = 100
    m = 30  # taken from Escabi & Schreiner (2002)

    frequencies = np.arange(n_freq)
    s_mn = np.zeros([n_freq, time.size])

    for ripple in range(n_ripples):
        omega = (2*np.random.rand()-1) / 0.5
        theta = (2*np.random.rand()-1) / 0.5
        phase = 2*np.pi*np.random.rand()
        for freq in frequencies:
            s_mn[freq, :] += np.cos(theta*time + omega*freq + phase)

    s_mean = s_mn.mean()
    s_std = s_mn.std()
    s_mn = m/2*erf((s_mn-s_mean)/s_std)

    s_lin = 10**((s_mn-m/2)/20)
    # s_lin = (s_dmr + m/2) / m
    s_lin *= amplitude

    return s_lin.T
