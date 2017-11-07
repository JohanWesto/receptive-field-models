#!/usr/bin/python
"""
" @section DESCRIPTION
" General configuration for generated figure windows and plots
"""

import numpy as np

color_set = np.array([[232, 125, 125],
                      [209, 209,  97],
                      [ 77, 179,  77],
                      [ 77, 119, 128],
                      [204, 204, 255],
                      [247, 161, 247]]) / 255.

font_conf = {'family': 'sans-serif',  # serif / sans-serif
             'sans-serif': 'Arial',
             'weight': 'normal',  # normal / bold
             'size': 8}

fig_conf = {'size': {'height': 8.,  # figure window height in cm
                     'width': 16.  # figure window width in cm}
                     },
            }

ax_conf = {'tick': {'max_n_tick': 3}
           }

line_conf = {'width': 2.0,
             'marker': 'o',
             'size': 4,
             'color': color_set}

bar_conf = {'color': color_set}

im_conf = {'interpolation': 'nearest',
           'cmap': 'coolwarm'}


def get_font_configuration():
    return font_conf


def get_fig_configuration():
    return fig_conf


def get_ax_configuration():
    return ax_conf


def get_line_conf():
    return line_conf


def get_bar_conf():
    return bar_conf


def get_im_conf():
    return im_conf
