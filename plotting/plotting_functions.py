#!/usr/bin/python
"""
" @section DESCRIPTION
" Plotting functions for visualizing RF models
"""

import plot_configuration as plt_conf
import numpy as np
import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

plt.rc('xtick', direction='in')
plt.rc('ytick', direction='in')


def plot_parameters(params, fig=None):

    if fig is None:
        scaling = {'width': 2,
                   'height': 1
                   }
        fig = create_fig_window(scaling=scaling)
    else:
        fig.clear()

    n_plots = 0
    n_rf_parts = len(params.rfs[0].parts) if params.rfs[0].multilin else 1
    n_plots += len(params.rfs)*n_rf_parts
    if params.cfs:
        n_cf_parts = len(params.cfs[0].parts) if params.cfs[0].multilin else 1
        n_plots += len(params.cfs) * n_cf_parts
    if hasattr(params, 'cf_act_funs'):
        n_plots += len(params.cf_act_funs)

    n_cols = min([5, n_plots])
    n_rows = np.ceil(n_plots / np.float64(n_cols))
    ax_id = 1

    for rf in params.rfs:
        if rf.multilin:
            for part in rf.parts:
                ax = fig.add_subplot(n_rows, n_cols, ax_id)
                plot_field(ax, part)
                ax_id += 1
                ax.set_title('Part, (min: %2.2f, max: %2.2f, bias: %2.2f)' %
                             (part.min(), part.max(), rf.bias))
        else:
            rf_dims = len([i for i in rf.shape if i > 1])
            rf_proj = None if rf_dims < 3  else '3d'
            ax = fig.add_subplot(n_rows, n_cols, ax_id, projection=rf_proj)
            plot_field(ax, rf.field)
            ax.set_title('RF, (min: %2.2f, max: %2.2f, bias: %2.2f)' %
                         (rf.field.min(), rf.field.max(), rf.bias))
            ax_id += 1

    if hasattr(params, 'cf_act_funs'):
        for act_fun in params.cf_act_funs:
            ax = fig.add_subplot(n_rows, n_cols, ax_id)
            plot_line(ax, act_fun.base_peaks, act_fun.alpha, marker='o')
            ax.set_title('CF activation function')
            ax_id += 1

    for cf in params.cfs:
        if cf.multilin:
            for part in cf.parts:
                ax = fig.add_subplot(n_rows, n_cols, ax_id)
                plot_field(ax, part)
                ax_id += 1
                ax.set_title('Part, (min: %2.2f, max: %2.2f, bias: %2.2f)' %
                             (part.min(), part.max(), cf.bias))
        else:
            cf_dims = len([i for i in cf.shape if i > 1])
            cf_proj = None if cf_dims < 3  else '3d'
            ax = fig.add_subplot(n_rows, n_cols, ax_id, projection=cf_proj)
            plot_field(ax, cf.field)
            ax.set_title('CF, (min: %2.2f, max: %2.2f, bias: %2.2f)' %
                         (cf.field.min(), cf.field.max(), cf.bias))
            ax_id += 1

    tight()
    plt.pause(0.001)

    return fig


def plot_field(ax, field, dt=1, tick_labels=[], labels=[], aspect='auto'):

    # Truncate field dimensions
    org_shape = field.shape
    new_shape = [i for i in org_shape if i > 1]
    field = field.reshape(new_shape)
    axes_idxs = [i for i, x in enumerate(org_shape) if x > 1]

    # Get tick positions and labels
    axis_values = [np.arange(value) for value in field.shape]
    tick_pos, tick_pos_rev = get_tick_positions(axis_values)

    if len(field.shape) == 1:
        plot_line(ax, [], field, marker='o')
    elif len(field.shape) == 2:
        plot_image(ax, field, aspect=aspect)
    elif len(field.shape) == 3:
        plot_surface(ax, field)

    # if len(field.shape) >= 1:
    #     # ax.set_xticks(tick_pos[0])
    #     # Reverse if time axis
    #     # if axes_idxs[0] == 0:
    #     #     ax.set_xticklabels(np.int64((tick_pos_rev[0]) * -dt))
    #     # if len(labels): ax.set_xlabel(labels[axes_idxs[0]])
    # if len(field.shape) >= 2:
    #     # ax.set_yticks(tick_pos[1])
    #     # if len(tick_labels) >= 2: ax.set_yticklabels(tick_labels[1])
    #     # if len(labels): ax.set_ylabel(labels[axes_idxs[1]])
    # if len(field.shape) >= 3:
    #     # ax.set_zticks(tick_pos[2])
    #     # if len(tick_labels) >= 3: ax.set_zticklabels(tick_labels[2])
    #     # if len(labels): ax.set_zlabel(labels[axes_idxs[2]])


def plot_nonlinearity(ax, nonlinearity):

    if len(nonlinearity.z_edges) == 1:
        line_conf_dict = plt_conf.get_line_conf()
        x_diff = np.diff(nonlinearity.z_edges[0][1:3])[0] / 2
        x = nonlinearity.z_edges[0][1:] - x_diff
        x[-1] = x[-2] + 2 * x_diff
        y = nonlinearity.firing_prob.ravel()
        color = line_conf_dict['color'][0, :]
        plot_line(ax, x, y, color=color, label='P(spike|z)')
        if nonlinearity.firing_prob_nonlin is not None:
            y = nonlinearity.firing_prob_nonlin.ravel()
            color = line_conf_dict['color'][1, :]
            plot_line(ax, x, y, color=color, label='f(z)')
        ax.legend()
        ax.set_xlabel('$z$')
        ax.set_ylabel('P(spike|z)')

    if len(nonlinearity.z_edges) == 2:
        x_diff = np.diff(nonlinearity.z_edges[0][1:3])[0] / 2
        x = nonlinearity.z_edges[0][1:] - x_diff
        x[-1] = x[-2] + 2 * x_diff
        y_diff = np.diff(nonlinearity.z_edges[1][1:3])[0] / 2
        y = nonlinearity.z_edges[1][1:] - y_diff
        y[-1] = y[-2] + 2 * y_diff
        prob = nonlinearity.firing_prob
        plot_image(ax, prob, balanced=False, color=False)
        x_ticks = [int(tick) for tick in ax.get_xticks()
                   if 0 <= tick < prob.shape[0]]
        y_ticks = [int(tick) for tick in ax.get_yticks()
                   if 0 <= tick < prob.shape[1]]
        x_labels = ['{:1.1f}'.format(val) for val in x[x_ticks].tolist()]
        y_labels = ['{:1.1f}'.format(val) for val in y[y_ticks].tolist()]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('$z_1$')
        ax.set_ylabel('$z_2$')

    ax.set_title('Nonlinearity')


def plot_ln_significance(ax, significance):

    line_conf_dict = plt_conf.get_line_conf()
    color = line_conf_dict['color'][0, :]
    x = np.arange(1, significance.values.size + 1)
    y = significance.values
    plot_line(ax, x, y, color=color, line_style='',  label='model')

    ax.legend()
    ax.set_xlabel('Dimension')
    ax.set_ylabel(significance.unit_label)


def plot_mi_estimation(ax, evaluation, key='stc'):

    if key in evaluation.mi:

        # Naive MI values for different histogram resolutions first
        x = np.array(evaluation.mi_hist_resolutions)
        # if evaluation.mi_squared_res:
        #     x *= x
        y = evaluation.mi_values[key]
        plot_line(ax, x, y, color='b', line_style='', label='naive')
        # Bias estimates
        y = evaluation.mi_bias[key]
        plot_line(ax, x, y, color='r', line_style='', label='bias')
        # Corrected values "true"
        y = evaluation.mi_values[key] - evaluation.mi_bias[key]
        plot_line(ax, x, y, color='k', line_style='', label='true')
        # Mean value taken as our estimate
        y = evaluation.mi[key] * np.ones(x.size)
        plot_line(ax, x, y, color='k', line_style=':', marker='', label='mean')

        ax.legend()
        ax.set_xlabel('Resolution')
        ax.set_ylabel('MI')
        ax.set_title('MI_%s: %2.2f, N/R: %2.1f' %
                     (key, evaluation.mi[key], evaluation.mi_nr_frac))


def plot_field_gradients(ana_der, num_der):

    line_conf_dict = plt_conf.get_line_conf()

    n_parts = len(ana_der.parts) + 1 if ana_der.multilin else 2
    scaling = {'width': n_parts,
              'height': 1
              }

    # Create a figure window
    fig = create_fig_window(scaling)

    for part_idx in range(n_parts):
        ax = fig.add_subplot(1, n_parts, part_idx + 1)
        if part_idx < n_parts - 1:
            if ana_der.multilin:
                ana = ana_der.parts[part_idx]
                num = num_der.parts[part_idx]
                title = "Part: {}".format(part_idx)
            else:
                ana = ana_der.field.ravel()
                num = num_der.field.ravel()
                title = "Field".format(part_idx)
        else:
            ana = np.array([ana_der.bias])
            num = np.array([num_der.bias])
            title = "Bias"

        plot_line(ax, [], ana.ravel(),
                  line_style='-',
                  marker='s',
                  color=line_conf_dict['color'][0, :])
        plot_line(ax, [], num.ravel(),
                  line_style=':',
                  marker='d',
                  color=line_conf_dict['color'][1, :])
        max_abs_val = max(np.max(np.abs(ana)),
                          np.max(np.abs(num)))
        if max_abs_val != 0:
            y_lim_val = 2 ** (np.ceil(np.log2(max_abs_val)))
            ax.set_ylim([-y_lim_val, y_lim_val])
        ax.set_title(title)
        ax.legend(['Analytical', 'Numerical'], loc='best')

    plt.show()


def plot_act_fun_gradients(ana_der, num_der):

    line_conf_dict = plt_conf.get_line_conf()

    n_parts = 1
    scaling = {'width': n_parts,
              'height': 1
              }

    # Create a figure window
    fig = create_fig_window(scaling)
    ax = fig.add_subplot(1, 1, 1)
    plot_line(ax, [], ana_der.alpha,
              line_style='-',
              marker='s',
              color=line_conf_dict['color'][0, :])
    plot_line(ax, [], num_der.alpha,
              line_style=':',
              marker='d',
              color=line_conf_dict['color'][1, :])
    max_abs_val = max(np.max(np.abs(ana_der.alpha)),
                      np.max(np.abs(num_der.alpha)))
    if max_abs_val != 0:
        y_lim_val = 2 ** (np.ceil(np.log2(max_abs_val)))
        ax.set_ylim([-y_lim_val, y_lim_val])
    ax.set_title('Act. fun. alphas')
    ax.legend(['Analytical', 'Numerical'], loc='best')

    plt.show()


def create_fig_window(scaling={}, size=None):

    if len(scaling) == 0:
        scaling= {'width': 1,
                  'height': 1
                  }

    fig_conf_dict = plt_conf.get_fig_configuration()
    font_conf_dict = plt_conf.get_font_configuration()

    plt.rc('font', **font_conf_dict)
    plt.rc('axes', titlesize='medium')
    plt.rc('legend', fontsize='medium')

    if size is not None:
        fig_size = [val / 2.54 for val in size]
    else:
        fig_size = [fig_conf_dict['size']['width']*scaling['width'] / 2.54,
                    fig_conf_dict['size']['height']*scaling['height'] / 2.54]

    fig = plt.figure(figsize=fig_size, dpi=150)

    return fig


def create_axes(fig, n_axes, dim=2):

    ax = []

    for ax_idx in range(n_axes):
        ax.append(fig.add_subplot(1, n_axes, ax_idx+1))

    # if n_axes == 1:
    #     position = [0.15, 0.15, 0.7, 0.7]
    #     if dim ==2:
    #         ax.append(fig.add_axes(position))
    #     elif dim == 3:
    #         ax.append(fig.add_axes(position, projection='3d'))
    # elif n_axes == 2:
    #     position_1 = [0.1, 0.15, 0.35, 0.7]
    #     position_2 = [0.6, 0.15, 0.35, 0.7]
    #     if dim == 2:
    #         ax.append(fig.add_axes(position_1))
    #         ax.append(fig.add_axes(position_2))
    #     elif dim == 3:
    #         ax.append(fig.add_axes(position_1, projection='3d'))
    #         ax.append(fig.add_axes(position_2, projection='3d'))

    plt.tight_layout()
    return ax


def get_tick_positions(axis_values):

    tick_pos = []
    tick_pos_rev = []
    ax_conf_dict = plt_conf.get_ax_configuration()

    for value_id in range(len(axis_values)):
        values = axis_values[value_id]

        if isinstance(values, list):
            n = float(len(values))
        else:
            n = np.float64(values.size)
        inc = np.int64(np.ceil(n / ax_conf_dict['tick']['max_n_tick']))

        forward_pos = np.arange(int(n)-1, 0, -inc)[::-1]
        backward_pos = np.flipud(values)[forward_pos]
        tick_pos.append(forward_pos)
        tick_pos_rev.append(backward_pos)

    return tick_pos, tick_pos_rev


def annotate(fig, x, y, text):
    font_conf_dict = plt_conf.get_font_configuration()
    fig.text(x, y, text,
             weight='bold',
             family='sans-serif',
             size=font_conf_dict['size']+2)


def plot_line(ax, x, y, line_style='-', marker='.', color='k', label=''):

    line_conf_dict = plt_conf.get_line_conf()
    if len(x) == 0: x = np.arange(y.size)

    ax.plot(x, y,
            color=color,
            lw=line_conf_dict['width'],
            ls=line_style,
            marker=marker,
            ms=line_conf_dict['size'],
            label=label)


def plot_bars(ax, bars, conf=[], colors=[], labels=[]):

    bar_conf_dict = plt_conf.get_bar_conf()

    n_bars = len(bars)
    n_classes = len(bars[0])
    width = 1. / (n_bars + 1)

    if len(labels) == 0:
        labels = [i_tmp + 1 for i_tmp in range(n_bars)]

    # Plot bar diagram
    for bar_idx in range(len(bars)):
        if len(colors) == 0:
            color = bar_conf_dict['color'][bar_idx, :]
        else:
            color = colors[bar_idx]
        x_val = np.arange(len(bars[bar_idx])) + bar_idx * width
        ax.bar(x_val, bars[bar_idx], width, color=color, label=labels[bar_idx])

    # Add error bars if a confidence interval is provided
    if len(conf) > 0:
        for bar_idx in range(len(bars)):
            x_val = np.arange(len(bars[bar_idx])) + (bar_idx + 0.5) * width
            ax.errorbar(x_val, bars[bar_idx],
                        fmt='none',
                        ecolor='k',
                        yerr=conf[bar_idx],)

    ax.set_xticks([i_tmp + width for i_tmp in range(n_classes)])


def plot_image(ax, image, c_lim=[], balanced=True, offset=0, color=True, aspect=1, positive_only=False):

    im_conf_dict = plt_conf.get_im_conf()

    if color:
        cmap = im_conf_dict['cmap']
    else:
        cmap = 'gray'

    if positive_only:
        cmap = plt.cm.coolwarm(np.arange(plt.cm.coolwarm.N))
        cmap = cmap[128:, :]
        cmap = ListedColormap(cmap)

    if len(c_lim) == 0:
        if balanced:
            diff = max([abs(offset-image.min()), abs(image.max()-offset)])
            c_max = offset + diff
            c_min = offset - diff
        else:
            c_max = image.max()
            c_min = image.min()
    else:
        c_max = c_lim[1]
        c_min = c_lim[0]

    ax.imshow(image.T,
              interpolation=im_conf_dict['interpolation'],
              vmin=c_min, vmax=c_max,
              cmap=cmap,
              aspect=aspect)
    ax.invert_yaxis()


def plot_surface(ax, field):

    c_lim = 0.25

    slice_dim = [field.shape.index(min(field.shape))]
    plane_dims = list(set(range(3)) ^ set(slice_dim))

    field_max = np.max(np.abs([field.min(), field.max()]))
    field = field * c_lim/field_max
    X, Y = np.meshgrid(range(field.shape[plane_dims[0]]),
                       range(field.shape[plane_dims[1]]))

    for z_level in range(field.shape[slice_dim[0]]):
        if slice_dim[0] == 0:
            tmp = field[z_level, :, :].T
        elif slice_dim[0] == 1:
            tmp = field[:, z_level, :].T
        elif slice_dim[0] == 2:
            tmp = field[:, :, z_level].T

        # ax.contourf(X, Y, z_level+field[:, :, z_level].T,
        #             levels=levels,
        #             antialiased=False,
        #             vmin=z_level-c_lim, vmax=z_level+c_lim,
        #             cmap='coolwarm')
        ax.plot_surface(X, Y, z_level + tmp,
                        linewidth=0,
                        antialiased=True,
                        rstride=1, cstride=1,
                        vmin=z_level-c_lim, vmax=z_level+c_lim,
                        cmap='coolwarm')


def legend(ax, frame=True): ax.legend(loc='best', frameon=frame)


def tight(): plt.tight_layout()


def save(fig, file_name): fig.savefig(file_name)


def show(): plt.show()


def interactive(on=True): plt.ion() if on else plt.ioff()


def clear(fig): fig.clear()


def close(fig): plt.close(fig)
