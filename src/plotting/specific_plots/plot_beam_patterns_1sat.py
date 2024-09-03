
import gzip
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.config.config import (
    Config,
)
from src.config.config_plotting import (
    PlotConfig,
    save_figures,
    generic_styling,
)


def plot_beam_patterns(
        width,
        height,
        path,
        plots: list,
        color_dict: dict,
        line_style_dict: dict,
        label_dict: dict,
        marker_style_dict: dict,
        xlim: list,
        plots_parent_path,
        name,
) -> None:

    rows = max([subdict['row'] for subdict in plots]) + 1
    cols = max([subdict['column'] for subdict in plots]) + 1

    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        sharex='all',
        sharey='all',
        figsize=(width, height),
    )

    with gzip.open(path, 'rb') as file:
        data_file = pickle.load(file)
        data = data_file[1]
        angle_sweep_range = data_file[0]

    for plot in plots:
        row = plot['row']
        col = plot['column']
        realization = plot['realization']
        precoders = plot['precoders']
        print(f'Plot {row}, {col}')

        if rows > 1 and cols > 1:
            ax = axes[row, col]
        elif cols > 1:
            ax = axes[col]
        elif rows > 1:
            ax = axes[row]
        else:
            ax = axes

        # User positions
        for user_id, user_position in enumerate(data[realization]['user_positions'][0]):  # todo: fix this for multi sat
            if user_id == 0:
                label = 'User Positions'
            else:
                label = '_UserHidden'

            ax.scatter(
                user_position,
                -0,
                label=label,
                color='black',
                marker='o',
                # facecolor='white',
                zorder=10,
                s=80,
            )
            ax.axvline(
                user_position,
                label='_UserHidden',
                # color=mpl_colors.TABLEAU_COLORS[list(mpl_colors.TABLEAU_COLORS.keys())[user_idx]],
                color='darkgrey',
                linestyle='dashed',
            )

        # Beam patterns
        for precoder_id, precoder in enumerate(precoders):
            num_users = data[realization][precoder]['power_gains'].shape[0]
            num_satellites = data[realization][precoder]['power_gains'].shape[2]

            print(num_users)
            print(num_satellites)
            for user_id in range(data[realization][precoder]['power_gains'].shape[0]):
                if user_id == 0:
                    label = label_dict[precoder]
                else:
                    label = '_PrecoderHidden'

                peak = np.argmax(data[realization][precoder]['power_gains'][user_id])

                ax.plot(
                    angle_sweep_range,
                    data[realization][precoder]['power_gains'][user_id, :, 0],  # todo: do we just add all sats here for multisat?
                    label=label,
                    color=color_dict[precoder],
                    linestyle=line_style_dict[precoder],
                    marker=marker_style_dict[precoder],
                    markevery=[peak],
                    fillstyle='none',
                )

            print(f'{precoder} sum rate: {data[realization][precoder]["sum_rate"]}')

        ax.set_xlim(xlim)

        legend = ax.legend(
            loc='upper left',
        )
        legend.get_frame().set_linewidth(0.8)

        if row == (rows-1):
            ax.set_xlabel('Angle of Departure')
            # ax.set_xticks([])
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel('Power Gain')
            # ax.set_yticks([])
            ax.set_yticklabels([])
            # ax.tick_params(direction='in')

        generic_styling(ax=ax)

    fig.tight_layout(pad=0)

    save_figures(plots_parent_path=plots_parent_path, plot_name=name, padding=0)


def print_realizations(
        path,
) -> None:

    with gzip.open(path, 'rb') as file:
        data = pickle.load(file)[1]

    for date_entry_id, data_entry in enumerate(data):
        print(f'{date_entry_id}', '', end='')
        for satellite_id in range(len(data_entry['estimation_errors'])):
            print(f'Sat {satellite_id}', '', end='')
            for error in data_entry['estimation_errors'][satellite_id]:
                if any(data_entry['estimation_errors'][satellite_id][error]) != 0:
                    print(error, data_entry['estimation_errors'][satellite_id][error], '', end='')

        print('sum rate: ', end='')
        for key in data_entry:
            if key not in ['estimation_errors', 'user_positions']:
                print(key, f'{data_entry[key]["sum_rate"]:.2f},', '', end='')
        print('')


if __name__ == '__main__':

    cfg = Config()
    plot_cfg = PlotConfig()

    list_patterns = True
    realization = 16

    which_plots = [
        {
            'row': 0,
            'column': 0,
            'realization': realization,
            'precoders': ['mmse', 'slnr', 'learned_0.0_error']
        },
        # {
        #     'row': 1,
        #     'column': 0,
        #     'realization': realization,
        #     'precoders': ['mmse', 'learned_0.0_error']
        # },
        # {
        #     'row': 2,
        #     'column': 0,
        #     'realization': realization,
        #     'precoders': ['mmse', 'learned_0.05_error']
        # },
    ]

    data_path = Path(cfg.output_metrics_path,
                     '1sat_16ant_100k~0_3usr_10k~5k', 'beam_patterns', 'beam_patterns.gzip')

    plot_width = 0.99 * plot_cfg.textwidth
    # plot_width = 0.99 * 3.5
    plot_height = plot_width * 1/2

    x_limits = [1.45, 1.7]
    # x_limits = None

    colors = {
        'mmse': plot_cfg.cp3['black'],
        'slnr': plot_cfg.cp3['black'],
        'learned_0.0_error': plot_cfg.cp3['blue2'],
        'learned_0.05_error': plot_cfg.cp2['green'],
    }

    line_styles = {
        'mmse': 'solid',
        'slnr': 'dotted',
        'learned_0.0_error': 'solid',
        'learned_0.05_error': 'solid',
    }

    marker_styles = {
        'mmse': 'o',
        'slnr': 'x',
        'learned_0.0_error': 's',
        'learned_0.05_error': 'x',
    }

    labels = {
        'mmse': 'MMSE',
        'slnr': 'SLNR',
        'learned_0.0_error': 'SAC',
        'learned_0.05_error': 'abc',
    }

    if list_patterns:
        print_realizations(data_path)

    plot_beam_patterns(
        width=plot_width,
        height=plot_height,
        path=data_path,
        plots=which_plots,
        color_dict=colors,
        line_style_dict=line_styles,
        label_dict=labels,
        marker_style_dict=marker_styles,
        xlim=x_limits,
        plots_parent_path=plot_cfg.plots_parent_path,
        name='beam_pattern_1sat',
    )

    plt.show()
