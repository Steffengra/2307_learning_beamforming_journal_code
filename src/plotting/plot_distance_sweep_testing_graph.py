
import gzip
import pickle

import matplotlib.pyplot as plt
import numpy as np
from pathlib import (
    Path,
)

from src.config.config import (
    Config,
)
from src.config.config_plotting import (
    PlotConfig,
    save_figures,
    generic_styling, change_lightness,
)


def plot_distance_sweep_testing_graph(
        paths,
        name,
        width,
        height,
        plots_parent_path,
        legend: list or None = None,
        colors: list or None = None,
        markerstyle: list or None = None,
        linestyles: list or None = None,
        metric: str = 'sumrate'
) -> None:

    def get_metric_key(data_dict):

        for key_id, key in enumerate(data_dict[1].keys()):
            if match_string in str(key):
                return key
        return ValueError('match string not found')


    fig, ax = plt.subplots(figsize=(width, height))

    if metric == 'sumrate':
        match_string = 'calc_sum_rate'
    elif metric == 'fairness':
        match_string = 'calc_jain_fairness'
    else:
        raise ValueError(f'unknown metric {metric}')

    data = []
    for path in paths:
        with gzip.open(path, 'rb') as file:
            data.append(pickle.load(file))
    for data_id, data_entry in enumerate(data):
        #first_entry = list(data_entry[1]['sum_rate'].keys())[0]
        metric_key = get_metric_key(data_entry)

        if markerstyle is not None:
            marker = markerstyle[data_id]
        else:
            marker = None

        if colors is not None:
            color = colors[data_id]
        else:
            color = None

        if linestyles is not None:
            linestyle = linestyles[data_id]
        else:
            linestyle = None

        # if len(data_entry[0][data_entry[1]['sum_rate']['mean'] > 0.999 * np.max(data_entry[1]['sum_rate']['mean'])]) < int(len(data_entry[0])/10):
        #     markevery = np.searchsorted(data_entry[0], data_entry[0][data_entry[1]['sum_rate']['mean'] > 0.999 * np.max(data_entry[1]['sum_rate']['mean'])])
        #     for value_id in reversed(range(1, len(markevery))):
        #         if markevery[value_id] / markevery[value_id-1] < 1.01:
        #             markevery = np.delete(markevery, value_id)
        # else:
        #     markevery = (int(len(data_entry[0])/10 / len(data)*data_id), int(len(data_entry[0])/10))

        ax.plot(data_entry[0],
                data_entry[1][metric_key]['mean'],
                color=color,
                marker=marker,
                markevery=7,
                linestyle=linestyle
                )

    ax.set_xlabel('User Distance $ D_{usr} $ [m]')

    if metric == 'sumrate':
        ax.set_ylabel('Rate $ R $ [bps/Hz]')
    elif metric == 'fairness':
        ax.set_ylabel('Fairness $F$')

    if legend:
        ax.legend(legend, ncols=3)

    generic_styling(ax=ax)
    fig.tight_layout(pad=0)

    save_figures(plots_parent_path=plots_parent_path, plot_name=name+'_'+metric, padding=0)


if __name__ == '__main__':

    cfg = Config()
    plot_cfg = PlotConfig()

    data_paths = [
        Path(cfg.output_metrics_path,
             'test', 'distance_sweep',
             'testing_rsma_user_sweep_alpha_0_0_50000.gzip'),
        Path(cfg.output_metrics_path,
             'test', 'distance_sweep',
             'testing_rsma_user_sweep_alpha_0_5_0_50000.gzip'),
        Path(cfg.output_metrics_path,
             'test', 'distance_sweep',
             'testing_rsma_user_sweep_alpha_0_75_0_50000.gzip'),
        Path(cfg.output_metrics_path,
             'test', 'distance_sweep',
             'testing_rsma_user_sweep_alpha_1_0_50000.gzip'),
        Path(cfg.output_metrics_path,
             'test', 'distance_sweep',
             'testing_learned_rsma_power_factor_usersweep_0_50000_without_error.gzip'),
        Path(cfg.output_metrics_path,
             'test', 'user_distance_sweep',
             'testing_rsma_genie_sweep_0.0_50000.0.gzip'),
    ]

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.66

    plot_legend = ['$ \\alpha =0 $', '$ \\alpha =0.5 $', '$ \\alpha =0.75 $', '$ \\alpha =1 $','RSMA power','RSMA power genie']
    plot_markerstyle = [ '', '', '', '','d','']
    plot_colors = [ change_lightness(plot_cfg.cp2['blue'], 0.5), plot_cfg.cp2['blue'], change_lightness(plot_cfg.cp2['green'], 0.5), plot_cfg.cp2['green'],plot_cfg.cp2['gold'],plot_cfg.cp2['magenta']]
    plot_linestyles = [ ':', ':', ':', ':','-','-']

    plot_distance_sweep_testing_graph(
        paths=data_paths,
        metric='sumrate',
        name='dist_sweep_test_long',
        width=plot_width,
        height=plot_height,
        legend=plot_legend,
        colors=plot_colors,
        markerstyle=plot_markerstyle,
        linestyles=plot_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
    )
    plt.show()
