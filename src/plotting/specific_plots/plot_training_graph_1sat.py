
import numpy as np
import matplotlib.pyplot as plt
from gzip import (
    open as gzip_open,
)
from pickle import (
    load as pickle_load,
)
from pathlib import (
    Path,
)

from src.config.config import (
    Config,
)
from src.config.config_plotting import (
    PlotConfig,
    generic_styling,
    save_figures,
    change_lightness,
)


def plot_training_graph(
        paths,
        name,
        width,
        height,
        window_length,
        plots_parent_path: Path,
        colors: None or list = None,
        legend: None or list = None,
        plot_markerstyles: None or list = None,
        xlabel: None or str = None,
        ylabel: None or str = None,
):

    fig, ax = plt.subplots(figsize=(width, height))
    for path_id, path in enumerate(paths):
        with gzip_open(path, 'rb') as file:
            data = pickle_load(file)

        averaged_data = np.zeros(len(data['mean_sum_rate_per_episode']))
        averaged_data_low = np.zeros(len(data['mean_sum_rate_per_episode']))
        averaged_data_high = np.zeros(len(data['mean_sum_rate_per_episode']))

        for data_id, datum in enumerate(data['mean_sum_rate_per_episode']):
            if data_id == 0:
                averaged_data[0] = data['mean_sum_rate_per_episode'][0]
                continue

            start = max(0, data_id - window_length)
            averaged_data[data_id] = np.mean(data['mean_sum_rate_per_episode'][start:data_id+1])
            # averaged_data_low[data_id] = np.min(data['mean_sum_rate_per_episode'][start:data_id+1])
            # averaged_data_high[data_id] = np.max(data['mean_sumrate_per_episode'][start:data_id+1])
            # averaged_data_low[data_id] = averaged_data[data_id]-np.std(data['mean_sum_rate_per_episode'][start:data_id+1])
            # averaged_data_high[data_id] = averaged_data[data_id]+np.std(data['mean_sum_rate_per_episode'][start:data_id+1])
            averaged_data_low[data_id] = min(averaged_data[data_id], data['mean_sum_rate_per_episode'][data_id])
            averaged_data_high[data_id] = max(averaged_data[data_id], data['mean_sum_rate_per_episode'][data_id])

        ax.plot(
            range(len(averaged_data)),
            averaged_data,
            color=colors[path_id] if colors else None,
            label=legend[path_id] if legend else None,
            marker=plot_markerstyles[path_id] if plot_markerstyles else None,
            # markevery=1000,
            markevery=[-1],
            fillstyle='none',
        )
        # ax.fill_between(
        #     range(len(averaged_data)),
        #     y1=averaged_data_low,
        #     y2=averaged_data_high,
        #     color=change_lightness(colors[path_id], amount=0.2) if colors else None,
        # )

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if legend:
        ax.legend(ncols=2)

    generic_styling(ax=ax)
    fig.tight_layout(pad=0)

    save_figures(plots_parent_path=plots_parent_path, plot_name=name, padding=0)


if __name__ == '__main__':
    cfg = Config()
    plot_cfg = PlotConfig()
    # path
    # = Path(cfg.output_metrics_path, 'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000', 'err_mult_on_steering_cos', 'single_error', 'training_error_0.0_userwiggle_30.gzip')
    paths = [
        Path(cfg.output_metrics_path, '1sat_16ant_100k~0_3usr_100k~50k_additive_0.0', 'base', 'training_error_learned_full.gzip'),
        Path(cfg.output_metrics_path, '1sat_16ant_100k~0_3usr_100k~50k_vanilla_sac', 'base', 'training_error_learned_full.gzip'),
        Path(cfg.output_metrics_path, '1sat_16ant_100k~0_3usr_100k~50k_additive_0.0', 'base', 'training_error_adapt_slnr_complete.gzip'),
        Path(cfg.output_metrics_path, '1sat_16ant_100k~0_3usr_100k~50k_additive_0.0', 'base', 'training_error_adapt_slnr_power.gzip'),
    ]

    colors = [
        plot_cfg.cp3['blue1'],
        plot_cfg.cp3['black'],
        plot_cfg.cp3['blue2'],
        plot_cfg.cp3['blue3'],
    ]

    legend = [
        'Base',
        'Vanilla',
        'Adapt SLNR',
        'Adapt SLNR Power',
    ]

    plot_markerstyles = [
        's',
        '|',
        'x',
        'o',
        'd',
    ]

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 9 / 20

    plot_training_graph(
        paths,
        name='journal_training_convergence',
        width=plot_width,
        height=plot_height,
        window_length=1000,
        colors=colors,
        legend=legend,
        plot_markerstyles=plot_markerstyles,
        plots_parent_path=plot_cfg.plots_parent_path,
        xlabel='Training Episode',
        ylabel='Avg. Sum Rate\n(bits/s/Hz)'
    )
    plt.show()
