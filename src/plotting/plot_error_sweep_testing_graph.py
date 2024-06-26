
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
    save_figures,
    generic_styling,
)


def plot_error_sweep_testing_graph(
        paths,
        name,
        width,
        height,
        xlabel,
        ylabel,
        plots_parent_path,
        legend: list or None = None,
        colors: list or None = None,
        markerstyle: list or None = None,
        linestyles: list or None = None,
) -> None:

    fig, ax = plt.subplots(figsize=(width, height))

    data = []
    for path in paths:
        with gzip_open(path, 'rb') as file:
            data.append(pickle_load(file))

    for data_id, data_entry in enumerate(data):

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

        ax.plot(
            data_entry[0],
            data_entry[1]['sum_rate']['mean'],
            marker=marker,
            color=color,
            linestyle=linestyle,
            label=legend[data_id],
            fillstyle='none',
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if legend:
        from matplotlib import container
        handles, labels = ax.get_legend_handles_labels()
        handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]  # remove error bars
        legend = ax.legend(
            handles, legend,
            ncols=2,
            # loc='lower left',
        )
        legend.get_frame().set_linewidth(0.8)

    generic_styling(ax=ax)
    fig.tight_layout(pad=0)

    save_figures(plots_parent_path=plots_parent_path, plot_name=name, padding=0)


if __name__ == '__main__':

    cfg = Config()
    plot_cfg = PlotConfig()

    data_paths = [
        Path(cfg.output_metrics_path,
             '1sat_16ant_100k~0_3usr_100k~50k', 'error_sweep',
             'testing_mmse_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path,
             '1sat_16ant_100k~0_3usr_100k~50k', 'error_sweep',
             'testing_robust_slnr_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path,
             '1sat_16ant_100k~0_3usr_100k~50k', 'error_sweep',
             'testing_learned_0.0_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path,
             '1sat_16ant_100k~0_3usr_100k~50k', 'error_sweep',
             'testing_learned_0.025_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path,
             '1sat_16ant_100k~0_3usr_100k~50k', 'error_sweep',
             'testing_learned_0.05_sweep_0.0_0.1.gzip'),
    ]

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 15 / 20

    plot_legend = [
        'MMSE',
        'SLNR',
        'SAC1',
        'SAC2',
        'SAC3',
    ]

    plot_markerstyle = [
        'o',
        'x',
        's',
        'D',
        'd',
    ]
    plot_colors = [
        plot_cfg.cp2['black'],
        plot_cfg.cp2['black'],
        plot_cfg.cp3['blue2'],
        plot_cfg.cp3['red2'],
        plot_cfg.cp3['red3'],
    ]
    plot_linestyles = [
        '-',
        ':',
        '-',
        '-',
        '-',
    ]

    plot_error_sweep_testing_graph(
        paths=data_paths,
        name='error_sweep_test',
        width=plot_width,
        xlabel='Error Bound',
        ylabel='Avg. Sum Rate (bits/s/Hz)',
        height=plot_height,
        legend=plot_legend,
        colors=plot_colors,
        markerstyle=plot_markerstyle,
        linestyles=plot_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
    )
    plt.show()
