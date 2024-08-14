
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    change_lightness,
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

        # ax.errorbar(
        #     data_entry[0],
        #     data_entry[1]['sum_rate']['mean'],
        #     yerr=data_entry[1]['sum_rate']['std'],
        #     marker=marker,
        #     color=color,
        #     linestyle=linestyle,
        #     label=legend[data_id],
        #     # solid_capstyle='round',
        #     # ecolor=change_lightness(color=color, amount=0.3),
        #     # elinewidth=2,
        #     # capthick=2,
        #     # markevery=[0, -1],
        #     # markeredgecolor='black',
        #     # fillstyle='none'
        # )

        ax.plot(
            data_entry[0],
            data_entry[1]['sum_rate']['mean'],
            marker=marker,
            color=color,
            linestyle=linestyle,
            label=legend[data_id],
            fillstyle='none',
        )

        # ax.annotate(
        #     text=legend[data_id],
        #     xy=(data_entry[0][-1], data_entry[1]['sum_rate']['mean'][-1]),
        #     xytext=(10, 0),
        #     textcoords='offset points',
        #     ha='left',
        #     va='center_baseline',
        #     color=color,
        #     fontsize=8.33,
        # )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if legend:
        from matplotlib import container
        handles, labels = ax.get_legend_handles_labels()
        handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]  # remove error bars
        legend = ax.legend(
            handles, legend,
            ncols=1,
            loc='lower left',
            bbox_to_anchor=(0.01, 0.15)
        )
        legend.get_frame().set_linewidth(0.8)

    # ax.annotate(
    #     text='test',
    #     xy=(data_entry[0][-1], data_entry[1]['sum_rate']['mean'][-1]),
    #     xytext=(10, 0),
    #     textcoords='offset points',
    #     ha='left',
    #     va='center_baseline',
    #     fontsize=
    # )
    arr = mpatches.FancyArrowPatch(
        (0.475, 1), (0.475, 3.3),
        arrowstyle='-|>',
        # arrowstyle='simple,head_width=0.7',
        mutation_scale=15,
        # fill='black',
        color='darkgrey',
    )
    ax.add_patch(arr)
    ax.annotate(
        'better',
        (1.0, .7),
        xycoords=arr,
        ha='left',
        va='center',
        rotation=90,
        fontsize=8.33,
        color=change_lightness('black', 0.7),
    )

    arr2 = mpatches.FancyArrowPatch(
        (0, 0.7), (0.50, 0.7),
        arrowstyle='-|>',
        # arrowstyle='simple,head_width=0.7',
        mutation_scale=15,
        # fill='black',
        color='darkgrey',
    )
    ax.add_patch(arr2)
    ax.annotate(
        'increasing error on CSIT',
        (0.5, 1.0),
        xycoords=arr2,
        ha='center',
        va='bottom',
        fontsize=8.33,
        color=change_lightness('black', 0.7),
    )

    # ax.set_xlim([-0.01, 0.2])

    generic_styling(ax=ax)
    fig.tight_layout(pad=0)

    save_figures(plots_parent_path=plots_parent_path, plot_name=name, padding=0)


if __name__ == '__main__':

    cfg = Config()
    plot_cfg = PlotConfig()

    data_paths = [
        Path(cfg.output_metrics_path,
             '2sat_4ant_100k~0_3usr_1k~500', 'error_sweep',
             'testing_mmse_decentralized_blind_sweep_0.0_0.5.gzip'),
        # Path(cfg.output_metrics_path,
        #      '2sat_4ant_100k~0_3usr_1k~500', 'error_sweep',
        #      'testing_robust_slnr_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path,
             '2sat_4ant_100k~0_3usr_1k~500', 'error_sweep',
             'testing_sac_0.0_decentralized_blind_sweep_0.0_0.5.gzip'),
        Path(cfg.output_metrics_path,
             '2sat_4ant_100k~0_3usr_1k~500', 'error_sweep',
             'testing_sac_0.25_decentralized_blind_sweep_0.0_0.5.gzip'),
        # Path(cfg.output_metrics_path,
        #      '2sat_4ant_100k~0_3usr_1k~500', 'error_sweep',
        #      'testing_learned_0.25_sweep_0.0_0.5.gzip'),
    ]

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 15 / 20

    plot_legend = [
        'MMSE',
        # 'SLNR',
        'SAC $\Delta\epsilon=0.0$',
        'SAC $\Delta\epsilon=0.25$',
    ]

    plot_markerstyle = [
        'o',
        # 'x',
        's',
        'D',
        'd',
    ]
    plot_colors = [
        plot_cfg.cp2['black'],
        # plot_cfg.cp2['black'],
        plot_cfg.cp3['blue2'],
        plot_cfg.cp3['red2'],
        plot_cfg.cp3['red3'],
    ]
    plot_linestyles = [
        '-',
        # ':',
        '-',
        '-',
        '-',
    ]

    plot_error_sweep_testing_graph(
        paths=data_paths,
        name='error_sweep_2sat_decentralized_blind',
        width=plot_width,
        xlabel='Error Bound $\Delta\\varepsilon_{\mathrm{aod}}$',
        ylabel='Avg. Sum Rate (bits/s/Hz)',
        height=plot_height,
        legend=plot_legend,
        colors=plot_colors,
        markerstyle=plot_markerstyle,
        linestyles=plot_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
    )
    plt.show()
