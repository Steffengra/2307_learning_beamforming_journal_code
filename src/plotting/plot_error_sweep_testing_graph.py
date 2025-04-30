
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
        with gzip_open(path, 'rb') as file:
            data.append(pickle_load(file))

    for data_id, data_entry in enumerate(data):
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

        ax.plot(
            data_entry[0],
            data_entry[1][metric_key]['mean'],
            marker=marker,
            color=color,
            linestyle=linestyle,
            label=legend[data_id],
            fillstyle='none',
        )

    ax.set_xlabel('Error Bound $\Delta \epsilon$')

    if metric == 'sumrate':
        ax.set_ylabel('Rate $ R $ [bps/Hz]')
    elif metric == 'fairness':
        ax.set_ylabel('Fairness $F$')

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

    save_figures(plots_parent_path=plots_parent_path, plot_name=name+'_'+metric, padding=0)


if __name__ == '__main__':

    cfg = Config()
    plot_cfg = PlotConfig()

    data_paths = [
        Path(cfg.output_metrics_path,
             'test', 'error_sweep',
             'testing_mmse_sweep_0.0_0.5.gzip'),
        Path(cfg.output_metrics_path,
             'test', 'error_sweep',
             'testing_learned_sweep_0.0_0.5.gzip'),
        Path(cfg.output_metrics_path,
             'test', 'error_sweep',
             'testing_learned_rsma_full_sweep_0.0_0.5.gzip'),
        Path(cfg.output_metrics_path,
             'test', 'error_sweep',
             'testing_learned_rsma_power_factor_sweep_0.0_0.5.gzip'),

    ]

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.42

    plot_legend = ['MMSE', 'SAC', 'RSMA full', 'RSMA power']
    plot_markerstyle = ['v', 'o', 's', '^']
    plot_colors = [plot_cfg.cp2['gold'], plot_cfg.cp2['blue'], plot_cfg.cp2['magenta'], plot_cfg.cp2['black']]
    plot_linestyles = ['-', '-', '-', '--']

    plot_error_sweep_testing_graph(
        paths=data_paths,
        metric='fairness',
        name='error_sweep_test',
        width=plot_width,
        height=plot_height,
        legend=plot_legend,
        colors=plot_colors,
        markerstyle=plot_markerstyle,
        linestyles=plot_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
    )
    plt.show()
