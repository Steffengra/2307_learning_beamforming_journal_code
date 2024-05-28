
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
)


def plot_training_graph(
        path,
        name,
        width,
        height,
        plots_parent_path: Path,
        xlabel: None or str = None,
        ylabel: None or str = None,
):

    with gzip_open(path, 'rb') as file:
        data = pickle_load(file)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.scatter(range(len(data['mean_sum_rate_per_episode'])), data['mean_sum_rate_per_episode'])

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    generic_styling(ax=ax)
    fig.tight_layout(pad=0)

    save_figures(plots_parent_path=plots_parent_path, plot_name=name, padding=0)


if __name__ == '__main__':
    cfg = Config()
    plot_cfg = PlotConfig()
    # path = Path(cfg.output_metrics_path, 'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000', 'err_mult_on_steering_cos', 'single_error', 'training_error_0.0_userwiggle_30.gzip')
    path = Path(cfg.output_metrics_path, 'training_error_full.gzip')

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 9 / 20

    plot_training_graph(
        path,
        name='training_test',
        width=plot_width,
        height=plot_height,
        plots_parent_path=plot_cfg.plots_parent_path,
        xlabel='Training Episode',
        ylabel='Mean Reward'
    )
    plt.show()
