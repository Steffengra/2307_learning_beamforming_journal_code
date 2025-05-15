
from datetime import datetime
from pathlib import Path
import gzip
import pickle

import numpy as np
from matplotlib.pyplot import show as plt_show

import src
from src.data.satellite_manager import (
    SatelliteManager,
)
from src.data.user_manager import (
    UserManager,
)
from src.utils.plot_sweep import (
    plot_sweep,
)
from src.utils.profiling import (
    start_profiling,
    end_profiling,
)
from src.utils.progress_printer import (
    progress_printer,
)
from src.utils.update_sim import (
    update_sim,
)


def test_precoder_user_distance_sweep(
    config: 'src.config.config.Config',
    distance_sweep_range,
    precoder_name: str,
    monte_carlo_iterations: int,
    mode: str,
    get_precoder_func,
    calc_reward_funcs: list,
) -> dict:
    """
    Calculate the sum rates that a given precoder achieves for a given config
    over a given range of inter-user-distances or inter-satellite-distances with no channel error
    mode: ['user', 'satellite']
    """

    def progress_print() -> None:
        progress = (distance_sweep_idx + 1) / (len(distance_sweep_range))
        progress_printer(progress=progress, real_time_start=real_time_start)

    def save_results():
        name = f'testing_{precoder_name}_{mode}sweep_{round(distance_sweep_range[0])}_{round(distance_sweep_range[-1])}.gzip'
        results_path = Path(config.output_metrics_path, config.config_learner.training_name, 'distance_sweep')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(Path(results_path, name), 'wb') as file:
            pickle.dump([distance_sweep_range, metrics], file=file)

    satellite_manager = SatelliteManager(config=config)
    user_manager = UserManager(config=config)

    real_time_start = datetime.now()

    profiler = None
    if config.profile:
        profiler = start_profiling()

    metrics = {
        calc_reward_func: {
            'mean': np.zeros(len(distance_sweep_range)),
            'std': np.zeros(len(distance_sweep_range)),
        }
        for calc_reward_func in calc_reward_funcs
    }

    for distance_sweep_idx, distance_sweep_value in enumerate(distance_sweep_range):

        metrics_per_monte_carlo = np.zeros((len(calc_reward_funcs), monte_carlo_iterations))

        if mode == 'user':
            config.user_dist_average = distance_sweep_value
            config.user_dist_bound = 0
        elif mode == 'satellite':
            config.sat_dist_average = distance_sweep_value
            config.sat_dist_bound = 0

        config.config_error_model.set_zero_error()

        for iter_idx in range(monte_carlo_iterations):

            update_sim(config, satellite_manager, user_manager)

            w_precoder = get_precoder_func(
                config,
                user_manager,
                satellite_manager,
            )

            for reward_func_id, reward_func in enumerate(calc_reward_funcs):
                metrics_per_monte_carlo[reward_func_id, iter_idx] = reward_func(
                    channel_state=satellite_manager.channel_state_information,
                    w_precoder=w_precoder,
                    noise_power_watt=config.noise_power_watt,
                )

            if config.verbosity > 0:
                if iter_idx % 50 == 0:
                    progress_print()

        for reward_func_id in range(metrics_per_monte_carlo.shape[0]):
            metrics[calc_reward_funcs[reward_func_id]]['mean'][distance_sweep_idx] = np.mean(metrics_per_monte_carlo[reward_func_id, :])
            metrics[calc_reward_funcs[reward_func_id]]['std'][distance_sweep_idx] = np.std(metrics_per_monte_carlo[reward_func_id, :])

    if profiler is not None:
        end_profiling(profiler)

    if config.verbosity > 0:
        print()
        for metric in metrics.keys():
            for error_sweep_value, mean_metric, std_metric in zip(distance_sweep_range, metrics[metric]['mean'], metrics[metric]['std']):
                print(f'{error_sweep_value:.2f}: {metric} - {mean_metric:.2f}+-{std_metric:.4f}')

    save_results()

    if config.show_plots:

        for metric in metrics.keys():
            plot_sweep(
                x=distance_sweep_range,
                y=metrics[metric]['mean'],
                yerr=metrics[metric]['std'],
                xlabel='user distance',
                ylabel=str(metric),
                title=precoder_name,
            )


        plt_show()

    return metrics
