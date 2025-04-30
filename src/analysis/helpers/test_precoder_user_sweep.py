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


def test_precoder_user_sweep(
    config: 'src.config.config.Config',
    user_number_sweep_range,
    monte_carlo_iterations: int,
    precoder_name: str,
    get_precoder_func,
    calc_reward_funcs: list,
) -> dict:
    """
    Calculate the sum rates that a given precoder achieves for a given config
    over a given range of numbers of users with no channel error
    mode: ['user', 'satellite']
    """

    def progress_print() -> None:
        progress = (user_number_idx + 1) / (len(user_number_sweep_range))
        progress_printer(progress=progress, real_time_start=real_time_start)

    def save_results():
        name = f'testing_{precoder_name}_sweep_{round(user_number_sweep_range[0])}_{round(user_number_sweep_range[-1])}.gzip'
        results_path = Path(config.output_metrics_path, config.config_learner.training_name, 'user_number_sweep')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(Path(results_path, name), 'wb') as file:
            pickle.dump([user_number_sweep_range, metrics], file=file)

    config.user_nr = user_number_sweep_range[-1]

    config.user_activity_selection = 'keep_as_is'

    satellite_manager = SatelliteManager(config=config)
    user_manager = UserManager(config=config)

    real_time_start = datetime.now()

    profiler = None
    if config.profile:
        profiler = start_profiling()

    metrics = {
        calc_reward_func: {
            'mean': np.zeros(len(user_number_sweep_range)),
            'std': np.zeros(len(user_number_sweep_range)),
        }
        for calc_reward_func in calc_reward_funcs
    }

    for user_number_idx, user_number_value in enumerate(user_number_sweep_range):

        user_mask = np.concatenate([np.ones(user_number_value), np.zeros(user_number_sweep_range[-1] - user_number_value)])
        user_mask = np.roll(user_mask, int(user_number_sweep_range[-1]/2)-int(user_number_value/2))  # move users to center to keep block with same distances
        user_manager.set_active_users(user_mask)

        # set up per monte carlo metrics
        metrics_per_monte_carlo = np.zeros((len(calc_reward_funcs), monte_carlo_iterations))

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
            metrics[calc_reward_funcs[reward_func_id]]['mean'][user_number_idx] = np.mean(metrics_per_monte_carlo[reward_func_id, :])
            metrics[calc_reward_funcs[reward_func_id]]['std'][user_number_idx] = np.std(metrics_per_monte_carlo[reward_func_id, :])

    if profiler is not None:
        end_profiling(profiler)

    if config.verbosity > 0:
        print()
        for metric in metrics.keys():
            for error_sweep_value, mean_metric, std_metric in zip(user_number_sweep_range, metrics[metric]['mean'], metrics[metric]['std']):
                print(f'{error_sweep_value:.2f}: {metric} - {mean_metric:.2f}+-{std_metric:.4f}')
    save_results()

    for metric in metrics.keys():
        plot_sweep(
            x=user_number_sweep_range,
            y=metrics[metric]['mean'],
            yerr=metrics[metric]['std'],
            xlabel='user number',
            ylabel=str(metric),
            title=precoder_name,
        )

    if config.show_plots:
        plt_show()

    return metrics
