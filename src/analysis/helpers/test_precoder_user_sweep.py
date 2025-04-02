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
    calc_sum_rate_func,
) -> None:
    """
    Calculate the sum rates that a given precoder achieves for a given config
    over a given range of numbers of users with no channel error
    mode: ['user', 'satellite']
    """

    def progress_print() -> None:
        progress = (user_number_idx + 1) / (len(user_number_sweep_range))
        progress_printer(progress=progress, real_time_start=real_time_start)

    def save_results():
        name = f'testing_{precoder_name}sweep_{round(user_number_sweep_range[0])}_{round(user_number_sweep_range[-1])}.gzip'
        results_path = Path(config.output_metrics_path, config.config_learner.training_name, 'distance_sweep')
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
        'sum_rate': {
            'mean': np.zeros(len(user_number_sweep_range)),
            'std': np.zeros(len(user_number_sweep_range)),
        },
    }

    for user_number_idx, user_number_value in enumerate(user_number_sweep_range):

        user_mask = np.concatenate([np.ones(user_number_value), np.zeros(user_number_sweep_range[-1] - user_number_value)])
        user_mask = np.roll(user_mask, int(user_number_sweep_range[-1]/2)-int(user_number_value/2))  # move users to center to keep block with same distances
        user_manager.set_active_users(user_mask)

        # set up per monte carlo metrics
        sum_rate_per_monte_carlo = np.zeros(monte_carlo_iterations)

        for iter_idx in range(monte_carlo_iterations):
            update_sim(config, satellite_manager, user_manager)

            w_precoder = get_precoder_func(
                config,
                user_manager,
                satellite_manager,
            )
            sum_rate = calc_sum_rate_func(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_precoder,
                noise_power_watt=config.noise_power_watt,
            )

            sum_rate_per_monte_carlo[iter_idx] = sum_rate

            if config.verbosity > 0:
                if iter_idx % 50 == 0:
                    progress_print()

        metrics['sum_rate']['mean'][user_number_idx] = np.mean(sum_rate_per_monte_carlo)
        metrics['sum_rate']['std'][user_number_idx] = np.std(sum_rate_per_monte_carlo)


    if profiler is not None:
        end_profiling(profiler)

    save_results()

    plot_sweep(
        x=user_number_sweep_range,
        y=metrics['sum_rate']['mean'],
        yerr=metrics['sum_rate']['std'],
        xlabel='user_num',
        ylabel='sum rate',
        title=precoder_name,
    )

    if config.show_plots:
        plt_show()
