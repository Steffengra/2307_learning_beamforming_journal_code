
from pathlib import Path
from sys import path as sys_path
project_root_path = Path(Path(__file__).parent, '../..', '..')
sys_path.append(str(project_root_path.resolve()))

import numpy as np

from src.config.config import Config
from src.models.train_sac import train_sac
from src.models.train_sac_adapt_robust_slnr_power import train_sac_adapt_robust_slnr_power
from src.models.train_sac_adapt_robust_slnr_complete import train_sac_adapt_robust_slnr_complete
from src.analysis.helpers.test_mmse_precoder import test_mmse_precoder_error_sweep
from src.analysis.helpers.test_robust_slnr_precoder import test_robust_slnr_precoder_error_sweep
from src.analysis.helpers.test_learned_precoder import (
    test_sac_precoder_error_sweep,
    test_adapted_slnr_powerscaled_error_sweep,
    test_adapted_slnr_complete_error_sweep,
)


def learn_baseline(
        user_dist: float,
        user_wiggle: float,
        additive_error_on_cosine_of_aod: float,
        testing_range: np.ndarray,
) -> None:

    config = Config()
    assert config.sat_nr == 1
    assert config.sat_tot_ant_nr == 16
    assert config.user_nr == 3
    assert config.config_error_model.error_rng_parametrizations['large_scale_fading']['args']['sigma'] == 0.1

    config.profile = False
    config.show_plots = False
    config.verbosity = 0

    config.user_dist_average = user_dist
    config.user_dist_bound = user_wiggle
    config.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['low'] = -additive_error_on_cosine_of_aod
    config.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['high'] = additive_error_on_cosine_of_aod

    config_name = config.generate_name_from_config()
    config_name += f'_additive_{additive_error_on_cosine_of_aod}'
    config.config_learner.training_name = config_name

    best_model_path, _ = train_sac(config=config)

    test_sac_precoder_error_sweep(
        config=config,
        model_path=best_model_path,
        error_sweep_range=testing_range,
        monte_carlo_iterations=5_000,
        error_sweep_parameter='additive_error_on_cosine_of_aod',
    )

    if additive_error_on_cosine_of_aod == 0.0:
        test_mmse_precoder_error_sweep(
            config=config,
            error_sweep_parameter='additive_error_on_cosine_of_aod',
            error_sweep_range=testing_range,
            monte_carlo_iterations=5_000,
        )

        test_robust_slnr_precoder_error_sweep(
            config=config,
            error_sweep_parameter='additive_error_on_cosine_of_aod',
            error_sweep_range=testing_range,
            monte_carlo_iterations=5_000,
        )


def learn_other_error_model(
        additive_error_on_cosine_of_aod,
        additive_error_on_aod,
        testing_range,
) -> None:

    config = Config()
    assert config.sat_nr == 1
    assert config.sat_tot_ant_nr == 16
    assert config.user_nr == 3
    assert config.config_error_model.error_rng_parametrizations['large_scale_fading']['args']['sigma'] == 0.1

    config.profile = False
    config.show_plots = False
    config.verbosity = 0

    config.user_dist_average = 100_000
    config.user_dist_bound = 50_000
    config.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['low'] = -additive_error_on_cosine_of_aod
    config.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['high'] = additive_error_on_cosine_of_aod
    config.config_error_model.error_rng_parametrizations['additive_error_on_aod']['args']['scale'] = additive_error_on_aod

    config_name = config.generate_name_from_config()
    config_name += f'_additive_{additive_error_on_cosine_of_aod}_{additive_error_on_aod}'
    config.config_learner.training_name = config_name

    print('training sac')
    best_model_path, _ = train_sac(config=config)
    print(f'best model path {best_model_path}')

    print('testing sac')
    test_sac_precoder_error_sweep(
        config=config,
        model_path=best_model_path,
        error_sweep_range=testing_range,
        monte_carlo_iterations=5_000,
        error_sweep_parameter='additive_error_on_aod',
    )

    if additive_error_on_cosine_of_aod == 0.0:
        print('testing mmse')
        test_mmse_precoder_error_sweep(
            config=config,
            error_sweep_parameter='additive_error_on_aod',
            error_sweep_range=testing_range,
            monte_carlo_iterations=5_000,
        )

        print('testing slnr')
        test_robust_slnr_precoder_error_sweep(
            config=config,
            error_sweep_parameter='additive_error_on_aod',
            error_sweep_range=testing_range,
            monte_carlo_iterations=5_000,
        )


def learn_adapt_slnr(
        user_dist,
        user_wiggle,
        additive_error_on_cosine_of_aod,
        testing_range,
) -> None:

    config = Config()
    assert config.sat_nr == 1
    assert config.sat_tot_ant_nr == 16
    assert config.user_nr == 3
    assert config.config_error_model.error_rng_parametrizations['large_scale_fading']['args']['sigma'] == 0.1

    config.profile = False
    config.show_plots = False
    config.verbosity = 0

    config.user_dist_average = user_dist
    config.user_dist_bound = user_wiggle
    config.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['low'] = -additive_error_on_cosine_of_aod
    config.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['high'] = additive_error_on_cosine_of_aod

    config_name = config.generate_name_from_config()
    config_name += f'_additive_{additive_error_on_cosine_of_aod}'
    config.config_learner.training_name = config_name

    print('training adapt')
    best_model_path_adapt_full, _ = train_sac_adapt_robust_slnr_complete(config=config)
    best_model_path_adapt_power, _ = train_sac_adapt_robust_slnr_power(config=config)

    print('testing adapt')
    test_adapted_slnr_complete_error_sweep(
        config=config,
        model_path=best_model_path_adapt_full,
        error_sweep_parameter='additive_error_on_cosine_of_aod',
        error_sweep_range=testing_range,
        monte_carlo_iterations=5_000,
    )
    test_adapted_slnr_powerscaled_error_sweep(
        config=config,
        model_path=best_model_path_adapt_power,
        error_sweep_parameter='additive_error_on_cosine_of_aod',
        error_sweep_range=testing_range,
        monte_carlo_iterations=5_000,
    )


def main():

    testing_error_sweep_range_100k = np.linspace(0, 0.1, 9)
    testing_error_sweep_range_10k = np.linspace(0, 0.12, 9)
    testing_error_sweep_range_100k_error2 = np.linspace(0, 0.1, 9)

    learn_baseline(user_dist=100_000, user_wiggle=50_000, additive_error_on_cosine_of_aod=0.0, testing_range=testing_error_sweep_range_100k)
    learn_baseline(user_dist=100_000, user_wiggle=50_000, additive_error_on_cosine_of_aod=0.025, testing_range=testing_error_sweep_range_100k)
    learn_baseline(user_dist=100_000, user_wiggle=50_000, additive_error_on_cosine_of_aod=0.05, testing_range=testing_error_sweep_range_100k)

    learn_baseline(user_dist=10_000, user_wiggle=5_000, additive_error_on_cosine_of_aod=0.00, testing_range=testing_error_sweep_range_10k)
    learn_baseline(user_dist=10_000, user_wiggle=5_000, additive_error_on_cosine_of_aod=0.03, testing_range=testing_error_sweep_range_10k)

    # learn_other_error_model(additive_error_on_cosine_of_aod=0.1, additive_error_on_aod=0.0, testing_range=testing_error_sweep_range_100k_error2)
    learn_other_error_model(additive_error_on_cosine_of_aod=0.1, additive_error_on_aod=0.025, testing_range=testing_error_sweep_range_100k_error2)
    learn_other_error_model(additive_error_on_cosine_of_aod=0.1, additive_error_on_aod=0.05, testing_range=testing_error_sweep_range_100k_error2)

    learn_adapt_slnr(user_dist=10_000, user_wiggle=5_000, additive_error_on_cosine_of_aod=0.00, testing_range=testing_error_sweep_range_10k)
    learn_adapt_slnr(user_dist=10_000, user_wiggle=5_000, additive_error_on_cosine_of_aod=0.03, testing_range=testing_error_sweep_range_10k)


if __name__ == '__main__':
    main()
