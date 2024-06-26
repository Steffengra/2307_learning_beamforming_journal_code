
from pathlib import Path
from sys import path as sys_path
project_root_path = Path(Path(__file__).parent, '../..', '..')
sys_path.append(str(project_root_path.resolve()))

import numpy as np

from src.config.config import Config
from src.models.helpers.get_state import get_state_erroneous_channel_state_information_local
from src.models.train_sac import train_sac
from src.models.train_sac_decentralized_blind import train_sac_decentralized_blind
from src.models.train_sac_decentralized_limited import train_sac_decentralized_limited
from src.analysis.helpers.test_mmse_precoder import test_mmse_precoder_error_sweep
from src.analysis.helpers.test_robust_slnr_precoder import test_robust_slnr_precoder_error_sweep
from src.analysis.helpers.test_learned_precoder import (
    test_sac_precoder_error_sweep,
    test_sac_precoder_decentralized_blind_error_sweep,
    test_sac_precoder_decentralized_limited_error_sweep,
)


def learn_baseline(
        sat_dist: float,
        user_dist: float,
        user_wiggle: float,
        additive_error_on_cosine_of_aod: float,
        testing_range: np.ndarray,
) -> None:

    config = Config()
    assert config.sat_nr == 2
    assert config.sat_tot_ant_nr == 4
    assert config.user_nr == 3
    assert config.sat_dist_bound == 0
    assert config.config_error_model.error_rng_parametrizations['large_scale_fading']['args']['sigma'] == 0.1

    config.profile = False
    config.show_plots = False
    config.verbosity = 0

    config.sat_dist_average = sat_dist
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


def learn_decentralized(
        sat_dist: float,
        user_dist: float,
        user_wiggle: float,
        additive_error_on_cosine_of_aod: float,
        mode: str,
        testing_range: np.ndarray,
) -> None:

    config = Config()
    assert config.sat_nr == 2
    assert config.sat_tot_ant_nr == 4
    assert config.user_nr == 3
    assert config.sat_dist_bound == 0
    assert config.config_error_model.error_rng_parametrizations['large_scale_fading']['args']['sigma'] == 0.1

    config.profile = False
    config.show_plots = False
    config.verbosity = 0

    config.sat_dist_average = sat_dist
    config.user_dist_average = user_dist
    config.user_dist_bound = user_wiggle
    config.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['low'] = -additive_error_on_cosine_of_aod
    config.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['high'] = additive_error_on_cosine_of_aod

    config_name = config.generate_name_from_config()
    config_name += f'_additive_{additive_error_on_cosine_of_aod}'
    config.config_learner.training_name = config_name

    if mode == 'blind':
        best_model_path, _ = train_sac_decentralized_blind(config=config)

        test_sac_precoder_decentralized_blind_error_sweep(
            config=config,
            models_path=best_model_path,
            error_sweep_parameter='additive_error_on_cosine_of_aod',
            error_sweep_range=testing_range,
            monte_carlo_iterations=5_000,
        )

    elif mode == 'L1':
        config.local_csi_own_quality = 'error_free'
        config.local_csi_others_quality = 'erroneous'
        best_model_path, _ = train_sac_decentralized_limited(config=config)

        test_sac_precoder_decentralized_limited_error_sweep(
            config=config,
            models_path=best_model_path,
            error_sweep_parameter='additive_error_on_cosine_of_aod',
            error_sweep_range=testing_range,
            monte_carlo_iterations=5_000,
        )

    elif mode == 'L2':
        config.local_csi_own_quality = 'erroneous'
        config.local_csi_others_quality = 'scaled_erroneous'
        best_model_path, _ = train_sac_decentralized_limited(config=config)

        test_sac_precoder_decentralized_limited_error_sweep(
            config=config,
            models_path=best_model_path,
            error_sweep_parameter='additive_error_on_cosine_of_aod',
            error_sweep_range=testing_range,
            monte_carlo_iterations=5_000,
        )

    else:
        raise ValueError(f'Unknown mode {mode}')


def main():

    testing_error_sweep_range_100k = np.linspace(0, 0.5, 9)
    testing_error_sweep_range_10k = np.linspace(0, 0.5, 9)

    learn_baseline(sat_dist=100_000, user_dist=1_000, user_wiggle=500, additive_error_on_cosine_of_aod=0.0, testing_range=testing_error_sweep_range_100k)
    learn_baseline(sat_dist=100_000, user_dist=1_000, user_wiggle=500, additive_error_on_cosine_of_aod=0.125, testing_range=testing_error_sweep_range_100k)
    learn_baseline(sat_dist=100_000, user_dist=1_000, user_wiggle=500, additive_error_on_cosine_of_aod=0.25, testing_range=testing_error_sweep_range_100k)

    learn_baseline(sat_dist=10_000, user_dist=1_000, user_wiggle=500, additive_error_on_cosine_of_aod=0.0, testing_range=testing_error_sweep_range_10k)
    learn_baseline(sat_dist=10_000, user_dist=1_000, user_wiggle=500, additive_error_on_cosine_of_aod=0.125, testing_range=testing_error_sweep_range_10k)
    learn_baseline(sat_dist=10_000, user_dist=1_000, user_wiggle=500, additive_error_on_cosine_of_aod=0.25, testing_range=testing_error_sweep_range_10k)

    learn_decentralized(sat_dist=100_000, user_dist=1_000, user_wiggle=500, additive_error_on_cosine_of_aod=0.0, mode='blind', testing_range=testing_error_sweep_range_100k)
    learn_decentralized(sat_dist=100_000, user_dist=1_000, user_wiggle=500, additive_error_on_cosine_of_aod=0.25, mode='blind', testing_range=testing_error_sweep_range_100k)
    learn_decentralized(sat_dist=100_000, user_dist=1_000, user_wiggle=500, additive_error_on_cosine_of_aod=0.0, mode='L1', testing_range=testing_error_sweep_range_100k)
    learn_decentralized(sat_dist=100_000, user_dist=1_000, user_wiggle=500, additive_error_on_cosine_of_aod=0.25, mode='L1', testing_range=testing_error_sweep_range_100k)
    learn_decentralized(sat_dist=100_000, user_dist=1_000, user_wiggle=500, additive_error_on_cosine_of_aod=0.0, mode='L2', testing_range=testing_error_sweep_range_100k)
    learn_decentralized(sat_dist=100_000, user_dist=1_000, user_wiggle=500, additive_error_on_cosine_of_aod=0.25, mode='L2', testing_range=testing_error_sweep_range_100k)


if __name__ == '__main__':
    main()
