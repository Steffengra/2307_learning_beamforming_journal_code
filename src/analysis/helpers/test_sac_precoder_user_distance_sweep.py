
from pathlib import Path

import numpy as np

import src
from src.analysis.helpers.test_precoder_user_distance_sweep import test_precoder_user_distance_sweep
from src.data.calc_sum_rate import calc_sum_rate
from src.utils.load_model import load_model
from src.analysis.helpers.get_precoding import get_precoding_learned


def test_sac_precoder_user_distance_sweep(
    config: 'src.config.config.Config',
    distance_sweep_range: np.ndarray,
    model_path: Path,
) -> None:
    """Test a precoder over a range of distances with zero error."""

    precoder_network, norm_factors = load_model(model_path)

    if norm_factors != {}:
        config.config_learner.get_state_args['norm_state'] = True

    test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='learned',
        get_precoder_func=lambda cfg, sat_man: get_precoding_learned(cfg, sat_man, norm_factors, precoder_network),
        calc_sum_rate_func=calc_sum_rate,
    )
