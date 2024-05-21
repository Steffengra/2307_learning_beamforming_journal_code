
from pathlib import Path

import numpy as np

import src
from src.analysis.helpers.test_precoder_error_sweep import test_precoder_error_sweep
from src.data.calc_sum_rate import calc_sum_rate
from src.utils.load_model import load_models
from src.analysis.helpers.get_precoding import get_precoding_learned_decentralized


def test_sac_precoder_decentralized_error_sweep(
        config: 'src.config.config.Config',
        model_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
) -> None:
    """Test decentralized learned SAC precoders for a range of error configuration with monte carlo average."""

    precoder_networks, norm_factors = load_models(model_path)

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='sac_decentralized',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, sat_man: get_precoding_learned_decentralized(cfg, sat_man, norm_factors, precoder_networks),
        calc_sum_rate_func=calc_sum_rate,
    )