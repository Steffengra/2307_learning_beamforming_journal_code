
from pathlib import Path

import numpy as np

import src
from src.analysis.helpers.test_precoder_error_sweep import test_precoder_error_sweep
from src.data.calc_sum_rate import calc_sum_rate
from src.utils.load_model import load_model
from src.utils.get_precoding import get_precoding_adapted_slnr_powerscaled


def test_adapted_slnr_powerscaled_error_sweep(
        config: 'src.config.config.Config',
        model_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
) -> None:

    scaling_network, norm_factors = load_model(model_path)

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='adapted_slnr_complete',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, sat_man: get_precoding_adapted_slnr_powerscaled(cfg, sat_man, norm_factors, scaling_network),
        calc_sum_rate_func=calc_sum_rate,
    )
