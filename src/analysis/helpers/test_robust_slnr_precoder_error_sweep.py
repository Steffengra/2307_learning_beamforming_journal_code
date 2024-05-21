
import numpy as np

import src
from src.analysis.helpers.test_precoder_error_sweep import test_precoder_error_sweep
from src.data.calc_sum_rate import calc_sum_rate
from src.analysis.helpers.get_precoding import get_precoding_robust_slnr


def test_robust_slnr_precoder_error_sweep(
        config: 'src.config.config.Config',
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
) -> None:
    """Test the robust SLNR precoder for a range of error configuration with monte carlo average."""

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='robust_slnr',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, sat_man: get_precoding_robust_slnr(cfg, sat_man),
        calc_sum_rate_func=calc_sum_rate,
    )
