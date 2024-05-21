
import numpy as np

import src
from src.analysis.helpers.test_precoder_user_distance_sweep import test_precoder_user_distance_sweep
from src.data.calc_sum_rate import calc_sum_rate
from src.analysis.helpers.get_precoding import get_precoding_mmse


def test_mmse_precoder_user_distance_sweep(
    config: 'src.config.config.Config',
    distance_sweep_range: np.ndarray,
) -> None:
    """Test the MMSE precoder over a range of distances with zero error."""

    test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='mmse',
        get_precoder_func=lambda cfg, sat_man: get_precoding_mmse(cfg, sat_man),
        calc_sum_rate_func=calc_sum_rate,
    )
