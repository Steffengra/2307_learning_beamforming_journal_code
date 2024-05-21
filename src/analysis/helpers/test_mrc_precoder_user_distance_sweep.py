
import numpy as np

import src
from src.analysis.helpers.test_precoder_user_distance_sweep import test_precoder_user_distance_sweep
from src.data.calc_sum_rate_no_iui import calc_sum_rate_no_iui
from src.analysis.helpers.get_precoding import get_precoding_mrc


def test_mrc_precoder_user_distance_sweep(
    config: 'src.config.config.Config',
    distance_sweep_range: np.ndarray,
) -> None:
    """Test the MRC precoder over a range of distances with zero error."""

    test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='mrc',
        get_precoder_func=lambda cfg, sat_man: get_precoding_mrc(cfg, sat_man),
        calc_sum_rate_func=calc_sum_rate_no_iui,
    )
