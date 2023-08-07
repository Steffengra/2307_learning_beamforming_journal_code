
import numpy as np

import src
from src.analysis.helpers.test_precoder_error_sweep import (
    test_precoder_error_sweep,
)
from src.data.precoder.mrc_precoder import (
    mrc_precoder_normalized,
)
from src.data.calc_sum_rate_no_iui import (
    calc_sum_rate_no_iui,
)


def test_mrc_precoder_error_sweep(
        config: 'src.config.config.Config',
        csit_error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
) -> None:

    def get_precoder_mrc(
        cfg: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
    ):
        w_mrc = mrc_precoder_normalized(
            channel_matrix=satellite_manager.erroneous_channel_state_information,
            **cfg.mrc_args,
        )

        return w_mrc

    test_precoder_error_sweep(
        config=config,
        csit_error_sweep_range=csit_error_sweep_range,
        precoder_name='mrc',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=get_precoder_mrc,
        calc_sum_rate_func=calc_sum_rate_no_iui,
    )
