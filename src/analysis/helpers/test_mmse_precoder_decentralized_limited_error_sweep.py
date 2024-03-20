
import numpy as np

import src
from src.analysis.helpers.test_precoder_error_sweep import (
    test_precoder_error_sweep,
)
from src.data.precoder.mmse_precoder_decentral import mmse_precoder_decentral_limited_normalized
from src.data.calc_sum_rate import (
    calc_sum_rate,
)


def test_mmse_precoder_decentralized_limited_error_sweep(
    config: 'src.config.config.Config',
    error_sweep_parameter: str,
    error_sweep_range: np.ndarray,
    monte_carlo_iterations: int,
) -> None:
    """Test the MMSE precoder for a range of error configuration with monte carlo average."""

    def get_precoder_decentralized_mmse(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
    ) -> np.ndarray:

        local_channel_matrices = [satellite_manager.get_local_channel_state_information(sat_idx, config.local_csi_own_quality, config.local_csi_others_quality)
                                  for sat_idx in range(config.sat_nr)]

        w_mmse_normalized = mmse_precoder_decentral_limited_normalized(
            local_channel_matrices=local_channel_matrices,
            **config.mmse_args,
        )

        return w_mmse_normalized

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='mmse_decentralized_limited',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=get_precoder_decentralized_mmse,
        calc_sum_rate_func=calc_sum_rate,
    )
