
import numpy as np

import src
from src.analysis.helpers.test_precoder_error_sweep import (
    test_precoder_error_sweep,
)
from src.data.precoder.mmse_precoder import (
    mmse_precoder_normalized,
    mmse_precoder_no_norm,
)
from src.data.calc_sum_rate import (
    calc_sum_rate,
)
from src.utils.norm_precoder import (
    norm_precoder
)


def test_mmse_precoder_decentralized_blind_error_sweep(
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

        erroneous_csit_per_sat = satellite_manager.get_erroneous_channel_state_information_per_sat()

        w_precoder = np.zeros((config.sat_nr * config.sat_ant_nr, config.user_nr), dtype='complex128')
        for sat_id, sat_csit in enumerate(erroneous_csit_per_sat):

            w_mmse_sat = mmse_precoder_no_norm(
                sat_csit,
                noise_power_watt=config.noise_power_watt,  # todo: teilen durch sat_nr? mutliplizieren weil receiver?
                power_constraint_watt=config.power_constraint_watt / config.sat_nr,
            )

            w_precoder[sat_id*config.sat_ant_nr:sat_id*config.sat_nr+config.sat_ant_nr, :] = w_mmse_sat

        w_precoder = norm_precoder(
            w_precoder,
            power_constraint_watt=config.power_constraint_watt,
            per_satellite=True,
            sat_nr=config.sat_nr,
            sat_ant_nr=config.sat_ant_nr,
        )

        return w_precoder

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='mmse_decentralized_blind',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=get_precoder_decentralized_mmse,
        calc_sum_rate_func=calc_sum_rate,
    )
