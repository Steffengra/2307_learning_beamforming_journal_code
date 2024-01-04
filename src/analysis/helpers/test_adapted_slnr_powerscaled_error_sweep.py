
from pathlib import Path
import gzip
import pickle

import numpy as np
from keras.models import load_model

import src
from src.analysis.helpers.test_precoder_error_sweep import test_precoder_error_sweep
from src.data.calc_sum_rate import calc_sum_rate
from src.models.precoders.scaled_precoder import scale_robust_slnr_complete_precoder_normed


def test_adapted_slnr_powerscaled_error_sweep(
        config: 'src.config.config.Config',
        model_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
) -> None:

    def get_precoder_function_adapted_powerscaled(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
    ) -> None:

        scaler_input_state = config.config_learner.get_state(
            satellite_manager,
            norm_factors=norm_factors,
            **config.config_learner.get_state_args
        )

        precoding = scale_robust_slnr_complete_precoder_normed(
            satellite=satellite_manager.satellites[0],
            error_model_config=config.config_error_model,
            error_distribution='uniform',
            channel_matrix=satellite_manager.erroneous_channel_state_information,
            noise_power_watt=config.noise_power_watt,
            power_constraint_watt=config.power_constraint_watt,
            scaling_network=scaling_network,
            scaler_input_state=scaler_input_state,
            sat_nr=config.sat_nr,
            sat_ant_nr=config.sat_ant_nr,
        )

        return precoding

    scaling_network = load_model(model_path)

    with gzip.open(Path(model_path, '..', 'config', 'norm_dict.gzip')) as file:
        norm_dict = pickle.load(file)
    norm_factors = norm_dict['norm_factors']

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='adapted_slnr_complete',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=get_precoder_function_adapted_powerscaled,
        calc_sum_rate_func=calc_sum_rate,
    )
