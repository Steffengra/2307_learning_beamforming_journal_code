
from pathlib import Path
import gzip
import pickle

import numpy as np
from keras.models import load_model

import src
from src.analysis.helpers.test_precoder_error_sweep import (
    test_precoder_error_sweep,
)
from src.data.calc_sum_rate import (
    calc_sum_rate,
)
from src.models.precoders.learned_precoder import get_learned_precoder_decentralized_normalized


def test_sac_precoder_decentralized_error_sweep(
        config: 'src.config.config.Config',
        model_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
) -> None:
    """Test decentralized learned SAC precoders for a range of error configuration with monte carlo average."""

    def get_precoder_function_learned(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
    ):

        states = config.config_learner.get_state(
            satellite_manager=satellite_manager,
            norm_factors=norm_dict['norm_factors'],
            **config.config_learner.get_state_args,
            per_sat=True
        )

        w_precoder_normalized = get_learned_precoder_decentralized_normalized(
            states=states,
            precoder_networks=precoder_networks,
            **config.learned_precoder_args,
        )

        return w_precoder_normalized

    precoder_paths = sorted([
        Path(path, 'model')
        for path in model_path.iterdir()
        if path.is_dir() and 'agent' in path.name
    ])
    precoder_networks = [load_model(model_path) for model_path in precoder_paths]

    with gzip.open(Path(model_path, 'config', 'norm_dict.gzip')) as file:
        norm_dict = pickle.load(file)
    norm_factors = norm_dict['norm_factors']

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='sac_decentralized',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=get_precoder_function_learned,
        calc_sum_rate_func=calc_sum_rate,
    )