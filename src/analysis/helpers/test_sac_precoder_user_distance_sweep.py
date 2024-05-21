
from pathlib import Path

import numpy as np

import src
from src.analysis.helpers.test_precoder_user_distance_sweep import (
    test_precoder_user_distance_sweep,
)
from src.data.calc_sum_rate import (
    calc_sum_rate,
)
from src.models.precoders.learned_precoder import get_learned_precoder_normalized
from src.utils.load_model import load_model


def test_sac_precoder_user_distance_sweep(
    config: 'src.config.config.Config',
    distance_sweep_range: np.ndarray,
    model_path: Path,
) -> None:
    """Test a precoder over a range of distances with zero error."""

    def get_precoder_function_learned(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
    ):

        state = config.config_learner.get_state(
            satellite_manager=satellite_manager,
            norm_factors=norm_factors,
            **config.config_learner.get_state_args
        )

        w_precoder_normalized = get_learned_precoder_normalized(
            state=state,
            precoder_network=precoder_network,
            **config.learned_precoder_args,
        )

        return w_precoder_normalized

    precoder_network, norm_factors = load_model(model_path)

    if norm_factors != {}:
        config.config_learner.get_state_args['norm_state'] = True

    test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='learned',
        get_precoder_func=get_precoder_function_learned,
        calc_sum_rate_func=calc_sum_rate,
    )
