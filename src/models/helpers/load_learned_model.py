
import gzip
import pickle
from pathlib import Path

import numpy as np
from keras.models import load_model

from src.models.helpers.bound_action import bound_actions


def load_learned_model(
        model_path: Path,
        action_bound_mode: str or None,
        get_state_method,
        get_state_args,
):

    model = load_model(Path(model_path, 'model'))

    with gzip.open(Path(model_path, 'config', 'norm_dict.gzip')) as file:
        norm_dict = pickle.load(file)
    norm_factors = norm_dict['norm_factors']

    def get_state(satellite_manager):
        return get_state_method(satellite_manager=satellite_manager, norm_factors=norm_factors, **get_state_args)[np.newaxis]

    def infer(state):
        actions = model(state)[0].numpy().flatten()
        bounded_action = bound_actions(actions, mode=action_bound_mode)

        return bounded_action

    return get_state, infer
