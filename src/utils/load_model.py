
from pathlib import Path
import gzip
import pickle

import keras


def load_model(
        model_path: Path,
) -> (keras.Model, dict):

    network = keras.models.load_model(Path(model_path, 'model'))

    with gzip.open(Path(model_path, 'config', 'norm_dict.gzip')) as file:
        norm_dict = pickle.load(file)
    norm_factors = norm_dict['norm_factors']

    return network, norm_factors


def load_models(
        models_path: Path,
) -> (list[keras.Model], dict):

    paths = sorted([
        Path(path, 'model')
        for path in models_path.iterdir()
        if path.is_dir() and 'agent' in path.name
    ])
    networks = [keras.models.load_model(model_path) for model_path in paths]

    with gzip.open(Path(models_path, 'config', 'norm_dict.gzip')) as file:
        norm_dict = pickle.load(file)
    norm_factors = norm_dict['norm_factors']

    return networks, norm_factors