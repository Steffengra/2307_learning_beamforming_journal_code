
from pathlib import Path

import tensorflow as tf

from src.config.config import Config

config = Config()
saved_model_path = Path(config.trained_models_path, '1sat_16ant_100k~0_3usr_100k~50k_additive_0.0', 'base', 'full_snap_4.620', 'model')

converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

with open(Path(config.project_root_path, 'src', 'notebooks', 'model.tflite'), 'wb') as f:
    f.write(tflite_quant_model)