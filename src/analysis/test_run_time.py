
from pathlib import Path
from sys import path as sys_path

project_root_path = Path(Path(__file__).parent, '..', '..')
sys_path.append(str(project_root_path.resolve()))

import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from src.config.config import Config
from src.data.user_manager import UserManager
from src.data.satellite_manager import SatelliteManager
from src.utils.update_sim import update_sim
from src.utils.profiling import start_profiling, end_profiling
from src.utils.progress_printer import progress_printer

from src.data.precoder.mmse_precoder import mmse_precoder_no_norm
from src.data.precoder.robust_SLNR_precoder import robust_SLNR_precoder_no_norm
from src.data.precoder.calc_autocorrelation import calc_autocorrelation
from src.utils.load_model import (
    load_model,
    load_models,
)

config = Config()
sat_man = SatelliteManager(config)
usr_man = UserManager(config)

num_iter = 100_000
model_path = Path(config.trained_models_path, '1sat_16ant_100k~0_3usr_10k~5k_additive_0.0', 'base', 'full_snap_3.948')

update_sim(config, sat_man, usr_man)






# profiler = start_profiling()
# for _ in range(num_iter):
#     autocorrelation = calc_autocorrelation(
#         satellite=sat_man.satellites[0],
#         error_model_config=config.config_error_model,
#         error_distribution='uniform',
#     )
#
#     w_robust_slnr = robust_SLNR_precoder_no_norm(
#         channel_matrix=sat_man.erroneous_channel_state_information,
#         autocorrelation_matrix=autocorrelation,
#         noise_power_watt=config.noise_power_watt,
#         power_constraint_watt=config.power_constraint_watt,
#     )
# end_profiling(profiler)






def infer_learned():
    state = config.config_learner.get_state(
            config=config,
            user_manager=usr_man,
            satellite_manager=sat_man,
            norm_factors=norm_factors,
            **config.config_learner.get_state_args
        )
    interpreter.set_tensor(input_details[0]['index'], state[np.newaxis].astype('float32'))
    interpreter.invoke()
    interpreter.get_tensor(output_details[0]['index'])

def infer_mmse():
    mmse_precoder_no_norm(channel_matrix=sat_man.channel_state_information, noise_power_watt=config.noise_power_watt,
                          power_constraint_watt=config.power_constraint_watt)

def infer_slnr():
    autocorrelation = calc_autocorrelation(
        satellite=sat_man.satellites[0],
        error_model_config=config.config_error_model,
        error_distribution='uniform',
    )

    robust_SLNR_precoder_no_norm(
        channel_matrix=sat_man.erroneous_channel_state_information,
        autocorrelation_matrix=autocorrelation,
        noise_power_watt=config.noise_power_watt,
        power_constraint_watt=config.power_constraint_watt,
    )


_, norm_factors = load_model(model_path)
interpreter = tf.lite.Interpreter(str(Path(config.project_root_path, 'src', 'notebooks', '1sat_32ant_6usr_compressed.tflite')))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

infer = infer_learned

real_time_start = datetime.now()
perf_time_start = time.perf_counter()
times = np.zeros(num_iter)
for iter_id in range(num_iter):
    for _ in range(3):  # warumup
        infer()
    start = time.perf_counter()
    infer()
    end = time.perf_counter()
    times[iter_id] = end - start
    update_sim(config, sat_man, usr_man)
    progress_printer(progress=(iter_id+1)/num_iter, real_time_start=real_time_start)

perf_time_stop = time.perf_counter()

print()
print(sum(times))
print(np.mean(times)*10**6)
print(np.median(times)*10**6)
print(np.std(times)*10**6)

print((perf_time_stop - perf_time_start)/(num_iter*4)*10**6)