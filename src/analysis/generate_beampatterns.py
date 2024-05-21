
import gzip
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
from src.utils.load_model import load_model

import src
from src.config.config import Config
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.data.calc_sum_rate import calc_sum_rate
from src.data.channel.get_steering_vec import get_steering_vec
from src.data.precoder.mmse_precoder import mmse_precoder_normalized
from src.data.precoder.calc_autocorrelation import calc_autocorrelation
from src.data.precoder.robust_SLNR_precoder import robust_SLNR_precoder_no_norm
from src.models.precoders.learned_precoder import get_learned_precoder_normalized
from src.utils.update_sim import update_sim
from src.utils.progress_printer import progress_printer


def generate_beampatterns(
        angle_sweep_range: np.ndarray,
        num_patterns: int,
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager',
        user_manager: 'src.data.user_manager.UserManager',
        learned_model_paths: dict,
        generate_mmse: bool = True,
        generate_slnr: bool = True,
        generate_ones: bool = True,
) -> None:

    def calc_power_gains(w_precoder):
        power_gains = np.empty((len(user_manager.users), len(angle_sweep_range), len(satellite_manager.satellites)))
        for user_idx in range(len(user_manager.users)):

            # slice precoder matrix for this user only
            w_precoder_user = w_precoder[:, user_idx]

            for satellite_idx, satellite in enumerate(satellite_manager.satellites):

                # slice precoder matrix for this user and satellite only
                w_precoder_satellite_user = w_precoder_user[satellite_idx*satellite.antenna_nr:satellite_idx*satellite.antenna_nr+satellite.antenna_nr]

                for angle_id, _ in enumerate(angle_sweep_range):

                    power_gains[user_idx, angle_id, satellite_idx] = abs(
                        np.matmul(
                            steering_vectors_to_users[user_idx, angle_id, satellite_idx, :],
                            w_precoder_satellite_user
                        )
                    ) ** 2

        return power_gains

    def save_results():
        name = 'beam_patterns'
        results_path = Path(config.output_metrics_path, config.config_learner.training_name, 'beam_patterns')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(Path(results_path, f'{name}.gzip'), 'wb') as file:
            pickle.dump([angle_sweep_range, data], file=file)

    # todo: hardcoded single satellite
    # steering_vectors_to_users = np.zeros((len(user_manager.users), len(angle_sweep_range), satellite_manager.satellites[0].antenna_nr), dtype='complex')
    steering_vectors_to_users = np.zeros((
        len(user_manager.users),
        len(angle_sweep_range),
        len(satellite_manager.satellites),
        satellite_manager.satellites[0].antenna_nr
    ), dtype='complex')  # todo: only works if all satellites have the same antenna nr as sat[0]
    for user in user_manager.users:
        for satellite in satellite_manager.satellites:
            for angle_id, angle in enumerate(angle_sweep_range):
                steering_vectors_to_users[user.idx, angle_id, satellite.idx, :] = get_steering_vec(satellite=satellite,
                                                                                                   phase_aod_steering=np.cos(angle))

    learned_models = {}
    for model_name, model_path in learned_model_paths.items():

        learned_models[model_name] = {}
        (
            learned_models[model_name]['model'],
            learned_models[model_name]['norm_dict']
        ) = load_model(model_path)

        if learned_models[model_name]['norm_dict'] != {}:  # todo
            config.config_learner.get_state_args['norm_state'] = True
        else:
            config.config_learner.get_state_args['norm_state'] = False

    real_time_start = datetime.now()

    data = []
    for iter_id in range(num_patterns):

        iter_data = {}

        update_sim(config=config, satellite_manager=satellite_manager, user_manager=user_manager)

        # iter_data['estimation_errors'] = satellite_manager.satellites[0].estimation_errors.copy()  # todo hardcoded single sat
        iter_data['estimation_errors'] = [satellite.estimation_errors.copy()
                                          for satellite in satellite_manager.satellites]
        # iter_data['user_positions'] = [satellite_manager.satellites[0].aods_to_users[user_idx].copy() for user_idx in range(len(user_manager.users))]  # todo hardcoded single sat
        iter_data['user_positions'] = [
            [satellite.aods_to_users[user_idx].copy()
             for user_idx in range(len(user_manager.users))]
            for satellite in satellite_manager.satellites
        ]

        for learned_model in learned_models:
            state = config.config_learner.get_state(
                satellite_manager=satellite_manager,
                norm_factors=learned_models[learned_model]['norm_dict'],
                **config.config_learner.get_state_args
            )

            w_learned = get_learned_precoder_normalized(
                state=state,
                precoder_network=learned_models[learned_model]['model'],
                **config.learned_precoder_args,
            )

            sum_rate_learned = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_learned,
                noise_power_watt=config.noise_power_watt,
            )

            power_gains = calc_power_gains(w_learned)

            iter_data[learned_model] = {
                'sum_rate': sum_rate_learned,
                'power_gains': power_gains,
            }

        if generate_mmse:
            w_mmse = mmse_precoder_normalized(
                channel_matrix=satellite_manager.erroneous_channel_state_information,
                **config.mmse_args,
            )

            sum_rate_mmse = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_mmse,
                noise_power_watt=config.noise_power_watt,
            )

            power_gains = calc_power_gains(w_mmse)

            iter_data['mmse'] = {
                'sum_rate': sum_rate_mmse,
                'power_gains': power_gains,
            }

        if generate_slnr:

            autocorrelation=calc_autocorrelation(
                satellite=satellite_manager.satellites[0],
                error_model_config=config.config_error_model,
                error_distribution='uniform',
            )

            w_slnr = robust_SLNR_precoder_no_norm(
                channel_matrix=satellite_manager.erroneous_channel_state_information,
                autocorrelation_matrix=autocorrelation,
                noise_power_watt=config.noise_power_watt,
                power_constraint_watt=config.power_constraint_watt,
            )

            sum_rate_slnr = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_slnr,
                noise_power_watt=config.noise_power_watt,
            )

            power_gains = calc_power_gains(w_slnr)

            iter_data['slnr'] = {
                'sum_rate': sum_rate_slnr,
                'power_gains': power_gains,
            }

        if generate_ones:

            w_ones = np.ones(w_mmse.shape)

            sum_rate_ones = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_ones,
                noise_power_watt=config.noise_power_watt,
            )

            power_gains = calc_power_gains(w_ones)

            iter_data['ones'] = {
                'sum_rate': sum_rate_ones,
                'power_gains': power_gains,
            }

        data.append(iter_data)

        if iter_id % 10 == 0:
            progress_printer(progress=(iter_id+1)/num_patterns, real_time_start=real_time_start)

    save_results()


if __name__ == '__main__':

    angle_sweep_range = np.arange(1.2, 1.9, 0.1 * np.pi / 180)
    num_patterns = 10
    generate_mmse = True
    generate_slnr = False
    generate_ones = True

    config = Config()

    # TODO: UPDATE THIS
    raise Exception('UPDATE THIS')
    config.config_learner.training_name = config.generate_name_from_config()

    satellite_manager = SatelliteManager(config)
    user_manager = UserManager(config)

    # todo: models currently must have the same get_state config
    model_paths = {
        # 'learned_0.0_error':
        #     Path(
        #         config.trained_models_path,
        #         '1_sat_16_ant_3_usr_100000_dist_0.0_error_on_cos_0.1_fading',
        #         'single_error',
        #         'userwiggle_50000_snap_4.565',
        #         'model',
        #     ),
        # 'learned_0.5_error':
        #     Path(
        #         config.trained_models_path,
        #         '1_sat_16_ant_3_usr_100000_dist_0.05_error_on_cos_0.1_fading',
        #         'single_error',
        #         'userwiggle_50000_snap_2.710',
        #         'model',
        #     ),
        'test':
            Path(
                config.trained_models_path,
                '1sat_16ant_100k~0_3usr_100k_50k_additive_0.0',
                'base',
                'full_snap_4.553',
                'model',
            ),
    }

    with tf.device('CPU:0'):
        generate_beampatterns(
            angle_sweep_range=angle_sweep_range,
            num_patterns=num_patterns,
            config=config,
            satellite_manager=satellite_manager,
            user_manager=user_manager,
            learned_model_paths=model_paths,
            generate_mmse=generate_mmse,
            generate_slnr=generate_slnr,
            generate_ones=generate_ones,
        )
