
from pathlib import Path
from sys import path as sys_path
project_root_path = Path(Path(__file__).parent, '..', '..')
sys_path.append(str(project_root_path.resolve()))

from datetime import datetime
import gzip
import pickle
from shutil import (
    copytree,
    rmtree,
)

import numpy as np
import optuna

import src
from src.config.config import (
    Config,
)
from src.data.satellite_manager import (
    SatelliteManager,
)
from src.data.user_manager import (
    UserManager,
)
from src.models.algorithms.soft_actor_critic import (
    SoftActorCritic,
)
from src.models.helpers.get_state_norm_factors import (
    get_state_norm_factors,
)
from src.data.calc_sum_rate import (
    calc_sum_rate,
)
from src.utils.real_complex_vector_reshaping import (
    real_vector_to_half_complex_vector,
    complex_vector_to_double_real_vector,
    rad_and_phase_to_complex_vector,
)
from src.utils.norm_precoder import (
    norm_precoder,
)
from src.utils.profiling import (
    start_profiling,
    end_profiling,
)
from src.utils.progress_printer import (
    progress_printer,
)
from src.utils.update_sim import (
    update_sim,
)


def train_sac_decentralized_blind(
        config: 'src.config.config.Config',
        optuna_trial: optuna.Trial or None = None,
) -> Path:
    """Train one Soft Actor Critic precoder per satellite according to the config."""

    def progress_print(to_log: bool = False) -> None:
        progress = (
                (training_episode_id * config.config_learner.training_steps_per_episode + training_step_id + 1)
                / (config.config_learner.training_episodes * config.config_learner.training_steps_per_episode)
        )
        if not to_log:
            progress_printer(progress=progress, real_time_start=real_time_start)
        else:
            progress_printer(progress=progress, real_time_start=real_time_start, logger=logger)

    def save_model_checkpoints(extra):

        name = f''
        if extra is not None:
            name += f'snap_{extra:.3f}'

        checkpoint_path = Path(
            config.trained_models_path,
            config.config_learner.training_name,
            'decentralized_blind',
            name,
        )

        for sac_id, sac in enumerate(sacs):
            sac_path = Path(checkpoint_path, f'agent_{sac_id}')
            sac.networks['policy'][0]['primary'].save(Path(sac_path, 'model'))

        logger.info(f'Saved model checkpoint at mean reward {extra:.3f}')

        # save config
        copytree(Path(config.project_root_path, 'src', 'config'),
                 Path(checkpoint_path, 'config'),
                 dirs_exist_ok=True)

        # save norm dict
        with gzip.open(Path(checkpoint_path, 'config', 'norm_dict.gzip'), 'wb') as file:
            pickle.dump(norm_dict, file)

        # clean model checkpoints
        for high_score_prior_id, high_score_prior in enumerate(reversed(high_scores)):
            if high_score > 1.05 * high_score_prior or high_score_prior_id > 3:

                name = f'snap_{high_score_prior:.3f}'

                prior_checkpoint_path = Path(
                    config.trained_models_path,
                    config.config_learner.training_name,
                    'decentralized_blind',
                    name
                )
                rmtree(path=prior_checkpoint_path, ignore_errors=True)
                high_scores.remove(high_score_prior)

        return checkpoint_path

    def save_results():

        name = f'training_error_decentralized_blind.gzip'

        results_path = Path(config.output_metrics_path, config.config_learner.training_name, 'decentralized_blind')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(Path(results_path, name), 'wb') as file:
            pickle.dump(metrics, file=file)

    logger = config.logger.getChild(__name__)

    config.config_learner.algorithm_args['network_args']['num_actions'] = 2 * config.sat_ant_nr * config.user_nr
    config.config_learner.network_args['size_state'] = int(config.config_learner.network_args['size_state'] / config.sat_nr)

    satellite_manager = SatelliteManager(config=config)
    user_manager = UserManager(config=config)
    sacs = [
        SoftActorCritic(rng=config.rng, **config.config_learner.algorithm_args)
        for _ in range(config.sat_nr)
    ]

    norm_dict = get_state_norm_factors(config=config, satellite_manager=satellite_manager, user_manager=user_manager, per_sat=True)
    logger.info('State normalization factors found')
    logger.info(norm_dict)

    metrics: dict = {
        'mean_sum_rate_per_episode': -np.infty * np.ones(config.config_learner.training_episodes)
    }
    high_score = -np.infty
    high_scores = []

    real_time_start = datetime.now()

    profiler = None
    if config.profile:
        profiler = start_profiling()

    step_experiences = [{'state': 0, 'action': 0, 'reward': 0, 'next_state': 0} for _ in range(config.sat_nr)]

    for training_episode_id in range(config.config_learner.training_episodes):

        episode_metrics: dict = {
            'sum_rate_per_step': -np.infty * np.ones(config.config_learner.training_steps_per_episode),
            'mean_log_prob_density': np.infty * np.ones(config.config_learner.training_steps_per_episode),
            'value_loss': -np.infty * np.ones(config.config_learner.training_steps_per_episode),
        }

        update_sim(config, satellite_manager, user_manager)  # reset for new episode
        states_next = config.config_learner.get_state(
            satellite_manager=satellite_manager,
            norm_factors=norm_dict['norm_factors'],
            **config.config_learner.get_state_args,
            per_sat=True
        )

        for training_step_id in range(config.config_learner.training_steps_per_episode):

            simulation_step = training_episode_id * config.config_learner.training_steps_per_episode + training_step_id

            # determine state
            states_current = states_next
            for sat_id in range(config.sat_nr):
                step_experiences[sat_id]['state'] = states_current[sat_id]

            # determine action based on state
            actions = [sac.get_action(state=states_current[sac_id]) for sac_id, sac in enumerate(sacs)]
            for sat_id in range(config.sat_nr):
                step_experiences[sat_id]['action'] = actions[sat_id]

            # reshape and stack each satellites individual precoder
            w_precoder = np.zeros((config.sat_nr*config.sat_ant_nr, config.user_nr), dtype='complex128')
            for sat_id in range(config.sat_nr):
                w_precoder_sat = real_vector_to_half_complex_vector(actions[sat_id])
                w_precoder_sat = w_precoder_sat.reshape(config.sat_ant_nr, config.user_nr)
                w_precoder[sat_id*config.sat_ant_nr:sat_id*config.sat_nr+config.sat_ant_nr, :] = w_precoder_sat

            w_precoder_normed = norm_precoder(precoding_matrix=w_precoder, power_constraint_watt=config.power_constraint_watt,
                                              per_satellite=True, sat_nr=config.sat_nr, sat_ant_nr=config.sat_ant_nr)

            # step simulation based on action, determine reward
            reward = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_precoder_normed,
                noise_power_watt=config.noise_power_watt,
            )
            for sat_id in range(config.sat_nr):
                step_experiences[sat_id]['reward'] = reward

            # update simulation state
            update_sim(config, satellite_manager, user_manager)

            # get new state
            states_next = config.config_learner.get_state(
                satellite_manager=satellite_manager,
                norm_factors=norm_dict['norm_factors'],
                **config.config_learner.get_state_args,
                per_sat=True
            )
            for sat_id in range(config.sat_nr):
                step_experiences[sat_id]['next_state'] = states_next[sat_id]

            for sac_id, sac in enumerate(sacs):
                sac.add_experience(experience=step_experiences[sac_id])

            # train allocator off-policy
            train_policy = config.config_learner.policy_training_criterion(simulation_step=simulation_step)
            train_value = config.config_learner.value_training_criterion(simulation_step=simulation_step)

            for sac in sacs:
                if train_value or train_policy:
                    mean_log_prob_density, value_loss = sac.train(
                        toggle_train_value_networks=train_value,
                        toggle_train_policy_network=train_policy,
                        toggle_train_entropy_scale_alpha=True,
                    )
                else:
                    mean_log_prob_density = np.nan
                    value_loss = np.nan

            # log results
            episode_metrics['sum_rate_per_step'][training_step_id] = reward
            episode_metrics['mean_log_prob_density'][training_step_id] = mean_log_prob_density  # todo: currently only logs last sac
            episode_metrics['value_loss'][training_step_id] = value_loss  # todo: currently only logs last sac

            if config.verbosity > 0:
                if training_step_id % 50 == 0:
                    progress_print()

        # log episode results
        episode_mean_sum_rate = np.nanmean(episode_metrics['sum_rate_per_step'])
        metrics['mean_sum_rate_per_episode'][training_episode_id] = episode_mean_sum_rate

        # If doing optuna optimization: check trial results, stop early if bad
        if optuna_trial:
            window = 10
            lower_end = max(training_episode_id-window, 0)
            episode_result = np.nanmean(metrics['mean_sum_rate_per_episode'][lower_end:training_episode_id+1])

            optuna_trial.report(episode_result, training_episode_id)
            if optuna_trial.should_prune():
                raise optuna.TrialPruned()

        if config.verbosity > 0:
            print('\r', end='')  # clear console for logging results
        progress_print(to_log=True)
        logger.info(
            f'Episode {training_episode_id}:'
            f' Episode mean reward: {episode_mean_sum_rate:.4f}'
            f' std {np.nanstd(episode_metrics["sum_rate_per_step"]):.2f},'
            f' current exploration: {np.nanmean(episode_metrics["mean_log_prob_density"]):.2f},'
            f' value loss: {np.nanmean(episode_metrics["value_loss"]):.5f}'
            # f' curr. lr: {sac.networks["policy"][0]["primary"].optimizer.learning_rate(sac.networks["policy"][0]["primary"].optimizer.iterations):.2E}'
        )

        # save network snapshot
        if episode_mean_sum_rate > high_score:
            high_score = episode_mean_sum_rate.copy()
            high_scores.append(high_score)
            best_model_path = save_model_checkpoints(extra=episode_mean_sum_rate)

    # end compute performance profiling
    if profiler is not None:
        end_profiling(profiler)

    save_results()

    return best_model_path, metrics


if __name__ == '__main__':
    cfg = Config()
    train_sac_decentralized_blind(config=cfg)
