
from pathlib import Path
from sys import path as sys_path
project_root_path = Path(Path(__file__).parent, '..', '..')
sys_path.append(str(project_root_path.resolve()))

import optuna
import numpy as np

from src.config.config import Config
from src.models.train_sac import train_sac


def objective(trial):

    # Set up config
    config = Config()
    config.show_plots = False
    config.verbosity = 0

    # Suggest new values for this trial
    lr_critic = trial.suggest_float(name='lr_critic', low=1e-6, high=1e-4)
    lr_actor = trial.suggest_float(name='lr_actor', low=1e-7, high=1e-4)

    # config.config_learner.training_episodes = int(0.1 * config.config_learner.training_episodes)  # % of training budget
    config.config_learner.training_episodes = 10

    # Apply values to config
    config.config_learner.network_args['policy_network_optimizer_args']['learning_rate'] = lr_actor
    config.config_learner.network_args['value_network_optimizer_args']['learning_rate'] = lr_critic

    # execute trial
    _, metrics = train_sac(
        config=config,
        optuna_trial=trial,
    )

    window_length = 10
    window_low = max(len(metrics['mean_sum_rate_per_episode'])-window_length, 0)

    return np.nanmean(metrics['mean_sum_rate_per_episode'][window_low:])


if __name__ == '__main__':
    # steps = episodes
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(),  # default: TPESampler
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=2,  # don't prune until n number of trials
            n_warmup_steps=1,  # don't prune until n number of steps within a trial
            n_min_trials=1,  # minimum number of trials to prune, ensure at least n trials survive
            interval_steps=10,  # number of steps between pruning checks
        )
    )

    study.optimize(
        func=objective,
        n_trials=2,
        gc_after_trial=True,  # garbage collection
    )
    print(study.best_trial)
    print(study.best_trials)
    # print(study.best_params)