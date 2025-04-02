
import numpy as np

import src


def user_activity_selection(
        config: 'src.config.config.Config',
        user_manager: 'src.data.user_manager.UserManager',
) -> None:

    if config.user_activity_selection == 'random_uniform_centered':
        num_active_users = config.rng.integers(1, config.user_nr+1)  # exclusive interval [low, high)
        user_mask = np.concatenate([np.ones(num_active_users), np.zeros(config.user_nr - num_active_users)])
        user_mask = np.roll(user_mask, int(config.user_nr/2)-int(num_active_users/2))  # move users to center to keep block with same distances

    elif config.user_activity_selection == 'all_active':
        user_mask = np.ones(shape=len(user_manager.users))

    elif config.user_activity_selection == 'keep_as_is':
        return

    else:
        raise ValueError(f'Unknown user_activity_selection mode {config.user_activity_selection}')

    user_manager.set_active_users(user_mask=user_mask)

