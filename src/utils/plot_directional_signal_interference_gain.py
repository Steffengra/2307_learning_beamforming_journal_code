
import numpy as np
import matplotlib.pyplot as plt

import src
from src.data.channel.get_steering_vec import get_steering_vec
from src.config.config_plotting import generic_styling


def plot_directional_signal_interference_gain(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        user_manager: 'src.data.user_manager.UserManager',
        w_precoder: np.ndarray,
        position_sweep_range: np.ndarray or None = None,
        log_scale: bool = False,
        plot_title: str or None = None,
) -> None:
    """
    Plots the directional signal/interference gain (no path loss) from satellites to user
    positions.
    """

    # save original positions
    original_pos = [
        user.spherical_coordinates
        for user in user_manager.users
    ]

    # create a figure
    fig, ax = plt.subplots()

    # mark user positions
    for user in user_manager.users:
        ax.scatter(
            user.spherical_coordinates[2],
            0,
            # color='black',
            label=f'user {user.idx}'
        )
        ax.axvline(
            user.spherical_coordinates[2],
            color='black',
            linestyle='dashed'
        )

    # calculate auto x axis scaling
    if position_sweep_range is None:
        max_dist = user_manager.users[-1].spherical_coordinates[2] - user_manager.users[0].spherical_coordinates[2]

        position_sweep_range = np.arange(
            user_manager.users[0].spherical_coordinates[2] - 0.5 * max_dist,
            user_manager.users[-1].spherical_coordinates[2] + 0.5 * max_dist,
            (user_manager.users[-1].spherical_coordinates[2] - user_manager.users[0].spherical_coordinates[2]) / 10000
        )

    # calculate power gains
    directional_power_gains = np.zeros((len(satellite_manager.satellites), len(user_manager.users), len(position_sweep_range)), dtype='complex128')
    for position_id, position in enumerate(position_sweep_range):

        user_manager.users[0].update_position(
            [user_manager.users[0].spherical_coordinates[0], user_manager.users[0].spherical_coordinates[1], position]
        )
        satellite_manager.calculate_satellite_distances_to_users(users=user_manager.users)
        satellite_manager.calculate_satellite_aods_to_users(users=user_manager.users)
        satellite_manager.update_channel_state_information(channel_model=config.channel_model,
                                                           user_manager=user_manager)

        for satellite in satellite_manager.satellites:

            for user in user_manager.users:

                directional_power_gain = np.matmul(satellite.channel_state_to_users[0], w_precoder[satellite.idx:satellite.idx + satellite.antenna_nr, user.idx])

                directional_power_gains[satellite.idx, user.idx, position_id] = directional_power_gain

    sum_directional_power_gains = np.sum(directional_power_gains, axis=0)
    sum_directional_power_gains = abs(sum_directional_power_gains) ** 2

    signal_to_interference_ratio_per_user = np.zeros((len(user_manager.users), len(position_sweep_range)))

    for user_id in range(len(user_manager.users)):
        signal_to_interference_ratio_per_user[user_id, :] = (
            sum_directional_power_gains[user_id, :] / np.sum(np.delete(sum_directional_power_gains, user_id, axis=0), axis=0)
        )

    signal_to_interference_ratio = np.sum(signal_to_interference_ratio_per_user, axis=0)
    if log_scale:
        signal_to_interference_ratio = np.log2(signal_to_interference_ratio)
        signal_to_interference_ratio_per_user = np.log2(signal_to_interference_ratio_per_user)

    # ax.plot(position_sweep_range, signal_to_interference_ratio)
    ax.plot(position_sweep_range, signal_to_interference_ratio_per_user.T)

    # ax.legend()
    ax.set_xlabel('User Position')
    ax.set_ylabel('ld S/I Directional Gain' if log_scale else 'S/I Directional Gain')
    if plot_title is not None:
        ax.set_title(plot_title)
    generic_styling(ax=ax)

    # reset original coordinates
    for user in user_manager.users:
        user.update_position(original_pos[user.idx])
        satellite_manager.calculate_satellite_distances_to_users(users=user_manager.users)
        satellite_manager.calculate_satellite_aods_to_users(users=user_manager.users)
        satellite_manager.update_channel_state_information(channel_model=config.channel_model,
                                                           user_manager=user_manager)
