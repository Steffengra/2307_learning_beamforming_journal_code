
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors

import src
from src.data.channel.get_steering_vec import get_steering_vec
from src.config.config_plotting import generic_styling


def plot_beampattern(
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        user_manager: 'src.data.user_manager.UserManager',
        w_precoder: np.ndarray,
        position_sweep_range: np.ndarray or None = None,
        plot_title: str or None = None,
) -> None:
    """Plots beam power toward each user from the point of view of a satellite for a given precoding w_precoder."""

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
            (user_manager.users[-1].spherical_coordinates[2] - user_manager.users[0].spherical_coordinates[2]) / 100
        )

    # calculate power gains
    directional_power_gains = np.zeros((len(satellite_manager.satellites), len(user_manager.users), len(position_sweep_range)))
    for user in user_manager.users:
        for position_id, position in enumerate(position_sweep_range):

            user.update_position(
                [user.spherical_coordinates[0], user.spherical_coordinates[1], position]
            )
            satellite_manager.calculate_satellite_distances_to_users(users=user_manager.users)
            satellite_manager.calculate_satellite_aods_to_users(users=user_manager.users)

            for satellite in satellite_manager.satellites:

                steering_vector_to_user = get_steering_vec(satellite=satellite, phase_aod_steering=np.cos(satellite.aods_to_users[user.idx]))

                directional_power_gain = abs(np.matmul(steering_vector_to_user, w_precoder[satellite.idx:satellite.idx + satellite.antenna_nr, user.idx])) ** 2

                directional_power_gains[satellite.idx, user.idx, position_id] = directional_power_gain

    sum_directional_power_gains = np.sum(directional_power_gains, axis=0)

    ax.plot(position_sweep_range, sum_directional_power_gains.T)

    ax.legend()
    ax.set_xlabel('User AOD from satellite')
    ax.set_ylabel('Directional Power Gain')
    if plot_title is not None:
        ax.set_title(plot_title)

    generic_styling(ax=ax)

    # reset original coordinates
    for user in user_manager.users:
        user.update_position(original_pos[user.idx])
    satellite_manager.calculate_satellite_distances_to_users(users=user_manager.users)
    satellite_manager.calculate_satellite_aods_to_users(users=user_manager.users)
