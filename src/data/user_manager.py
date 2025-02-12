
import numpy as np

import src
from src.data.user import (
    User,
)


class UserManager:
    """A UserManager object holds user objects and gives functions to interface with all users."""

    def __init__(
            self,
            config: 'src.config.config.Config',
    ) -> None:

        self.rng = config.rng
        self.logger = config.logger.getChild(__name__)

        self.users: list[src.data.user.User] = []
        self._initialize_users(config=config)

        self.user_mask: np.ndarray = None
        self.active_user_idx: np.ndarray = None
        self.set_active_users(np.ones(len(self.users)))

        self.logger.info('user setup complete')

    def calc_spherical_coordinates(
            self,
            config: 'src.config.config.Config',
    ) -> (np.ndarray, list):
        """TODO: Comment"""

        # calculate average user positions
        user_pos_average = (np.arange(0, config.user_nr, dtype='float128') - (config.user_nr - 1) / 2) * config.user_dist_average

        # add random value on user distances
        random_factor = self.rng.uniform(low=-config.user_dist_bound,
                                         high=config.user_dist_bound,
                                         size=config.user_nr)
        # random_factor = self.rng.choice([-config.user_dist_bound, 0, config.user_dist_bound])
        user_dist = user_pos_average + random_factor

        # calculate user_aods_diff_earth_rad
        user_aods_diff_earth_rad = np.zeros(config.user_nr)

        for user_idx in range(config.user_nr):

            if user_dist[user_idx] < 0:
                user_aods_diff_earth_rad[user_idx] = -1 * np.arccos(1 - 0.5 * (user_dist[user_idx] / config.radius_earth)**2)
            elif user_dist[user_idx] >= 0:
                user_aods_diff_earth_rad[user_idx] = np.arccos(1 - 0.5 * (user_dist[user_idx] / config.radius_earth)**2)

        user_center_aod_earth_rad = config.user_center_aod_earth_deg * np.pi / 180

        # TODO: if any(user_pos_average == 0) == 1, vllt Fallunterscheidung für gerade und ungerade

        # calculate user_aods_earth_rad
        user_aods_earth_rad = user_center_aod_earth_rad + user_aods_diff_earth_rad

        # create user objects
        user_radii = config.radius_earth * np.ones(config.user_nr)
        user_inclinations = np.pi / 2 * np.ones(config.user_nr)

        user_spherical_coordinates = np.array([user_radii, user_inclinations, user_aods_earth_rad])

        return user_spherical_coordinates

    def _initialize_users(
            self,
            config: 'src.config.config.Config',
    ) -> None:
        """TODO: Comment"""

        user_spherical_coordinates = np.flip(
            self.calc_spherical_coordinates(config=config),
            axis=1,
        )

        for user_idx in range(config.user_nr):
            self.users.append(
                User(
                    idx=user_idx,
                    spherical_coordinates=user_spherical_coordinates[:, user_idx],
                    **config.user_args,
                )
            )

    def set_active_users(
            self,
            user_mask: np.ndarray,
    ) -> None:
        """
        Set enabled status of users to True or False.
        enabled status can be used in csi calculation to disable users.
        """

        self.user_mask = user_mask
        self.active_user_idx = np.where(user_mask == 1)[0]

        for user_id, user in enumerate(self.users):
            if user_mask[user_id] == 0:
                user.enabled = False
            elif user_mask[user_id] == 1:
                user.enabled = True
            else:
                raise ValueError(f'user_mask must be 0 or 1, is {user_mask[user_id]}')


    def update_positions(
            self,
            config: 'src.config.config.Config',
    ) -> None:
        """TODO: Comment"""

        user_spherical_coordinates = np.flip(
            self.calc_spherical_coordinates(config=config),
            axis=1,
        )

        for user in self.users:
            user.update_position(
                spherical_coordinates=user_spherical_coordinates[:, user.idx],
            )
