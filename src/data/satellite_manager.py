
import numpy as np

import src
from src.data.satellite import (
    Satellite,
)


class SatelliteManager:
    """
    Satellites holds all satellite objects and helper functions
    """

    def __init__(
            self,
            config: 'src.config.config.Config',
    ) -> None:

        self.rng = config.rng
        self.logger = config.logger.getChild(__name__)

        self.satellites: list[Satellite] = []
        self._initialize_satellites(config=config)

        self.channel_state_information: np.ndarray = np.zeros((config.user_nr, config.sat_ant_nr*config.sat_nr), dtype='complex128')  # nr_user x (nr_antennas * nr_satellites)
                                                                   #   per user: sat 1 ant1, sat 1 ant 2, sat 1 ant 3, sat 2 ant 1, ...
        self.erroneous_channel_state_information: np.ndarray = np.zeros((config.user_nr, config.sat_ant_nr*config.sat_nr), dtype='complex128')  # nr_user x (nr_antennas * nr_satellites)

        self.logger.info('satellites setup complete')

    def calc_spherical_coordinates(
            self,
            config: 'src.config.config.Config',
    ) -> np.ndarray:
        """Todo: doc"""

        # calculate average satellite positions
        sat_pos_average = (np.arange(0, config.sat_nr, dtype='float128') - (config.sat_nr - 1) / 2) * config.sat_dist_average

        # add random value on satellite distances
        random_factor = self.rng.uniform(low=-config.sat_dist_bound * config.sat_dist_average,
                                         high=config.sat_dist_bound * config.sat_dist_average,
                                         size=config.sat_nr)
        sat_dist = sat_pos_average + random_factor

        # calculate sat_aods_diff_earth_rad
        sat_aods_diff_earth_rad = np.zeros(config.sat_nr)

        for sat_idx in range(config.sat_nr):

            if sat_dist[sat_idx] < 0:
                sat_aods_diff_earth_rad[sat_idx] = -1 * np.arccos(1 - 0.5 * (sat_dist[sat_idx] / config.radius_orbit)**2)
            elif sat_dist[sat_idx] >= 0:
                sat_aods_diff_earth_rad[sat_idx] = np.arccos(1 - 0.5 * (sat_dist[sat_idx] / config.radius_orbit)**2)

        # calculate sat_center_aod_earth_rad
        sat_center_aod_earth_rad = config.sat_center_aod_earth_deg * np.pi / 180

        # TODO: if any(sat_pos_average == 0) == 1, vllt Fallunterscheidung für gerade und ungerade

        # calculate sat_aods_earth_rad
        sat_aods_earth_rad = sat_center_aod_earth_rad + sat_aods_diff_earth_rad

        # create satellite objects
        sat_radii = config.radius_orbit * np.ones(config.sat_nr)
        sat_inclinations = np.pi / 2 * np.ones(config.sat_nr)

        sat_spherical_coordinates = np.array([sat_radii, sat_inclinations, sat_aods_earth_rad])

        return sat_spherical_coordinates

    def _initialize_satellites(
            self,
            config: 'src.config.config.Config',
    ) -> None:
        """
        Initializes satellite object list for given configuration
        """

        sat_spherical_coordinates = np.flip(
            self.calc_spherical_coordinates(config=config),
            axis=1,
        )

        for sat_idx in range(config.sat_nr):
            self.satellites.append(
                Satellite(
                    idx=sat_idx,
                    spherical_coordinates=sat_spherical_coordinates[:, sat_idx],
                    **config.satellite_args,
                )
            )

    def update_positions(
            self,
            config: 'src.config.config.Config',
    ) -> None:
        """Todo: doc"""

        sat_spherical_coordinates = np.flip(
            self.calc_spherical_coordinates(config=config),
            axis=1,
        )

        for satellite in self.satellites:
            satellite.update_position(
                spherical_coordinates=sat_spherical_coordinates[:, satellite.idx],
            )

    def calculate_satellite_distances_to_users(
            self,
            users: list,
    ) -> None:
        """
        This function calculates the distances between each satellite and user
        """

        for satellite in self.satellites:
            satellite.calculate_distance_to_users(users=users)

    def calculate_satellite_aods_to_users(
            self,
            users: list,
    ) -> None:
        """
        This function calculates the AODs (angles of departure) from each satellite to
        each user (Earth and satellite orbits are assumed to be circular)
        """

        for satellite in self.satellites:
            satellite.calculate_aods_to_users(users=users)

    def roll_estimation_errors(
            self,
    ) -> None:
        """Todo: doc"""

        for satellite in self.satellites:
            satellite.roll_estimation_errors()

    def update_channel_state_information(
            self,
            channel_model,
            user_manager: 'src.data.user_manager.UserManager'
    ) -> None:

        """
        This function builds channel state information between each satellite antenna and user, then
        accumulates all into a global channel state information matrix
        """

        # TODO: This will break when satellites dont have the same nr of antennas
        # TODO: This will also produce weird results when users or sats are not numbered consecutively

        # update channel state per satellite
        for satellite in self.satellites:
            satellite.update_channel_state_information(
                channel_model=channel_model,
                users=user_manager.users,
            )


        channel_state = [
            satellite.channel_state_to_users
            for satellite in self.satellites
        ]
        channel_state = np.concatenate(channel_state, axis=1)

        self.channel_state_information = channel_state[user_manager.active_user_idx]

    def update_erroneous_channel_state_information(
            self,
            channel_model,
            user_manager: 'src.data.user_manager.UserManager'
    ) -> None:
        """Todo: doc"""

        # TODO: This will break when satellites dont have the same nr of antennas
        # TODO: This will also produce weird results when users or sats are not numbered consecutively

        # apply error model per satellite
        for satellite in self.satellites:
            satellite.update_erroneous_channel_state_information(
                channel_model=channel_model,
                users=user_manager.users,
            )

        erroneous_channel_state = [
            satellite.erroneous_channel_state_to_users
            for satellite in self.satellites
        ]
        erroneous_channel_state = np.concatenate(erroneous_channel_state, axis=1)

        self.erroneous_channel_state_information = erroneous_channel_state[user_manager.active_user_idx]

    def update_scaled_erroneous_channel_state_information(
            self,
            channel_model,
            users: list,
    ) -> None:
        """
        Updates scaled erroneous csit per satellite. Note: a scaling factor must be set.
        """

        for satellite in self.satellites:
            satellite.update_scaled_erroneous_channel_state_information(channel_model=channel_model, users=users)

    def set_csi_error_scale(
            self,
            scale: float,
    ) -> None:
        """
        Set csi error scale for all satellites.
        """

        for satellite in self.satellites:
            satellite.csi_error_scale = scale

    def get_local_channel_state_information(
            self,
            satellite_id: int,
            own: str,  # ['error_free', 'erroneous']
            others: str,  # ['erroneous', 'scaled_erroneous']
    ) -> np.ndarray:
        """
        Generate a full csi matrix for all satellites & users. CSI values have different
        degrees of precision: own csi can be error-free or erroneous, other satellites
        csi can be erroneous or scaled_erroneous (usually higher error, simulates communication delay)
        """

        num_users = self.satellites[satellite_id].channel_state_to_users.shape[0]

        local_channel_state = np.zeros(
            (num_users, self.satellites[satellite_id].antenna_nr*len(self.satellites)),
            dtype='complex128',
        )

        # own
        if own == 'error_free':
            local_channel_state[:, satellite_id*self.satellites[satellite_id].antenna_nr:satellite_id*self.satellites[satellite_id].antenna_nr+self.satellites[satellite_id].antenna_nr] = self.satellites[satellite_id].channel_state_to_users
        elif own == 'erroneous':
            local_channel_state[:, satellite_id*self.satellites[satellite_id].antenna_nr:satellite_id*self.satellites[satellite_id].antenna_nr+self.satellites[satellite_id].antenna_nr] = self.satellites[satellite_id].erroneous_channel_state_to_users
        else:
            raise ValueError(f'invalid config {own}')

        for satellite in self.satellites:
            if satellite.idx == satellite_id:
                continue  # skip this one
            if others == 'erroneous':
                local_channel_state[:, satellite.idx*satellite.antenna_nr:satellite.idx*satellite.antenna_nr+satellite.antenna_nr] = satellite.erroneous_channel_state_to_users
            elif others == 'scaled_erroneous':
                local_channel_state[:, satellite.idx*satellite.antenna_nr:satellite.idx*satellite.antenna_nr+satellite.antenna_nr] = satellite.scaled_erroneous_channel_state_to_users
            else:
                raise ValueError(f'invalid config {others}')

        return local_channel_state

    def get_erroneous_channel_state_information_per_sat(
            self,
    ) -> list[np.ndarray]:

        erroneous_channel_states = [satellite.erroneous_channel_state_to_users for satellite in self.satellites]
        return erroneous_channel_states

    def get_aods_to_users(
            self,
    ) -> np.ndarray:
        """Todo: doc"""

        aods_to_users = np.zeros((len(self.satellites), len(self.satellites[0].aods_to_users)))
        for satellite_id, satellite in enumerate(self.satellites):
            aods_to_users[satellite_id, :] = satellite.aods_to_users

        return aods_to_users

    def get_inter_satellite_distances(
            self
    ) -> np.ndarray:
        """todo: doc"""

        from src.utils.euclidian_distance import euclidian_distance

        distances = np.zeros((len(self.satellites), len(self.satellites)))
        for satellite_id in range(len(self.satellites)):
            for other_satellite_id in range(satellite_id+1, len(self.satellites)):
                distance = euclidian_distance(
                    self.satellites[satellite_id].cartesian_coordinates,
                    self.satellites[other_satellite_id].cartesian_coordinates
                )
                distances[satellite_id, other_satellite_id] = distance
                distances[other_satellite_id, satellite_id] = -distance

        return distances

