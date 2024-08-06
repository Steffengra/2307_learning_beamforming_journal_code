
import numpy as np

from src.utils.spherical_to_cartesian_coordinates import (
    spherical_to_cartesian_coordinates,
)
from src.utils.euclidian_distance import (
    euclidian_distance,
)
from src.utils.get_wavelength import (
    get_wavelength,
)
from src.utils.vector_functions import angle_between


class Satellite:
    """A satellite object represents a physical satellite."""

    def __init__(
            self,
            rng: np.random.default_rng,
            idx: int,
            spherical_coordinates: np.ndarray,
            antenna_nr: int,
            antenna_distance: float,
            antenna_gain_linear: float,
            user_nr: int,
            freq: float,
            center_aod_earth_deg: float,
            error_functions: dict,
    ) -> None:

        self.rng = rng

        self.idx: int = idx
        self.spherical_coordinates: np.ndarray = spherical_coordinates
        self.cartesian_coordinates: np.ndarray = spherical_to_cartesian_coordinates(spherical_coordinates)

        self.antenna_nr: int = antenna_nr
        self.antenna_distance: float = antenna_distance  # antenna distance in meters
        self.antenna_gain_linear: float = antenna_gain_linear

        self.user_nr: int = user_nr

        self.freq: float = freq
        self.wavelength: float = get_wavelength(self.freq)

        self.center_aod_earth_deg: float = center_aod_earth_deg

        self.distance_to_users = None  # user_idx[int]: dist[float]
        self.aods_to_users = None  # user_idx[int]: aod[float] in rad, [0, 2pi], most commonly ~pi/2, aod looks from sat towards users
        self.steering_error = None

        self.csi_error_scale = None
        self.channel_state_to_users: np.ndarray = np.array([])  # depends on channel model
        self.erroneous_channel_state_to_users: np.ndarray = np.array([])  # depends on channel & error model
        self.scaled_erroneous_channel_state_to_users: np.ndarray = np.array([])  # e.g., for sharing with other sats

        self.estimation_error_functions: dict = error_functions
        self.estimation_errors: dict = {}

    def update_position(
            self,
            spherical_coordinates: np.ndarray,
    ) -> None:
        """TODO: Comment"""

        self.spherical_coordinates = spherical_coordinates
        self.cartesian_coordinates = spherical_to_cartesian_coordinates(spherical_coordinates)

    def calculate_distance_to_users(
            self,
            users: list,
    ) -> None:
        """TODO: Comment"""

        if self.distance_to_users is None:
            self.distance_to_users = np.zeros(len(users))

        for user in users:
            self.distance_to_users[user.idx] = euclidian_distance(self.cartesian_coordinates,
                                                                  user.cartesian_coordinates)

    def calculate_aods_to_users(
            self,
            users: list
    ) -> None:
        """
        The calculation of AOD from the point of view of the satellite.
        """

        if self.aods_to_users is None:
            self.aods_to_users = np.zeros(len(users))

        for user in users:
            vec = user.cartesian_coordinates - self.cartesian_coordinates
            self.aods_to_users[user.idx] = angle_between(np.array([-1, 0, 0]), vec)

    def roll_estimation_errors(
            self,
    ) -> None:
        """TODO: Comment"""

        for estimation_error_name, error_function in self.estimation_error_functions.items():
            self.estimation_errors[estimation_error_name] = error_function()

    def update_channel_state_information(
            self,
            channel_model,
            users: list,
    ) -> None:
        """
        This function updates the channel state to given users
        according to a given channel model.
        """
        self.channel_state_to_users = channel_model(self, users, scale=0)

    def update_erroneous_channel_state_information(
            self,
            channel_model,
            users: list,
    ) -> None:
        """
        This function updates erroneous channel state information to users
        according to a given user list and error model config.
        """

        self.erroneous_channel_state_to_users = channel_model(satellite=self, users=users, scale=1)

    def update_scaled_erroneous_channel_state_information(
            self,
            channel_model,
            users: list,
    ) -> None:
        """
        This function updates erroneous channel state information to users
        with scaled error realizations, e.g., for sharing csi with increased error w/
        other users.
        """

        if self.csi_error_scale is not None:
            self.scaled_erroneous_channel_state_to_users = channel_model(satellite=self, users=users, scale=self.csi_error_scale)
