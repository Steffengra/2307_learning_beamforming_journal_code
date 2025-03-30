
import numpy as np

import src
from src.utils.get_wavelength import get_wavelength
from src.utils.euclidian_distance import euclidian_distance


def calc_autocorrelation(
        satellite,
        error_model_config: 'src.config.config_error_model.ConfigErrorModel',
        error_distribution: str,
) -> np.ndarray:
    """
    TODO: Comment
    error distribution: 'uniform', 'gaussian'
    """

    wavelength = get_wavelength(satellite.freq)
    wavenumber = 2 * np.pi / wavelength
    user_nr = satellite.user_nr
    sat_ant_nr = satellite.antenna_nr
    sat_antenna_spacing = satellite.antenna_distance
    errors = satellite.estimation_errors

    # calc characteristic function all at once for performance reasons
    antenna_shift_idxs = np.zeros((sat_ant_nr, sat_ant_nr))
    for antenna_row_id in range(sat_ant_nr):
        for antenna_col_id in range(sat_ant_nr):
            antenna_shift_idxs[antenna_row_id, antenna_col_id] = antenna_col_id - antenna_row_id
    antenna_shift_idxs = antenna_shift_idxs

    if error_distribution == 'uniform':
        temp = (
                antenna_shift_idxs * wavenumber * sat_antenna_spacing
                * error_model_config.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['high']
                / np.pi
        )
        characteristic_functions = np.sinc(temp)

    elif error_distribution == 'gaussian':
        temp = (
                (-(antenna_shift_idxs * wavenumber * sat_antenna_spacing)) ** 2
                * error_model_config.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['scale'] ** 2
                / 2
        )
        characteristic_functions = np.exp(temp)

    else:
        raise ValueError('Unknown error distribution on cosine of AODs')

    temp = np.cos(satellite.aods_to_users + errors['additive_error_on_aod']) + errors['additive_error_on_cosine_of_aod']
    temp = temp[:, np.newaxis, np.newaxis] * (1j * wavenumber * sat_antenna_spacing * (-antenna_shift_idxs))
    temp = np.exp(temp)
    autocorrelation_matrix = characteristic_functions[np.newaxis, :, :] * temp

    return autocorrelation_matrix


def calc_autocorrelation_multisat(
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        error_model_config: 'src.config.config_error_model.ConfigErrorModel',
        error_distribution: str,
) -> np.ndarray:

    wavelength = get_wavelength(satellite_manager.satellites[0].freq)
    wavenumber = 2 * np.pi / wavelength
    user_nr = satellite_manager.satellites[0].user_nr
    sat_nr = len(satellite_manager.satellites)
    sat_ant_nr = satellite_manager.satellites[0].antenna_nr
    tot_ant_nr = sat_nr * sat_ant_nr
    sat_antenna_spacing = satellite_manager.satellites[0].antenna_distance

    # get local antenna distances
    local_antenna_dist_matrix = np.zeros((sat_ant_nr, sat_ant_nr))
    for antenna_row_id in range(sat_ant_nr):
        for antenna_col_id in range(sat_ant_nr):
            local_antenna_dist_matrix[antenna_row_id, antenna_col_id] = (antenna_col_id - antenna_row_id) * sat_antenna_spacing

    local_antenna_dist_matrix_per_sat = np.tile(local_antenna_dist_matrix, (sat_nr, sat_nr))
    local_antenna_dist_vector_per_sat = local_antenna_dist_matrix_per_sat[0, :]

    satellite_positions = [satellite_manager.satellites[satellite_id].cartesian_coordinates for satellite_id in range(sat_nr)]
    inter_satellite_distance_matrix = np.zeros((sat_nr, sat_nr))
    for satellite_row_id in range(sat_nr):
        for satellite_col_id in range(sat_nr):
            inter_satellite_distance_matrix[satellite_row_id, satellite_col_id] = euclidian_distance(satellite_positions[satellite_row_id], satellite_positions[satellite_col_id])
            if satellite_row_id > satellite_col_id:
                inter_satellite_distance_matrix[satellite_row_id, satellite_col_id] *= -1

    inter_satellite_distance_matrix_per_antenna = np.repeat(
        np.repeat(inter_satellite_distance_matrix, sat_ant_nr, axis=0),
        sat_ant_nr,
        axis=1,
    )

    # init autocorrelation matrix
    autocorrelation_matrix = np.zeros((user_nr, tot_ant_nr, tot_ant_nr), dtype='complex128')

    for user_id in range(user_nr):

        # cos(aod)+error per satellite
        phi_dach = [
                np.cos(
                    satellite.aods_to_users[user_id]
                    + satellite.estimation_errors['additive_error_on_aod'][user_id]
                )
                + satellite.estimation_errors['additive_error_on_cosine_of_aod'][user_id]
                for satellite in satellite_manager.satellites
        ]

        satellite_distances_to_user = [satellite.distance_to_users[user_id] for satellite in satellite_manager.satellites]
        extra_sat_distance_vector = np.array(
            [
                (
                        satellite_distances_to_user[satellite_id]
                        -
                        satellite_distances_to_user[0]
                )
                for satellite_id in range(len(satellite_manager.satellites))]
            , dtype='float128',
        )

        # correction terms
        for satellite_id in range(1, sat_nr):
            extra_sat_distance_vector[satellite_id] += np.cos(satellite_manager.satellites[0].aods_to_users[user_id]) * sat_antenna_spacing / 2
            extra_sat_distance_vector[satellite_id] -= np.cos(satellite_manager.satellites[satellite_id].aods_to_users[user_id]) * sat_antenna_spacing / 2

        extra_sat_distance_vector_tiled = np.repeat(extra_sat_distance_vector, sat_ant_nr)

        steering = np.exp(
            -1j
            *
            wavenumber *
            (
                local_antenna_dist_vector_per_sat
                *
                np.repeat(phi_dach, sat_ant_nr)
                +
                extra_sat_distance_vector_tiled
            )
        )

        if error_distribution == 'uniform':
            characteristic_functions = np.sinc(
                wavenumber * (
                    local_antenna_dist_matrix_per_sat
                    +
                    inter_satellite_distance_matrix_per_antenna
                )
                * error_model_config.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['high']
                / np.pi
            )

        # todo: copy gaussian

        else:
            raise ValueError('Unknown error distribution on cosine of AODs')

        # print(local_antenna_dist_matrix_per_sat)
        # print(inter_satellite_distance_matrix_per_antenna)
        # print(characteristic_functions)
        # exit()

        autocorrelation_matrix[user_id] = np.outer(
            a=steering.T.conj(),
            b=steering,
        # ) * characteristic_functions
        ) * 1

    return autocorrelation_matrix
