
import numpy as np

import src
from src.utils.real_complex_vector_reshaping import (
    complex_vector_to_double_real_vector,
    complex_vector_to_rad_and_phase,
)


# todo: cleanup

def get_state_erroneous_channel_state_information(
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        csi_format: str,
        norm_state: bool,
        norm_factors: dict or list or None = None,
        per_sat: bool = False,
) -> list[np.ndarray] or np.ndarray:
    """TODO: Comment"""

    # FOR CONFIG:
    # self.get_state_args = {
    #     'csi_format': 'rad_phase',  # 'rad_phase', 'real_imag'
    #     'norm_state': True,  # !!HEURISTIC!!, this will break if you dramatically change the setup
    # }

    def method_rad_phase(complex_input, norm_factors):
        state_real = complex_vector_to_rad_and_phase(complex_input)
        if norm_state:
            half_length_idx = int(len(state_real) / 2)

            # normalize radius
            # heuristic standardization
            state_real[:half_length_idx] -= norm_factors['radius_mean']  # needs a moderate amount of samples
            state_real[:half_length_idx] /= norm_factors['radius_std']  # needs few samples

            # normalize phase
            # heuristic standardization
            # state_real[half_length_idx:] -= norm_factors['phase_mean']  # needs A LOT of samples
            state_real[half_length_idx:] /= norm_factors['phase_std']  # needs few samples

        return state_real

    def method_rad_phase_reduced(complex_input, norm_factors):
        state_real = method_rad_phase(complex_input, norm_factors)

        num_users = satellite_manager.satellites[0].user_nr
        num_antennas = satellite_manager.satellites[0].antenna_nr
        num_satellites = len(satellite_manager.satellites)
        if num_satellites > 1:
            raise ValueError('Not implemented yet, you need to add some math here')  # todo
        keep_indices = np.arange(num_users) * num_antennas
        remove_indices = np.delete(np.arange(num_users * num_antennas), keep_indices)
        state_real = np.delete(state_real, remove_indices)

        return state_real

    def method_real_imag(complex_input, norm_factors):
        state_real = complex_vector_to_double_real_vector(complex_input)
        if norm_state:
            # heuristic standardization
            state_real -= norm_factors['mean']
            state_real /= norm_factors['std']

        return state_real

    if norm_state and norm_factors is None:
        raise ValueError('no norm factors provided')

    if per_sat and norm_state and type(norm_factors) is not list:
        raise ValueError('not enough norm factors')

    if per_sat:
        erroneous_csi = satellite_manager.get_erroneous_channel_state_information_per_sat()
        erroneous_csi = [erroneous_csi_sat.flatten() for erroneous_csi_sat in erroneous_csi]

        states_real = []
        for entry_id, entry in enumerate(erroneous_csi):

            if norm_state:
                norm_factors_sat = norm_factors[entry_id]
            else:
                norm_factors_sat = None

            if csi_format == 'rad_phase':
                states_real.append(method_rad_phase(entry, norm_factors_sat))

            # like rad_phase, but only one radius per satellite+user. Rationale: path loss dominates as d_satuser >> d_antenna
            elif csi_format == 'rad_phase_reduced':
                states_real.append(method_rad_phase_reduced(entry, norm_factors_sat))

            elif csi_format == 'real_imag':
                states_real.append(method_real_imag(entry, norm_factors_sat))

            else:
                raise ValueError(f'Unknown CSI Format {csi_format}')

        return states_real

    else:
        erroneous_csi = satellite_manager.erroneous_channel_state_information.flatten()
        if csi_format == 'rad_phase':
            state_real = method_rad_phase(erroneous_csi, norm_factors)

        # like rad_phase, but only one radius per satellite+user. Rationale: path loss dominates as d_satuser >> d_antenna
        elif csi_format == 'rad_phase_reduced':
            state_real = method_rad_phase_reduced(erroneous_csi, norm_factors)

        elif csi_format == 'real_imag':
            state_real = method_real_imag(erroneous_csi, norm_factors)

        else:
            raise ValueError(f'Unknown CSI Format {csi_format}')

        return state_real


def get_state_erroneous_channel_state_information_local(
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        csi_format: str,
        local_csi_own_quality: str,
        local_csi_others_quality: str,
        norm_state: bool,
        norm_factors: dict or list or None = None,
) -> list[np.ndarray]:

    def method_rad_phase(complex_input, norm_factors):
        state_real = complex_vector_to_rad_and_phase(complex_input)
        if norm_state:
            half_length_idx = int(len(state_real) / 2)

            # normalize radius
            # heuristic standardization
            state_real[:half_length_idx] -= norm_factors['radius_mean']  # needs a moderate amount of samples
            state_real[:half_length_idx] /= norm_factors['radius_std']  # needs few samples

            # normalize phase
            # heuristic standardization
            # state_real[half_length_idx:] -= norm_factors['phase_mean']  # needs A LOT of samples
            state_real[half_length_idx:] /= norm_factors['phase_std']  # needs few samples

        return state_real

    if norm_state and norm_factors is None:
        raise ValueError('no norm factors provided')

    local_csits = []
    for satellite in satellite_manager.satellites:

        if csi_format == 'rad_phase':

            local_csi = satellite_manager.get_local_channel_state_information(
                satellite_id=satellite.idx,
                own=local_csi_own_quality,
                others=local_csi_others_quality,
            ).flatten()

            local_csi_real = method_rad_phase(local_csi, norm_factors)

            local_csits.append(local_csi_real)

        else:
            raise ValueError(f'Unknown CSI Format {csi_format}')

    return local_csits


def get_state_aods(
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_state: bool,
        norm_factors: dict = None,
) -> np.ndarray:
    """TODO: Comment"""

    # FOR CONFIG:
    # self.get_state_args = {
    #     'norm_state': True,  # !!HEURISTIC!!, this will break if you dramatically change the setup
    # }

    if norm_state and norm_factors is None:
        raise ValueError('no norm factors provided')

    state = satellite_manager.get_aods_to_users().flatten()
    if norm_state:

        # heuristic standardization
        state -= norm_factors['mean']
        state /= norm_factors['std']

    return state.flatten()
