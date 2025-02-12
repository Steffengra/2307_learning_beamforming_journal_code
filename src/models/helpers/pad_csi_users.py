
import numpy as np

import src


def pad_csi_users(
        config: 'src.config.config.Config',
        user_manager: 'src.data.user_manager.UserManager',
        csi: np.ndarray,
) -> np.ndarray:
    """zeropad csi to a certain user length"""

    total_users = config.user_nr
    active_user_idx = user_manager.active_user_idx

    padded_csi = []
    for user_idx in range(total_users):
        if user_idx in active_user_idx:
            padded_csi.append(
                csi[np.where(active_user_idx == user_idx)[0], :][0]
            )
        else:
            padded_csi.append(np.zeros(shape=csi[0, :].shape, dtype='complex128'))

    padded_csi = np.vstack(padded_csi)

    return padded_csi
