
import tensorflow as tf
import numpy as np


def bound_actions(
        actions: np.ndarray,
        mode: str or None = None,
) -> np.ndarray:
    """
    Bound actions to a certain output range / support, in numpy
    """

    if mode is None:
        return actions

    elif mode == 'tanh':  # (-1, 1)
        return np.tanh(actions)

    elif mode == 'tanh_positive':  # (0, 1)
        return 0.5 * (np.tanh(actions) + 1)

    else:
        raise ValueError('unknown mode')


def bound_actions_and_log_prob_densities(
        actions: np.ndarray,
        action_log_prob_densities: np.ndarray,
        mode: str or None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bound actions to a certain output range / support and correct logprob densities accordingly, in tf
    from Soft Actor-Critic Algorithms and Applications, haarnoja et al.
    """

    if mode is None:
        return actions, action_log_prob_densities

    elif mode == 'tanh':  # (-1, 1)
        actions_bounded = tf.math.tanh(actions)

        correction_term = tf.reduce_sum(
            tf.math.log(
                1
                - actions_bounded**2
                + 1e-6  # for numerical stability, needed?
            ),
            axis=1,
        ) / actions.shape[1]

    elif mode == 'tanh_positive':  # (0, 1)
        actions_bounded = 0.5 * (tf.math.tanh(actions) + 1)

        correction_term = tf.reduce_sum(
            tf.math.log(
                0.5 * (
                    1
                    - actions_bounded**2
                    + 1e-6  # for numerical stability, needed?
                )
            ),
            axis=1,
        ) / actions.shape[1]

    else:
        raise ValueError('unknown mode')

    action_log_prob_densities_adjusted = action_log_prob_densities - tf.transpose(correction_term[tf.newaxis])

    return actions_bounded, action_log_prob_densities_adjusted


@tf.function
def bound_actions_and_log_prob_densities_graph(
        actions: np.ndarray,
        action_log_prob_densities: np.ndarray,
        mode: str or None = None,
) -> tuple[tf.Tensor, tf.Tensor]:

    return bound_actions_and_log_prob_densities(actions, action_log_prob_densities, mode)
