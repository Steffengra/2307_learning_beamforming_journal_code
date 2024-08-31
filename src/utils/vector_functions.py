
import numpy as np


def unit_vector(vector):
    """
    Returns the unit vector of the vector.
    from https://stackoverflow.com/a/13849249
    """

    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            # >>> angle_between((1, 0, 0), (0, 1, 0))
            # 1.5707963267948966
            # >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            # >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    from https://stackoverflow.com/a/13849249
    """

    dot_product = np.dot(v1, v2)
    length_square_v1 = np.dot(v1, v1)
    length_square_v2 = np.dot(v2, v2)
    norm = np.sqrt(length_square_v1 * length_square_v2)

    return np.arccos(np.clip(dot_product / norm, -1, 1))
