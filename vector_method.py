import numpy as np

def perpendicular_vector(v):
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 1])
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, vec2)
    else:
        return np.cross(v, vec1)
