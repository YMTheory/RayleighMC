import numpy as np

def rotationAngle(vec1, vec2):
    return np.arccos(vec1.dot(vec2) / np.sqrt(vec1.dot(vec1) * vec2.dot(vec2)))


def rotationAxis(vec1, vec2):
    return np.cross(vec1, vec2)


def Normalize(vec):
    return np.sqrt(vec.dot(vec))


def RotationMatrix(angle, ax):
    norm = Normalize(ax)

    rotMat = np.ones([3, 3])
    ax_norm = ax / Normalize(ax)
    ax0, ax1, ax2 = ax_norm[0], ax_norm[1], ax_norm[2]

    rotMat[0, 0] = np.cos(angle) + ax0**2 * (1-np.cos(angle))
    rotMat[0, 1] = ax0*ax1*(1 - np.cos(angle)) - ax2*np.sin(angle)
    rotMat[0, 2] = ax1*np.sin(angle) + ax0*ax2*(1 - np.cos(angle))

    rotMat[1, 0] = ax2 * np.sin(angle) + ax0*ax1*(1-np.cos(angle))
    rotMat[1, 1] = np.cos(angle) + ax1**2 * (1 - np.cos(angle))
    rotMat[1, 2] = -ax0 * np.sin(angle) + ax1 * ax2 *(1 - np.cos(angle))

    rotMat[2, 0] = -ax1 * np.sin(angle) + ax0 * ax2 * (1 - np.cos(angle))
    rotMat[2, 1] = ax0 * np.sin(angle) + ax1 * ax2 * (1 - np.cos(angle))
    rotMat[2, 2] = np.cos(angle) + ax2**2 * (1 - np.cos(angle))

    return rotMat


def rotateVector(mat, vec):
    vec_new = np.matmul(mat, vec)
    return vec_new


#if __name__ == "__main__":
#
#    ax = np.array([0, 0, 1])
#    rotMat = RotationMatrix(np.pi/2, ax)
#
#    vec = np.array([1, 0, 0])
#    vec_new = rotateVector(rotMat, vec)
#    print(vec_new)

    
