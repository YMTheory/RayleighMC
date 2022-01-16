import numpy as np
import ROOT

class detector(object):

    def __init__(self) -> None:

        self.inPol = np.array([1., 0., 0.])

        self.detPol = np.array([1., 0., 0.])
        self.detTheta = 90
        self.detPhi = 90

        self.detProb = 1.


    def set_inPol(self, x, y ,z):
        self.inPol[0] = x
        self.inPol[1] = y
        self.inPol[2] = z

    def get_inPol(self):
        return self.inPol


    def set_detPol(self, x, y, z):
        self.detPol[0] = x
        self.detPol[1] = y
        self.detPol[2] = z

    def get_detPol(self):
        return self.detPol
    

    def set_detTheta(self, theta):
        self.detTheta = theta


    def set_detPhi(self, phi):
        self.detPhi = phi

    def get_detTheta(self, theta):
        return self.detTheta


    def get_detPhi(self, phi):
        return self.detPhi


    def calcDetProb(self):
        return (self.inPol.dot(self.detPol))**2



