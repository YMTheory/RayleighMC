import numpy as np
import matplotlib.pyplot as plt

from Rayleigh_class import Rayleigh
from detector import detector

import sys, os
import random
from tqdm import tqdm

import uproot3

if __name__ == "__main__" :

    ray = Rayleigh()
    det = detector()

    ray.set_rhov(0.18)
    ray.calcTensor()

    ray.set_inPol(0, 1, 0)
    ray.set_inMom(0, 0, 1)

    Nphoton = 10000
    photon_momx = []
    photon_momy = []
    photon_momz = []
    photon_polx = []
    photon_poly = []
    photon_polz = []

    for i in tqdm(range(Nphoton)):

        #beta = random.uniform(0, 2*np.pi)
        #polx = np.cos(beta)
        #poly = np.sin(beta)
        #ray.set_inPol(polx, poly, 0)

        flag = True

        while flag:
            phi = random.uniform(0, 2*np.pi)
            costheta = random.uniform(-1, 1)
            theta = np.arccos(costheta)

            ray.set_outMomTheta(theta)
            ray.set_outMomPhi(phi)
            ray.set_outMom(ray.get_inE()*np.cos(ray.get_outMomPhi())*np.sin(ray.get_outMomTheta()), ray.get_inE()*np.sin(ray.get_outMomPhi())*np.sin(ray.get_outMomTheta()), ray.get_inE()*np.cos(ray.get_outMomTheta()) )

            ray.rotate_inPol_twice()
            
            prob = ray.get_scatProb()

            sample = random.uniform(0, 1)
            if sample <= prob:
                ## keep
                photon_polx.append(ray.get_outPol()[0])
                photon_poly.append(ray.get_outPol()[1])
                photon_polz.append(ray.get_outPol()[1])
                photon_momx.append(ray.get_outMom()[0])
                photon_momy.append(ray.get_outMom()[1])
                photon_momz.append(ray.get_outMom()[2])
                flag = False

    photon_momx = np.array(photon_momx)
    photon_momy = np.array(photon_momy)
    photon_momz = np.array(photon_momz)
    photon_polx = np.array(photon_polx)
    photon_poly = np.array(photon_poly)
    photon_polz = np.array(photon_polz)

    with uproot3.recreate("./rootfiles/"+sys.argv[1]) as f:
        f["Ray"] = uproot3.newtree({"photon_momx" : "float64", "photon_momy" : "float64", "photon_momz":"float64", "photon_polx":"float64", "photon_poly":"float64", "photon_polz":"float64"})
        f["Ray"].extend({"photon_momx" : photon_momx, "photon_momy" : photon_momy, "photon_momz":photon_momz, "photon_polx":photon_polx, "photon_poly":photon_poly, "photon_polz":photon_polz})


