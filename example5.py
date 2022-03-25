import numpy as np
import matplotlib.pyplot as plt

from Rayleigh_class import Rayleigh
from detector import detector

import sys, os
import random
from tqdm import tqdm

import uproot3
#import h5py

if __name__ == "__main__" :

    ray = Rayleigh()
    det = detector()

    ray.set_rhov(0.00)
    ray.calcTensor()

    ray.set_inPol(0, 1, 0)
    ray.set_inMom(0, 0, 1)

    Nphoton = 100000
    in_polx = []
    in_poly = []
    in_polz = []
    photon_momx = []
    photon_momy = []
    photon_momz = []
    photon_polx = []
    photon_poly = []
    photon_polz = []

    for i in tqdm(range(Nphoton)):

        
        beta = 0.
        #beta = random.uniform(0, 2*np.pi)
        #beta = np.pi/2
        polx = np.cos(beta)
        poly = np.sin(beta)
        ray.set_inPol(polx, poly, 0)
        in_polx.append(polx)
        in_poly.append(poly)
        in_polz.append(0)

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
        f["Ray"] = uproot3.newtree({"in_polx":"float64", "in_poly":"float64", "in_polz":"float64", "photon_momx" : "float64", "photon_momy" : "float64", "photon_momz":"float64", "photon_polx":"float64", "photon_poly":"float64", "photon_polz":"float64"})
        f["Ray"].extend({"in_polx":in_polx, "in_poly":in_poly, "in_polz":in_polz, "photon_momx" : photon_momx, "photon_momy" : photon_momy, "photon_momz":photon_momz, "photon_polx":photon_polx, "photon_poly":photon_poly, "photon_polz":photon_polz})

    #with h5py.File("./rootfiles/"+sys.argv[1], "w") as hf:
    #    hf.create_dataset("in_polx", data=in_polx)
    #    hf.create_dataset("in_poly", data=in_poly)
    #    hf.create_dataset("in_polz", data=in_polz)
    #    hf.create_dataset("photon_momx", data=photon_momx)
    #    hf.create_dataset("photon_momy", data=photon_momy)
    #    hf.create_dataset("photon_momz", data=photon_momz)
    #    hf.create_dataset("photon_polx", data=photon_polx)
    #    hf.create_dataset("photon_poly", data=photon_poly)
    #    hf.create_dataset("photon_polz", data=photon_polz)




