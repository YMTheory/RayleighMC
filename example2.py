import numpy as np
import matplotlib.pyplot as plt
import hist
from hist import Hist

from Rayleigh_class import Rayleigh
from detector import detector

import sys, os


if __name__ == "__main__" :

    ## detector :
    det = detector()

    ## Rayleigh process:
    ray = Rayleigh()


    # input configuration
    ray.set_inMom(0, 0, 1)     # incident light momentum
    ray.set_inPol(1, 0, 0)     # incident light polarisation
    
    # sef depolarisation ratio
    ray.set_rhou(0.31)      # LAB case
    ray.calcTensor()
    
    polAng = np.arange(0, 361, 1)
    Nsample = 10

    for i in range(1000):
        ray.rotateTensor()
        inPol_mod = ray.get_inPol()

        inPol_modXY = np.sqrt(inPol_mod[0]**2 + inPol_mod[1]**2)
        inPol_modR = np.sqrt(inPol_mod.dot(inPol_mod))

        theta = np.arccos(inPol_mod[2]/inPol_modR)
        phi = np.arctan(inPol_mod[1]/inPol_mod[0])




    """
    #start_theta, stop_theta, step_theta = 30., 160., 10.
    start_theta, stop_theta, step_theta = 90, 100, 10.
    N_theta = int((stop_theta - start_theta ) /step_theta)

    print("N_theta = %d" %N_theta)
    Prob = [[] for i in range(N_theta)]


    n = 0
    for angle in np.arange(start_theta, stop_theta, step_theta):

        print("outgoing theta angle : %.1f" %angle)

        ray.set_outMomTheta(angle/180*np.pi)
        ray.set_outMomPhi(90/180*np.pi)

        detPol2 = np.array([0, np.cos(angle/180*np.pi), np.sin(angle/180*np.pi)])

        for i in polAng:
            prob = 0
            print("Polarisation detection angle : %.1f"%i)
            for j in range(Nsample):
                # randomly rotate tensor
                ray.rotateTensor()
                ray.calculatePol()

                outPol = ray.get_outPol()
                det.set_inPol(outPol[0], outPol[1], outPol[2])
                detPol1 = outPol

                CosAng = np.cos(i/180*np.pi)
                SinAng = np.sin(i/180*np.pi)

                C1 = CosAng 
                C2 = 1

                detPol = CosAng * detPol1 + SinAng * detPol2

                det.set_detPol(detPol[0], detPol[1], detPol[2])
                
                prob += det.calcDetProb()
                
            Prob[n].append(prob)

        n += 1
     


    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    for i in range(N_theta):
        ax.plot(polAng*np.pi/180, Prob[i], lw=2)
    plt.show()
    """
