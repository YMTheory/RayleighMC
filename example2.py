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
    Nsample = 10000

    polAngle_arr = np.arange(0, 370, 10)

    theta_start, theta_stop, theta_step = 90, 100, 10.
    Nstep = int((theta_stop - theta_start) / theta_step)
    prob_arr = [[] for i in range(Nstep)]

    count = 0

    for tt in np.arange(theta_start, theta_stop, theta_step):
        ray.set_outMomTheta(tt/180*np.pi)
        ray.set_outMomPhi(np.pi/2)
        ray.set_outMom(ray.get_inE()*np.cos(ray.get_outMomPhi())*np.sin(ray.get_outMomTheta()), ray.get_inE()*np.sin(ray.get_outMomPhi())*np.sin(ray.get_outMomTheta()), ray.get_inE()*np.cos(ray.get_outMomTheta()) )

        outPol_arr = [ [] for i in range(Nsample)]
        for ii in range(Nsample):
            ray.set_inPol(1, 0, 0)
            ray.rotate_inPol()
            ray.calculatePol()
            outPol_arr[ii].append(ray.get_outPol()[0])
            outPol_arr[ii].append(ray.get_outPol()[1])
            outPol_arr[ii].append(ray.get_outPol()[2])

        detPol1 = np.array([1, 0 ,0])
        detPol2 = np.array([0, np.cos(tt/180*np.pi), np.sin(tt/180*np.pi)])

        for detAngle in polAngle_arr:
            print("Detection Polarisation Angle : ", detAngle)
            amp = 0
            
            CosAng = np.cos(detAngle/180*np.pi)
            SinAng = np.sin(detAngle/180*np.pi)

            detPol = CosAng * detPol1 + SinAng * detPol2

            for photon in outPol_arr:
                det.set_inPol(photon[0], photon[1], photon[2])
                det.set_detPol(detPol[0], detPol[1], detPol[2])
                amp += det.calcDetProb()

            prob_arr[count].append(amp)
            print(amp)

        count += 1


    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    for i in range(Nstep):
        ax.plot(polAngle_arr*np.pi/180, prob_arr[i], lw=2)
    #plt.legend()
    plt.show()



    


