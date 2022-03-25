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
    ray.set_inMom(0, 0, 1)
    ray.set_inPol(0, 1, 0)


    polAng = np.arange(0, 361, 1)

    #start_theta, stop_theta, step_theta = 30., 160., 10.
    start_theta, stop_theta, step_theta = 90, 100, 10.
    N_theta = int((stop_theta - start_theta ) /step_theta)

    print("N_theta = %d" %N_theta)
    Prob = [[] for i in range(N_theta+1)]

    n = 0
    for angle in np.arange(start_theta, stop_theta, step_theta):

        print("outgoing theta angle : %.1f" %angle)

        ray.set_outMomTheta(angle/180*np.pi)
        ray.set_outMomPhi(90/180*np.pi)
        ray.set_outMom(ray.get_inE()*np.cos(ray.get_outMomPhi())*np.sin(ray.get_outMomTheta()), ray.get_inE()*np.sin(ray.get_outMomPhi())*np.sin(ray.get_outMomTheta()), ray.get_inE()*np.cos(ray.get_outMomTheta()) )

        ray.calculatePol()
        outPol = ray.get_outPol()
        #print(outPol)
        det.set_inPol(outPol[0], outPol[1], outPol[2])

        detPol1 = np.array([1, 0 ,0])
        detPol2 = np.array([0, np.cos(angle/180*np.pi), np.sin(angle/180*np.pi)])

        for i in polAng:
            CosAng = np.cos(i/180*np.pi)
            SinAng = np.sin(i/180*np.pi)

            detPol = CosAng * detPol1 + SinAng * detPol2

            det.set_detPol(detPol[0], detPol[1], detPol[2])
            Prob[n].append(det.calcDetProb())

            #print(ray.get_outPol(), i, det.get_detPol(), det.calcDetProb())

        n += 1
     


    #fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    #for i in range(N_theta):
    #    ax.plot(polAng*np.pi/180, Prob[i], lw=2, label="theta = %d"%(start_theta+i*step_theta))
    #plt.legend()
    #plt.show()




