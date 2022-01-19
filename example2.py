import numpy as np
import matplotlib.pyplot as plt
import hist
from hist import Hist

from Rayleigh_class import Rayleigh
from detector import detector

import sys, os

from tqdm import tqdm


if __name__ == "__main__" :

    ## detector :
    det = detector()

    ## Rayleigh process:
    ray = Rayleigh()
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})


    # input configuration
    ray.set_inMom(0, 0, 1)     # incident light momentum
    ray.set_inPol(1, 0, 0)     # incident light polarisation
    
    # sef depolarisation ratio
    ray.set_rhou(0.30)      # LAB case
    ray.calcTensor()

    #### output depolarisation related properties ############
    print("======================================")
    print(r"$\rho_v$ = %.2f" %ray.get_rhov())
    print(r"$\rho_u$ = %.2f" %ray.get_rhou())
    print(r"$\alpha$ = %.2f, $\beta$ = %.2f" %(ray.get_alpha(), ray.get_beta()))
    print("======================================")

    
    ver_flag = False
    hor_flag = True

    Nsample = 10000

    polAngle_arr = np.arange(0, 361, 1)

    theta_start, theta_stop, theta_step = 30, 100, 10.
    Nstep = int((theta_stop - theta_start) / theta_step)

    if ver_flag:
        prob_arr = [[] for i in range(Nstep)]

        count = 0

        for tt in tqdm(np.arange(theta_start, theta_stop, theta_step)):
            print("Detecting at theta = %d degree" %tt)
            ray.set_outMomTheta(tt/180*np.pi)
            ray.set_outMomPhi(3*np.pi/2)
            ray.set_outMom(ray.get_inE()*np.cos(ray.get_outMomPhi())*np.sin(ray.get_outMomTheta()), ray.get_inE()*np.sin(ray.get_outMomPhi())*np.sin(ray.get_outMomTheta()), ray.get_inE()*np.cos(ray.get_outMomTheta()) )

            outPol_arr = [ [] for i in range(Nsample)]
            amp_arr = []
            for ii in range(Nsample):
                ray.set_inPol(1, 0, 0)
                ray.rotate_inPol()
                ray.calculatePol()
                outPol_arr[ii].append(ray.get_outPol()[0])
                outPol_arr[ii].append(ray.get_outPol()[1])
                outPol_arr[ii].append(ray.get_outPol()[2])
                amp_arr.append(ray.get_scatProb())

            detPol1 = np.array([1, 0 ,0])
            detPol2 = np.array([0, np.cos(tt/180*np.pi), np.sin(tt/180*np.pi)])

            for detAngle in polAngle_arr:
                amp = 0
                
                CosAng = np.cos(detAngle/180*np.pi)
                SinAng = np.sin(detAngle/180*np.pi)

                detPol = CosAng * detPol1 + SinAng * detPol2

                for idx, photon in enumerate(outPol_arr):
                    det.set_inPol(photon[0], photon[1], photon[2])
                    det.set_detPol(detPol[0], detPol[1], detPol[2])
                    amp += det.calcDetProb() * amp_arr[idx]

                prob_arr[count].append(amp)

            count += 1
            
        print("Vertical incident light: %.2f, %.2f" %(prob_arr[0][0], prob_arr[0][90]) )

        for i in range(Nstep):
            ax.plot(polAngle_arr*np.pi/180-np.pi/2, prob_arr[i], lw=2)
        ax.set_yticklabels([])
        #plt.legend()



    if hor_flag:
        prob_arr = [[] for i in range(Nstep)]
        amp_arr = []

        count = 0
        for tt in tqdm(np.arange(theta_start, theta_stop, theta_step)):
            print("Detecting at theta = %d degree" %tt)
            ray.set_outMomTheta(tt/180*np.pi)
            ray.set_outMomPhi(3*np.pi/2)
            ray.set_outMom(ray.get_inE()*np.cos(ray.get_outMomPhi())*np.sin(ray.get_outMomTheta()), ray.get_inE()*np.sin(ray.get_outMomPhi())*np.sin(ray.get_outMomTheta()), ray.get_inE()*np.cos(ray.get_outMomTheta()) )

            outPol_arr = [ [] for i in range(Nsample)]
            amp_arr = []
            for ii in range(Nsample):
                ray.set_inPol(0, 1, 0)
                ray.rotate_inPol()
                ray.calculatePol()
                outPol_arr[ii].append(ray.get_outPol()[0])
                outPol_arr[ii].append(ray.get_outPol()[1])
                outPol_arr[ii].append(ray.get_outPol()[2])
                amp_arr.append(ray.get_scatProb())

            detPol1 = np.array([1, 0 ,0])
            detPol2 = np.array([0, np.cos(tt/180*np.pi), np.sin(tt/180*np.pi)])

            for detAngle in polAngle_arr:
                amp = 0
                
                CosAng = np.cos(detAngle/180*np.pi)
                SinAng = np.sin(detAngle/180*np.pi)

                detPol = CosAng * detPol1 + SinAng * detPol2

                for idx, photon in enumerate(outPol_arr):
                    det.set_inPol(photon[0], photon[1], photon[2])
                    det.set_detPol(detPol[0], detPol[1], detPol[2])
                    amp += det.calcDetProb() * amp_arr[idx]

                prob_arr[count].append(amp)

            count += 1

        #print("Horizontal incident light: %.2f, %.2f" %(prob_arr[0][0], prob_arr[0][90]) )

        for i in range(Nstep):
            print("Horizontal incident light: %.2f" %((prob_arr[i][270]+prob_arr[i][90])/2.) )
            ax.plot(polAngle_arr*np.pi/180-np.pi/2, prob_arr[i], lw=2)
        ax.set_yticklabels([])
        #plt.legend()

    
    #plt.savefig("Anisotropic_rhou03_HorVer_Theta90.pdf")
    plt.show()
    
    


