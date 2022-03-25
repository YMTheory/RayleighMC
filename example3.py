import numpy as np
import matplotlib.pyplot as plt

from Rayleigh_class import Rayleigh
from detector import detector

import sys, os

from tqdm import tqdm


if __name__ == "__main__" :

    ## detector :
    det = detector()

    ## Rayleigh process:
    ray = Rayleigh()
    #fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    fig, (ax1, ax2) = plt.subplots(1, 2)


    # input configuration
    ray.set_inMom(0, 0, 1)     # incident light momentum
    ray.set_inPol(1, 0, 0)     # incident light polarisation
    
    rhou_arr = np.arange(0, 0.6, 0.1)
    rhov_arr = []
    Nrhou = len(rhou_arr)
    Hh, Hv, Vh, Vv = [], [], [], []

    # sef depolarisation ratio
    for rhou in rhou_arr:
        print("Material rhou = %.3f" %rhou)

        ray.set_rhou(rhou)
        ray.calcTensor()

        rhov_arr.append(ray.get_rhov())

        #### output depolarisation related properties ############
        print("======================================")
        print(r"$\rho_v$ = %.3f" %ray.get_rhov())
        print(r"$\rho_u$ = %.3f" %ray.get_rhou())
        print(r"$\alpha$ = %.3f, $\beta$ = %.3f" %(ray.get_alpha(), ray.get_beta()))
        print("======================================")

        tt = 90
        
        ver_flag = True
        hor_flag = True

        Nsample = 10000

        polAngle_arr = np.arange(0, 361, 1)

        if ver_flag:
            prob_arr = []

            ray.set_outMomTheta(tt/180*np.pi)
            ray.set_outMomPhi(3*np.pi/2)
            ray.set_outMom(ray.get_inE()*np.cos(ray.get_outMomPhi())*np.sin(ray.get_outMomTheta()), ray.get_inE()*np.sin(ray.get_outMomPhi())*np.sin(ray.get_outMomTheta()), ray.get_inE()*np.cos(ray.get_outMomTheta()) )

            outPol_arr = [ [] for i in range(Nsample)]
            amp_arr = []
            for ii in range(Nsample):
                ray.set_inPol(1, 0, 0)
                #ray.rotate_inPol()
                #ray.calculatePol()
                #ray.rotate_inPol_twice()
                #ray.rotate_inPol_once()
                ray.calculatePol_modified()
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

                prob_arr.append(amp)

                

            #ax.plot(polAngle_arr*np.pi/180-np.pi/2, prob_arr, lw=2)
            #ax.set_yticklabels([])
            #plt.legend()

            Hv.append( (prob_arr[90]+prob_arr[270])/2. )
            Vv.append( (prob_arr[0]+prob_arr[180])/2. )


        if hor_flag:
            prob_arr = []
            amp_arr = []

            ray.set_outMomTheta(tt/180*np.pi)
            ray.set_outMomPhi(3*np.pi/2)
            ray.set_outMom(ray.get_inE()*np.cos(ray.get_outMomPhi())*np.sin(ray.get_outMomTheta()), ray.get_inE()*np.sin(ray.get_outMomPhi())*np.sin(ray.get_outMomTheta()), ray.get_inE()*np.cos(ray.get_outMomTheta()) )

            outPol_arr = [ [] for i in range(Nsample)]
            amp_arr = []
            for ii in range(Nsample):
                ray.set_inPol(0, 1, 0)
                #ray.rotate_inPol()
                #ray.calculatePol()
                #ray.rotate_inPol_twice()
                ray.calculatePol_modified()
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

                prob_arr.append(amp)


            #print("Horizontal incident light: %.2f, %.2f" %(prob_arr[0][0], prob_arr[0][90]) )

            #ax.plot(polAngle_arr*np.pi/180-np.pi/2, prob_arr, lw=2)
            #ax.set_yticklabels([])
            #plt.legend()

            Hh.append( (prob_arr[90]+prob_arr[270])/2. )
            Vh.append( (prob_arr[0]+prob_arr[180])/2. )
    
    #plt.savefig("Anisotropic_rhou03_HorVer_Theta90.pdf")
    
    Hv = np.array(Hv)
    Vv = np.array(Vv)
    rhov = Hv / Vv
    rhou_calc = rhov *2 /(1+rhov)
    
    ax1.plot(rhou_arr, rhou_calc, "o-")
    ax1.set_xlabel(r"$\rho_u$ (input)", fontsize=15)
    ax1.set_ylabel(r"$\rho_u$ (calc)", fontsize=15)
    ax1.plot([0, 0.6], [0, 0.6], "--", color="red")
    ax1.grid(True)

    ax2.plot(rhov_arr, rhov, "o-")
    ax2.set_xlabel(r"$\rho_v$ (input)", fontsize=15)
    ax2.set_ylabel(r"$\rho_v$ (calc)", fontsize=15)
    ax2.plot([0, 0.4], [0, 0.4], "--", color="red")
    ax2.grid(True)

    #ax1.plot(rhou_arr, Hv, "o-", label=r"$H_v$")
    #ax1.plot(rhou_arr, Vv, "o-", label=r"$V_v$")
    #ax1.set_xlabel(r"$\rho_u$", fontsize=14)
    #ax1.set_xlabel("Intensity", fontsize=14)
    #ax1.legend()

    #ax2.plot(rhou_arr, Hh, "o-", label=r"$H_h$")
    #ax2.plot(rhou_arr, Vh, "o-", label=r"$V_h$")
    #ax2.set_xlabel(r"$\rho_u$", fontsize=14)
    #ax2.set_xlabel("Intensity", fontsize=14)
    #ax2.legend()
    
    plt.tight_layout()

    plt.savefig("depolarization.pdf")

    plt.show()
    
    


