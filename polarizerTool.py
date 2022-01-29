import numpy as np
from Rayleigh_class import Rayleigh
from detector import detector
from tqdm import tqdm


def generator(px, py, pz, rhov, tt):
    ## detector :
    det = detector()

    ## Rayleigh process:
    ray = Rayleigh()


    # input configuration
    ray.set_inMom(0, 0, 1)     # incident light momentum
    ray.set_inPol(px, py, pz)     # incident light polarisation
    
    # sef depolarisation ratio
    ray.set_rhov(rhov)      # LAB case
    ray.calcTensor()

    #### output depolarisation related properties ############
    print("======================================")
    print(r"$\rho_v$ = %.3f" %ray.get_rhov())
    print(r"$\rho_u$ = %.3f" %ray.get_rhou())
    print(r"$\alpha$ = %.3f, $\beta$ = %.3f" %(ray.get_alpha(), ray.get_beta()))
    print("======================================")


    Nsample = 10000

    polAngle_arr = np.arange(0, 361, 1)

    prob_arr = []

    print("Detecting at theta = %d degree" %tt)

    ray.set_outMomTheta(tt/180*np.pi)
    ray.set_outMomPhi(3*np.pi/2)
    ray.set_outMom(ray.get_inE()*np.cos(ray.get_outMomPhi())*np.sin(ray.get_outMomTheta()), ray.get_inE()*np.sin(ray.get_outMomPhi())*np.sin(ray.get_outMomTheta()), ray.get_inE()*np.cos(ray.get_outMomTheta()) )

    outPol_arr = [ [] for i in range(Nsample)]
    amp_arr = []
    for ii in tqdm(range(Nsample)):
        ray.set_inPol(px, py, pz)
        #ray.rotate_inPol()
        #ray.calculatePol()
        ray.rotate_inPol_twice()
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

        
    print("Vertical incident light: %.2f, %.2f" %(prob_arr[0], prob_arr[90]) )

    return polAngle_arr/180.*np.pi, prob_arr


