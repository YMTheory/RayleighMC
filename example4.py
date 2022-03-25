import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ROOT
from tqdm import tqdm

from Rayleigh_class import Rayleigh
from detector import detector

def func(p0, p1, p2, x):
    f1 = ROOT.TF1("f1", "[0] + [1]*cos(x-[2])*cos(x-[2])", 0, 2*np.pi)
    f1.SetParameters(p0, p1, p2)
    return f1.Eval(x)


def polarization_fit(x, y, yerr):
    g1 = ROOT.TGraphErrors()
    for i in range(len(x)):
        g1.SetPoint(i, x[i], y[i])
        g1.SetPointError(i, 0, yerr[i])

    f1 = ROOT.TF1("f1", "[0] + [1]*cos(x-[2])*cos(x-[2])", 0, 2*np.pi)
    g1.Fit(f1, "RE")

    return f1.GetParameter(0), f1.GetParameter(1), f1.GetParameter(2), f1.GetParError(0), f1.GetParError(1), f1.GetParError(2)

def line_fit(x, y, yerr):
    g1 = ROOT.TGraphErrors()
    for i in range(len(x)):
        g1.SetPoint(i, x[i], y[i])
        g1.SetPointError(i, 0, yerr[i])

    f1 = ROOT.TF1("f1", "[0] + 0*x", 70, 110)
    g1.Fit(f1, "REQ")

    return f1.GetParameter(0)

def simulation(polx, poly, polz):

    ray = Rayleigh()
    det = detector()

    ray.set_rhov(0.18)
    ray.calcTensor()

    ray.set_inMom(0, 0, 1)
    ray.set_inPol(polx, poly, polz)

    Nsample = 10000

    polAngle_arr = np.arange(0, 361, 1)

    theta_start, theta_stop, theta_step = 75, 110, 5.
    Nstep = int((theta_stop - theta_start) / theta_step)
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

            prob_arr[count].append(amp)

        print("Vertical incident light: %.2f, %.2f" %(prob_arr[count][0], prob_arr[count][90]) )
        count += 1
        
    return prob_arr




if __name__ == "__main__" :

    # vertical incident
    data = np.loadtxt("./exp/vertical0518.txt")

    theta = [75, 80, 85, 90, 95, 100, 105]

    det  = data[np.where(data[:, 0]==90)][:, 1] / 180. * np.pi

    data_ver_angcorr, data_ver_V, data_ver_H = [], [], []
    data_ver_R, data_ver_Rerr = [], []
    data_ver_Vmax, data_ver_Vmin, data_ver_Hmax, data_ver_Hmin = [], [], [], []

    for i in theta:
        R     = data[np.where(data[:, 0]==i)][:, 2]
        Rerr  = data[np.where(data[:, 0]==i)][:, 3]
        data_ver_R.append(R)
        data_ver_Rerr.append(Rerr)

    for i, j in zip(data_ver_R, data_ver_Rerr):
        p0, p1, p2, p0err, p1err, p2err = polarization_fit(det, i, j)
        data_ver_angcorr.append(p2)
        data_ver_V.append(func(p0, p1, p2, p2))
        data_ver_Vmin.append(func(p0-p0err, p1-p1err, p2-p2err, p2))
        data_ver_Vmax.append(func(p0+p0err, p1+p1err, p2, p2))
        data_ver_H.append(func(p0, p1, p2, p2+np.pi/2))
        data_ver_Hmin.append(func(p0-p0err, p1-p1err, p2-p2err, p2+np.pi/2))
        data_ver_Hmax.append(func(p0+p0err, p1+p1err, p2, p2+np.pi/2))

    data_ver_V = np.array(data_ver_V)
    data_ver_Vmin = np.array(data_ver_Vmin)
    data_ver_Vmax = np.array(data_ver_Vmax)
    data_ver_H = np.array(data_ver_H)
    data_ver_Hmin = np.array(data_ver_Hmin)
    data_ver_Hmax = np.array(data_ver_Hmax)

    data_lVv = line_fit(theta, data_ver_V, (data_ver_Vmax - data_ver_Vmin)/2.)
    data_lHv = line_fit(theta, data_ver_H, (data_ver_Hmax - data_ver_Hmin)/2.)

    data_norm = 1./ data_lHv
    print(data_lHv, data_lVv)
    print("Normalization factor : %.2f" %data_norm)

    ############################################
    ############# Simulation ###################
    ############################################
    prob_arr = simulation(1, 0, 0)

    sim_ver_V, sim_ver_Vmin, sim_ver_Vmax = [], [], []
    sim_ver_H, sim_ver_Hmin, sim_ver_Hmax = [], [], []

    polAng = np.arange(0, 361, 1) / 180*np.pi
    for i in range(len(prob_arr)):
        #prob = np.array(prob_arr[i])
        #p0, p1, p2, p0err, p1err, p2err = polarization_fit(polAng, prob, np.sqrt(prob))
        #sim_ver_V.append(func(p0, p1, p2, p2))
        #sim_ver_Vmin.append(func(p0-p0err, p1-p1err, p2-p2err, p2))
        #sim_ver_Vmax.append(func(p0+p0err, p1+p1err, p2, p2))
        #sim_ver_H.append(func(p0, p1, p2, p2+np.pi/2))
        #sim_ver_Hmin.append(func(p0-p0err, p1-p1err, p2-p2err, p2+np.pi/2))
        #sim_ver_Hmax.append(func(p0+p0err, p1+p1err, p2, p2+np.pi/2))
        print(prob_arr[i][90])
        sim_ver_V.append((prob_arr[i][90]+prob_arr[i][270])/2.)
        sim_ver_Vmin.append(sim_ver_V[-1] - (np.sqrt(prob_arr[i][90])+np.sqrt(prob_arr[i][270]))/2.)
        sim_ver_Vmax.append(sim_ver_V[-1] + (np.sqrt(prob_arr[i][90])+np.sqrt(prob_arr[i][270]))/2.)
        sim_ver_H.append((prob_arr[i][0]+prob_arr[i][180])/2.)
        sim_ver_Hmin.append(sim_ver_H[-1] - (np.sqrt(prob_arr[i][0])+np.sqrt(prob_arr[i][180]))/2.)
        sim_ver_Hmax.append(sim_ver_H[-1] + (np.sqrt(prob_arr[i][0])+np.sqrt(prob_arr[i][180]))/2.)

    sim_ver_V = np.array(sim_ver_V)
    sim_ver_Vmin = np.array(sim_ver_Vmin)
    sim_ver_Vmax = np.array(sim_ver_Vmax)
    print(sim_ver_V)
    print(sim_ver_Vmin)
    print(sim_ver_Vmax)
    sim_ver_H = np.array(sim_ver_H)
    sim_ver_Hmin = np.array(sim_ver_Hmin)
    sim_ver_Hmax = np.array(sim_ver_Hmax)


    sim_lVv = line_fit(theta, sim_ver_V, (sim_ver_Vmax - sim_ver_Vmin)/2.)
    sim_lHv = line_fit(theta, sim_ver_H, (sim_ver_Hmax - sim_ver_Hmin)/2.)
    print(sim_lHv, sim_lVv)

    sim_norm = 1./ sim_lHv
    print("Normalization factor : %.2f" %sim_norm)


    # horizontal incident 
    #data = np.loadtxt("./exp/horizontal.txt")

    #det  = data[np.where(data[:, 0]==90)][:, 1] / 180. * np.pi

    #hor_angcorr, hor_V, hor_H = [], [], []
    #hor_R, hor_Rerr = [], []

    #for i in theta:
    #    R     = data[np.where(data[:, 0]==i)][:, 2]
    #    Rerr  = data[np.where(data[:, 0]==i)][:, 3]
    #    hor_R.append(R)
    #    hor_Rerr.append(Rerr)

    #for i, j in zip(hor_R, hor_Rerr):
    #    p0, p1, p2 = polarization_fit(det, i, j)
    #    hor_angcorr.append(p2)
    #    hor_V.append(func(p0, p1, p2, p2))
    #    hor_H.append(func(p0, p1, p2, p2+np.pi/2))


    fig, ax = plt.subplots()

    ax.plot(theta, data_ver_V*data_norm, "o-", color="blue", label=r"Exp: $V_v$")
    ax.fill_between(theta, data_ver_Vmin*data_norm, data_ver_Vmax*data_norm, color="royalblue", alpha=0.3)
    ax.plot(theta, data_ver_H*data_norm, "*--", color="orange", label=r"Exp: $V_h$") 
    ax.fill_between(theta, data_ver_Hmin*data_norm, data_ver_Hmax*data_norm, color="orange", alpha=0.3)

    ax.plot(theta, sim_ver_V*sim_norm, "-", color="black", label=r"Sim: $V_v$")
    ax.fill_between(theta, sim_ver_Vmin*sim_norm, sim_ver_Vmax*sim_norm, color="slategray", alpha=0.3)
    ax.plot(theta, sim_ver_H*sim_norm, "--", color="black", label=r"Sim: $V_h$") 
    ax.fill_between(theta, sim_ver_Hmin*sim_norm, sim_ver_Hmax*sim_norm, color="slategray", alpha=0.3)


    #ax.plot(theta, hor_V, "o-", label=r"$H_v$")
    #ax.plot(theta, hor_H, "*-", label=r"$H_h$") 

    ax.legend(prop={"size":15})
    ax.set_xlabel(r"$\theta$", fontsize=15)
    ax.set_ylabel(r"Normalized intensity", fontsize=15)
    ax.grid(True)

    plt.tight_layout()
    plt.show()
