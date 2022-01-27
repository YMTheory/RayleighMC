import numpy as np
import matplotlib.pyplot as plt

import ROOT

def func(p0, p1, p2, x) :
    return p0 + p1*np.cos(x-p2)*np.cos(x-p2)

if __name__ == "__main__" :

    vec_pol_angle = [179, 209, 239, 269, 299, 329, 359, 29, 59, 89, 119, 149]

    color = ["blue", "red", "black", "green", "darkviolet", "orange", "gray"]

    theta = [75, 80, 85, 90, 95, 100, 105]

    rhov = []
    rhov_err = []

    Vv, Vverr = [], []
    Hv, Hverr = [], []

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    for n, i in enumerate(theta):

        data = np.loadtxt("./exp/ver"+str(i)+".txt")

        R    = data[0:12, 0]
        Rerr = data[0:12, 1]

        g1 = ROOT.TGraphErrors()
        for k in range(len(vec_pol_angle)):
            g1.SetPoint(k, vec_pol_angle[k]/180.*np.pi, R[k])
            g1.SetPointError(k, 0, Rerr[k])

        f1 = ROOT.TF1("f1", "[0] + [1]*cos(x-[2])*cos(x-[2])", 0, np.pi*2)
        g1.Fit(f1, "RE")
        p0, p1, p2 = f1.GetParameter(0), f1.GetParameter(1), f1.GetParameter(2)
        p0err, p1err, p2err = f1.GetParError(0), f1.GetParError(1), f1.GetParError(2)

        Vv.append(func(p0, p1, p2, p2))
        Vvmax = func(p0+p0err, p1+p1err, p2, p2)
        Vvmin = func(p0-p0err, p1-p1err, p2, p2)
        Vverr.append((Vvmax-Vvmin)/2.)

        Hv.append(func(p0, p1, p2, p2+np.pi/2.))
        Hvmax = func(p0+p0err, p1+p1err, p2, p2+np.pi/2.)
        Hvmin = func(p0-p0err, p1-p1err, p2, p2+np.pi/2.)
        Hverr.append((Hvmax-Hvmin)/2.)

        rhov.append(f1.Eval(f1.GetParameter(2)+np.pi/2) / f1.Eval(f1.GetParameter(2)))
        rhov_min = func(p0-p0err, p1-p1err, p2, p2+np.pi/2) / func(p0+p0err, p1+p1err, p2, p2)
        rhov_max = func(p0+p0err, p1+p1err, p2, p2+np.pi/2) / func(p0-p0err, p1-p1err, p2, p2)
        rhov_err.append((rhov_max - rhov_min)/2.)

        dx = np.arange(0, np.pi*2, 0.01)
        dy = []
        for j in dx:
            dy.append(f1.Eval(j))


        ax1.errorbar(vec_pol_angle, R, yerr=Rerr, fmt="o", color=color[n])
        ax1.plot(dx*180/np.pi, dy, "-", color=color[n], label=r"$\theta$=%d deg"%i)

    ax1.legend()
    ax1.set_xlabel("Polarizer degree", fontsize=14)
    ax1.set_ylabel("Scaled intensity (I/I0)", fontsize=14)
    
    ax2.errorbar(theta, rhov, yerr=rhov_err, fmt="o-")
    ax2.set_xlabel(r"$\theta$ [deg]", fontsize=14)
    ax2.set_ylabel(r"$\rho_v$", fontsize=14)
    ax2.grid(True)

    ax3.errorbar(theta, Vv, yerr=Vverr, fmt="o-", label=r"$V_v$")
    ax3.errorbar(theta, Hv, yerr=Hverr, fmt="o-", label=r"$H_v$")
    ax3.set_xlabel(r"$\theta$ [deg]", fontsize=14)
    ax3.set_ylabel(r"Scaled intensity", fontsize=14)
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.savefig("ver.pdf")

    plt.show()
