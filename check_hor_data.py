import numpy as np
import matplotlib.pyplot as plt

import ROOT

def func(p0, p1, p2, x) :
    return p0 + p1*np.cos(x-p2)*np.cos(x-p2)

if __name__ == "__main__" :

    vec_pol_angle = [179, 209, 239, 269, 299, 329, 359, 29, 59, 89, 119, 149]   # hor_1.txt
    vec_pol_angle1 = [22, 67, 112, 157, 202, 247, 292, 337]   # hor_2.txt
    color = ["blue", "red", "black", "green", "darkviolet", "orange", "gray"]

    theta = [75, 80, 85, 90, 95, 100, 105]

    rhoh = []
    rhoh_err = []

    Vh, Vherr = [], []
    Hh, Hherr = [], []

    rhoh1 = []
    rhoh_err1 = []

    Vh1, Vherr1 = [], []
    Hh1, Hherr1 = [], []

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    for n, i in enumerate(theta):

        data = np.loadtxt("./exp/hor"+str(i)+"_1.txt")
        data1 = np.loadtxt("./exp/hor"+str(i)+"_2.txt")

        R    = data[0:12, 0]
        Rerr = data[0:12, 1]
        R1    = data[:, 0]
        Rerr1 = data[:, 1]

        g1 = ROOT.TGraphErrors()
        for k in range(len(vec_pol_angle)):
            g1.SetPoint(k, vec_pol_angle[k]/180.*np.pi, R[k])
            g1.SetPointError(k, 0, Rerr[k])

        f1 = ROOT.TF1("f1", "[0] + [1]*cos(x-[2])*cos(x-[2])", 0, np.pi*2)
        g1.Fit(f1, "RE")
        p0, p1, p2 = f1.GetParameter(0), f1.GetParameter(1), f1.GetParameter(2)
        p0err, p1err, p2err = f1.GetParError(0), f1.GetParError(1), f1.GetParError(2)

        if p1 < 0:
            Vh.append(p0+p1)
            Vhmax = p0 + p1 + p0err + p1err
            Vhmin = p0 + p1 - p0err - p1err
            Vherr.append((Vhmax-Vhmin)/2.)

            Hh.append(p0)
            Hhmax = p0 + p0err
            Hhmin = p0 - p0err

        g1 = ROOT.TGraphErrors()
        for k in range(len(vec_pol_angle)):
            g1.SetPoint(k, vec_pol_angle[k]/180.*np.pi, R[k])
            g1.SetPointError(k, 0, Rerr[k])

        f1 = ROOT.TF1("f1", "[0] + [1]*cos(x-[2])*cos(x-[2])", 0, np.pi*2)
        g1.Fit(f1, "RE")
        p0, p1, p2 = f1.GetParameter(0), f1.GetParameter(1), f1.GetParameter(2)
        p0err, p1err, p2err = f1.GetParError(0), f1.GetParError(1), f1.GetParError(2)

        if p1 < 0:
            Vh.append(p0+p1)
            Vhmax = p0 + p1 + p0err + p1err
            Vhmin = p0 + p1 - p0err - p1err
            Vherr.append((Vhmax-Vhmin)/2.)

            Hh.append(p0)
            Hhmax = p0 + p0err
            Hhmin = p0 - p0err
            Hherr.append((Hhmax-Hhmin)/2.)

            print("Vh = %.3f, Hh = %.3f" %(Vh[-1], Hh[-1]))
    
        if p1 >= 0:
            Hh.append(p0+p1)
            Hhmax = p0 + p1 + p0err + p1err
            Hhmin = p0 + p1 - p0err - p1err
            Hherr.append((Hhmax-Hhmin)/2.)

            Vh.append(p0)
            Vhmax = p0 + p0err
            Vhmin = p0 - p0err
            Vherr.append((Vhmax-Vhmin)/2.)

            print("Vh = %.3f, Hh = %.3f" %(Vh[-1], Hh[-1]))

        rhoh.append(Hh[-1]/Vh[-1])
        rhoh_max = Hhmax / Vhmin
        rhoh_min = Hhmin / Vhmax
        rhoh_err.append((rhoh_max - rhoh_min)/2.)

        dx = np.arange(0, np.pi*2, 0.01)
        dy = []
        for j in dx:
            dy.append(f1.Eval(j))


        ax1.errorbar(vec_pol_angle, R, yerr=Rerr, fmt="o", color=color[n])
        ax1.plot(dx*180/np.pi, dy, "-", color=color[n], label=r"$\theta$=%d deg"%i)

    ax1.legend()
    ax1.set_xlabel("Polarizer degree", fontsize=14)
    ax1.set_ylabel("Scaled intensity (I/I0)", fontsize=14)
    
    ax2.errorbar(theta, rhoh, yerr=rhoh_err, fmt="o-")
    ax2.set_xlabel(r"$\theta$ [deg]", fontsize=14)
    ax2.set_ylabel(r"$\rho_h$", fontsize=14)
    ax2.grid(True)

    ax3.errorbar(theta, Vh, yerr=Vherr, fmt="o-", label=r"$V_h$")
    ax3.errorbar(theta, Hh, yerr=Hherr, fmt="o-", label=r"$H_h$")
    ax3.set_xlabel(r"$\theta$ [deg]", fontsize=14)
    ax3.set_ylabel(r"Scaled intensity", fontsize=14)
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    #plt.savefig("ver.pdf")

    plt.show()
