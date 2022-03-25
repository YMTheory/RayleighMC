import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple



if __name__ == "__main__" :

    MC_costheta_unp_rho2_x, MC_costheta_unp_rho2_y, MC_costheta_unp_rho2_yerr = [], [], []
    MC_costheta_unp_rho0_x, MC_costheta_unp_rho0_y, MC_costheta_unp_rho0_yerr = [], [], []
    theo_costheta_unp_rho2_x, theo_costheta_unp_rho2_y = [], []
    theo_costheta_unp_rho0_x, theo_costheta_unp_rho0_y = [], []

    MC_phi_unp_rho2_x, MC_phi_unp_rho2_y, MC_phi_unp_rho2_yerr = [], [], []
    MC_phi_unp_rho0_x, MC_phi_unp_rho0_y, MC_phi_unp_rho0_yerr = [], [], []
    theo_phi_unp_rho2_x, theo_phi_unp_rho2_y = [], []
    theo_phi_unp_rho0_x, theo_phi_unp_rho0_y = [], []

    MC_phi_ver_rho2_x, MC_phi_ver_rho2_y, MC_phi_ver_rho2_yerr = [], [], []
    MC_phi_ver_rho0_x, MC_phi_ver_rho0_y, MC_phi_ver_rho0_yerr = [], [], []
    theo_phi_ver_rho2_x, theo_phi_ver_rho2_y = [], []
    theo_phi_ver_rho0_x, theo_phi_ver_rho0_y = [], []

    MC_phi_hor_rho2_x, MC_phi_hor_rho2_y, MC_phi_hor_rho2_yerr = [], [], []
    MC_phi_hor_rho0_x, MC_phi_hor_rho0_y, MC_phi_hor_rho0_yerr = [], [], []
    theo_phi_hor_rho2_x, theo_phi_hor_rho2_y = [], []
    theo_phi_hor_rho0_x, theo_phi_hor_rho0_y = [], []
    
    with open("data_1D.txt") as f:

        
        for lines in f.readlines():
            line = lines.strip("\n")
            data = line.split(" ")

            if data[0] == "MCunp2Theta":
                MC_costheta_unp_rho2_x.append(float(data[1]))
                MC_costheta_unp_rho2_y.append(float(data[2]))
                MC_costheta_unp_rho2_yerr.append(float(data[3]))

            if data[0] == "MCver0Theta":
                MC_costheta_unp_rho0_x.append(float(data[1]))
                MC_costheta_unp_rho0_y.append(float(data[2]))
                MC_costheta_unp_rho0_yerr.append(float(data[3]))

            if data[0] == "Theo2Theta":
                theo_costheta_unp_rho2_x.append(float(data[1]))
                theo_costheta_unp_rho2_y.append(float(data[2]))

            if data[0] == "Theo0Theta":
                theo_costheta_unp_rho0_x.append(float(data[1]))
                theo_costheta_unp_rho0_y.append(float(data[2]))

            if data[0] == "MCver2Phi" :
                MC_phi_ver_rho2_x.append(float(data[1]))
                MC_phi_ver_rho2_y.append(float(data[2]))
                MC_phi_ver_rho2_yerr.append(float(data[3]))

            if data[0] == "MCver0Phi" :
                MC_phi_ver_rho0_x.append(float(data[1]))
                MC_phi_ver_rho0_y.append(float(data[2]))
                MC_phi_ver_rho0_yerr.append(float(data[3]))


            if data[0] == "Theo2verPhi":
                theo_phi_ver_rho2_x.append(float(data[1]))
                theo_phi_ver_rho2_y.append(float(data[2]))
    

            if data[0] == "Theo0verPhi":
                theo_phi_ver_rho0_x.append(float(data[1]))
                theo_phi_ver_rho0_y.append(float(data[2]))



            if data[0] == "MChor2Phi" :
                MC_phi_hor_rho2_x.append(float(data[1]))
                MC_phi_hor_rho2_y.append(float(data[2]))
                MC_phi_hor_rho2_yerr.append(float(data[3]))

            if data[0] == "MChor0Phi" :
                MC_phi_hor_rho0_x.append(float(data[1]))
                MC_phi_hor_rho0_y.append(float(data[2]))
                MC_phi_hor_rho0_yerr.append(float(data[3]))


            if data[0] == "Theo2horPhi":
                theo_phi_hor_rho2_x.append(float(data[1]))
                theo_phi_hor_rho2_y.append(float(data[2]))
    

            if data[0] == "Theo0horPhi":
                theo_phi_hor_rho0_x.append(float(data[1]))
                theo_phi_hor_rho0_y.append(float(data[2]))

            if data[0] == "MCunp2Phi" :
                MC_phi_unp_rho2_x.append(float(data[1]))
                MC_phi_unp_rho2_y.append(float(data[2]))
                MC_phi_unp_rho2_yerr.append(float(data[3]))

            if data[0] == "MCunp0Phi" :
                MC_phi_unp_rho0_x.append(float(data[1]))
                MC_phi_unp_rho0_y.append(float(data[2]))
                MC_phi_unp_rho0_yerr.append(float(data[3]))


            if data[0] == "TheounpPhi":
                theo_phi_unp_rho2_x.append(float(data[1]))
                theo_phi_unp_rho2_y.append(float(data[2]))
    

            if data[0] == "TheounpPhi":
                theo_phi_unp_rho0_x.append(float(data[1]))
                theo_phi_unp_rho0_y.append(float(data[2]))





        fig = plt.figure(dpi=100,
                         constrained_layout=True,
                         figsize=(12, 6))
        gs = GridSpec(6, 2, figure=fig, width_ratios=[1., 1])#GridSpec将fiure分为3行3列，每行三个axes，gs为一个matplotlib.gridspec.GridSpec对象，可灵活的切片figure

    ax0 = fig.add_subplot(gs[0:4, 0])



    d1 = ax0.errorbar(MC_costheta_unp_rho2_x, MC_costheta_unp_rho2_y, yerr=MC_costheta_unp_rho2_yerr, capsize=2, fmt="o", color="blue", ms=4, label="MC")
    d2 = ax0.errorbar(MC_costheta_unp_rho0_x, MC_costheta_unp_rho0_y, yerr=MC_costheta_unp_rho0_yerr, capsize=2, fmt="o", fillstyle="none", color="red", ms=6)

    l1, = ax0.plot(theo_costheta_unp_rho2_x, theo_costheta_unp_rho2_y, "-", lw=1.5, color="black", label="Theory")
    l2, = ax0.plot(theo_costheta_unp_rho0_x, theo_costheta_unp_rho0_y, "--", lw=1.5, color="black")

    ax0.set_title("(a)", fontsize=13)
    ax0.set_xlabel(r"cos$\theta$", fontsize=13)
    ax0.set_ylabel(r"$N_s(\theta)/N_s(\theta=90^\circ)$", fontsize=13)

    p1 = ax0.legend(title=r"$\rho_v = 0.2$, unpolarized", title_fontsize=13, prop={"size":13})
    p2 = ax0.legend([l2, d2], ["Theory", "MC"], loc='center', title=r"$\rho_v = 0$, unpolarized", title_fontsize=13, prop={"size":13})
    plt.gca().add_artist(p1)
    plt.gca().add_artist(p2)


    ax3 = fig.add_subplot(gs[4:6, 0])
    ax1 = fig.add_subplot(gs[0:3, 1])
    ax2 = fig.add_subplot(gs[3:6, 1])
    xtick = [0, np.pi/2, np.pi, np.pi*3/2, np.pi*2]
    xticknames = ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]

    d1 = ax1.errorbar(MC_phi_ver_rho2_x, MC_phi_ver_rho2_y, yerr=MC_phi_ver_rho2_yerr, fmt="^", color="blue", capsize=2, ms=4)
    d2 = ax1.errorbar(MC_phi_ver_rho0_x, MC_phi_ver_rho0_y, yerr=MC_phi_ver_rho0_yerr, fmt="^", color="red", capsize=2, ms=6, fillstyle="none")
    l1, = ax1.plot(theo_phi_ver_rho2_x, theo_phi_ver_rho2_y, "-", lw=1.5, color="black")
    l2, = ax1.plot(theo_phi_ver_rho0_x, theo_phi_ver_rho0_y, "--", lw=1.5, color="black")

    ax1.set_title("(b)", fontsize=13)
    ax1.set_xlabel(r"$\phi$ [rad]", fontsize=13)
    ax1.set_ylabel(r"$N_s(\phi)/N_s(\phi=90^\circ)$", fontsize=13)
    ax1.set_xticks(xtick)
    ax1.set_xticklabels(xticknames, fontsize=13)
    ax1.legend([(d1, d2)], ["MC: vertical"], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})


    d1 = ax2.errorbar(MC_phi_hor_rho2_x, MC_phi_hor_rho2_y, yerr=MC_phi_hor_rho2_yerr, fmt=">", color="blue", capsize=2, ms=4)
    d2 = ax2.errorbar(MC_phi_hor_rho0_x, MC_phi_hor_rho0_y, yerr=MC_phi_hor_rho0_yerr, fmt=">", color="red", capsize=2, ms=6, fillstyle="none")
    l1, = ax2.plot(theo_phi_hor_rho2_x, theo_phi_hor_rho2_y, "-", lw=1.5, color="black")
    l2, = ax2.plot(theo_phi_hor_rho0_x, theo_phi_hor_rho0_y, "--", lw=1.5, color="black")
    ax2.set_title("(c)", fontsize=13)
    ax2.set_xlabel(r"$\phi$ [rad]", fontsize=13)
    ax2.set_ylabel(r"$N_s(\phi)/N_s(\phi=90^\circ)$", fontsize=13)
    ax2.set_xticks(xtick)
    ax2.set_xticklabels(xticknames, fontsize=13)
    ax2.legend([(d1, d2)], ["MC: horizontal"], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})



    d1 = ax3.errorbar(MC_phi_unp_rho2_x, MC_phi_unp_rho2_y, yerr=MC_phi_unp_rho2_yerr, fmt="o", color="blue", capsize=2, ms=4)
    d2 = ax3.errorbar(MC_phi_unp_rho0_x, MC_phi_unp_rho0_y, yerr=MC_phi_unp_rho0_yerr, fmt="o", color="red", capsize=2, ms=6, fillstyle="none")
    l1, = ax3.plot(theo_phi_unp_rho2_x, theo_phi_unp_rho2_y, "-", lw=1.5, color="black")
    l2, = ax3.plot(theo_phi_unp_rho0_x, theo_phi_unp_rho0_y, "--", lw=1.5, color="black")
    ax3.set_title("(d)", fontsize=13)
    ax3.set_xlabel(r"$\phi$ [rad]", fontsize=13)
    ax3.set_ylabel(r"$N_s(\phi)/N_s(\phi=90^\circ)$", fontsize=13)
    ax3.set_xticks(xtick)
    ax3.set_xticklabels(xticknames, fontsize=13)
    ax3.set_ylim(0.99, 1.01)
    ax3.legend([(d1, d2)], ["MC: unpolarized"], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})



    plt.tight_layout()
    plt.savefig("revised_1D.pdf")

    plt.show()
    


