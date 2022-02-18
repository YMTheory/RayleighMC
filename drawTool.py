import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm

import ROOT

from Rayleigh_class import Rayleigh
from detector import detector

import sys, os

#from tqdm import tqdm

import polarizerTool as pt
import analyser as ana
import loader as ld
#import pre_fit as fit


def f2d(x, y, rhov):
    return (1 + (rhov - 1)/(2+2*rhov) * (2*np.cos(y)**2*(1-x**2))) * (3+3*rhov) / (8*np.pi+16*np.pi*rhov)

def func(p0, p1, p2, x) :
    return p0 + p1*np.cos(x-p2)*np.cos(x-p2)


def phi_theory(x, rhov):
    #return (1542.395+308.46*np.cos(2*x)-205.64*np.sin(x)**2)/9045.12
    return (3+3*rhov) / (4*np.pi+8*np.pi*rhov) *(1 - (2-2*rhov)/(3+3*rhov)*np.cos(x)**2)


def theta_theory(x, rhov):
    rhou = 2*rhov / (1 + rhov)
    scale = 1 / (2+2./3*(1-rhou)/(1+rhou))
    #return 1 + (1-rhou)/(1+rhou) * np.cos(x)**2
    return scale*(1 + (1-rhou)/(1+rhou) * np.cos(x)**2)


if __name__ == "__main__" :
    
    drawPolar = False
    drawHist1D = True
    drawHist2D = False
    drawHist3D = False
    drawRhoPolar = False
    drawCompare = False

    theta = [75, 80, 85, 90, 95, 100, 105]

    color = ["blue", "red", "black", "green", "darkviolet", "orange", "gray"]

    #if drawRhoPolar:
    #    fig = plt.figure(dpi=100,
    #                     constrained_layout=True,
    #                     figsize=(12, 7))
    #    gs = GridSpec(3, 3, figure=fig, height_ratios=[1., 2., 0.3])

    #    ax = fig.add_subplot(gs[0, 0:3])
    #    data_x, y1, y2, rhov, rhov_err, rhoh, rhoh_err, fval, lens, nfit, mparameters, mvalues, merrors = fit.fitter()
    #    ax.errorbar(theta, rhov, yerr=rhov_err, fmt="^", color="red", label=r"$\rho_v$")

    #    ax.plot(data_x*180./np.pi, y1, color="black", label="Best fit")

    #    ax.errorbar(theta, rhoh, yerr=rhoh_err, fmt="v", color="red", label=r"$\rho_h$")
    #    ax.plot(data_x*180./np.pi, y2, color="black")
    #    ax.set_xlabel(r"$\theta$ [deg]" , fontsize=14)
    #    ax.set_ylabel("Intensity ratio", fontsize=14)
    #    #ax.set_title("(a)", fontsize=14, labelpad=-10)

    #    # display legend with some fit info
    #    fit_info = [
    #        f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {fval:.1f} / {lens - nfit}",
    #    ]
    #    for p, v, e in zip(mparameters, mvalues, merrors):
    #        fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")
    #    textstr = "\n".join(fit_info)
    #    ax.text(70, 0.5, textstr, fontsize=14, bbox=dict(boxstyle="round", ec="black", fc="white"))
    #    ax.grid(True)
    #    ax.legend(loc="center right", prop={"size":14}, ncol=3);

    #    ax1 = fig.add_subplot(gs[1, 0], projection="polar")
    #    ang = 75.
    #    vec_pol_angle = np.array([179, 209, 239, 269, 299, 329, 359, 29, 59, 89, 119, 149])
    #    R, Rerr, p0, p1, p2, p0err, p1err, p2err = ana.read_data(ang, "ver")
    #    dx = np.arange(0, 2*np.pi, 0.01)
    #    dy = []
    #    for i in dx:
    #        dy.append(func(p0, p1, 0, i))
    #    dy = np.array(dy)
    #    ## simulation
    #    polAngle_arr, prob_arr = pt.generator(1, 0, 0, 0.208, ang)
    #    norm1 = (prob_arr[90] + prob_arr[270]) / 2.
    #    ax1.errorbar(vec_pol_angle/180.*np.pi-p2, R/p0, yerr=Rerr/p0, fmt="^", ms=7, fillstyle="none", color="red", label="Exp: vertical")
    #    ax1.plot(polAngle_arr, prob_arr/norm1, "-", lw=2, color="blue", label="MC: vertical")
    #    # horizontal
    #    R, Rerr, p0, p1, p2, p0err, p1err, p2err = ana.read_data(ang, "hor")
    #    dx = np.arange(0, 2*np.pi, 0.01)
    #    dy = []
    #    for i in dx:
    #        dy.append(func(p0, p1, 0, i))
    #    dy = np.array(dy)
    #    ## simulation
    #    polAngle_arr, prob_arr = pt.generator(0, 1, 0, 0.208, ang)
    #    norm2 = (prob_arr[0] + prob_arr[180]) / 2.
    #    print("MC normalization factor = %.3e" %norm2)
    #    ax1.errorbar(vec_pol_angle/180.*np.pi-p2, R/(p0+p1), yerr=Rerr/(p0+p1), fmt="v", ms=5, fillstyle="none", color="red", label="Exp: horizontal")
    #    ax1.plot(polAngle_arr, prob_arr/norm2, "-", lw=2, color="blue", label="MC: horizontal")
    #    ax1.set_theta_zero_location("N", np.pi/2.)
    #    ax1.text(200, 6, r"$\theta=%d$ deg"%ang, bbox=dict(facecolor='white', alpha=0.5), fontsize=14)
    #    #ax1.set_title("(b)", fontsize=14, labelpad=-10)


    #    ax2 = fig.add_subplot(gs[1, 1], projection="polar")
    #    ang = 90.
    #    vec_pol_angle = np.array([179, 209, 239, 269, 299, 329, 359, 29, 59, 89, 119, 149])
    #    R, Rerr, p0, p1, p2, p0err, p1err, p2err = ana.read_data(ang, "ver")
    #    dx = np.arange(0, 2*np.pi, 0.01)
    #    dy = []
    #    for i in dx:
    #        dy.append(func(p0, p1, 0, i))
    #    dy = np.array(dy)
    #    ## simulation
    #    polAngle_arr, prob_arr = pt.generator(1, 0, 0, 0.208, ang)
    #    norm1 = (prob_arr[90] + prob_arr[270]) / 2.
    #    ax2.errorbar(vec_pol_angle/180.*np.pi-p2, R/p0, yerr=Rerr/p0, fmt="^", ms=7, fillstyle="none", color="red", label="Exp: vertical")
    #    ax2.plot(polAngle_arr, prob_arr/norm1, "-", lw=2, color="blue", label="MC: vertical")
    #    # horizontal
    #    R, Rerr, p0, p1, p2, p0err, p1err, p2err = ana.read_data(ang, "hor")
    #    dx = np.arange(0, 2*np.pi, 0.01)
    #    dy = []
    #    for i in dx:
    #        dy.append(func(p0, p1, 0, i))
    #    dy = np.array(dy)
    #    ## simulation
    #    polAngle_arr, prob_arr = pt.generator(0, 1, 0, 0.208, ang)
    #    norm2 = (prob_arr[0] + prob_arr[180]) / 2.
    #    print("MC normalization factor = %.3e" %norm2)
    #    ax2.errorbar(vec_pol_angle/180.*np.pi-p2+np.pi/2, R/(p0), yerr=Rerr/(p0), fmt="v", ms=5, fillstyle="none", color="red", label="Exp: horizontal")
    #    ax2.plot(polAngle_arr, prob_arr/norm2, "-", lw=2, color="blue", label="MC: horizontal")
    #    ax2.set_theta_zero_location("N", np.pi/2.)
    #    ax2.text(200, 6, r"$\theta=%d$ deg"%ang, bbox=dict(facecolor='white', alpha=0.5), fontsize=14)
    #    #ax2.set_title("(c)", fontsize=14, labelpad=-10)


    #    ax3 = fig.add_subplot(gs[1, 2], projection="polar")
    #    ang = 105.
    #    vec_pol_angle = np.array([179, 209, 239, 269, 299, 329, 359, 29, 59, 89, 119, 149])
    #    R, Rerr, p0, p1, p2, p0err, p1err, p2err = ana.read_data(ang, "ver")
    #    dx = np.arange(0, 2*np.pi, 0.01)
    #    dy = []
    #    for i in dx:
    #        dy.append(func(p0, p1, 0, i))
    #    dy = np.array(dy)
    #    ## simulation
    #    polAngle_arr, prob_arr = pt.generator(1, 0, 0, 0.208, ang)
    #    norm1 = (prob_arr[90] + prob_arr[270]) / 2.
    #    ax3.errorbar(vec_pol_angle/180.*np.pi-p2, R/p0, yerr=Rerr/p0, fmt="^", ms=7, fillstyle="none", color="red", label="Exp: vertical")
    #    ax3.plot(polAngle_arr, prob_arr/norm1, "-", lw=2, color="blue", label="MC: vertical")
    #    # horizontal
    #    R, Rerr, p0, p1, p2, p0err, p1err, p2err = ana.read_data(ang, "hor")
    #    dx = np.arange(0, 2*np.pi, 0.01)
    #    dy = []
    #    for i in dx:
    #        dy.append(func(p0, p1, 0, i))
    #    dy = np.array(dy)
    #    ## simulation
    #    polAngle_arr, prob_arr = pt.generator(0, 1, 0, 0.208, ang)
    #    norm2 = (prob_arr[0] + prob_arr[180]) / 2.
    #    print("MC normalization factor = %.3e" %norm2)
    #    ax3.errorbar(vec_pol_angle/180.*np.pi-p2+np.pi/2, R/(p0), yerr=Rerr/(p0), fmt="v", ms=5, fillstyle="none", color="red", label="Exp: horizontal")
    #    ax3.plot(polAngle_arr, prob_arr/norm2, "-", lw=2, color="blue", label="MC: horizontal")
    #    ax3.set_theta_zero_location("N", np.pi/2.)
    #    ax3.text(200, 6, r"$\theta=%d$ deg"%ang, bbox=dict(facecolor='white', alpha=0.5), fontsize=14)
    #    ax3.set_title("(d)", fontsize=14)

    #    ax4 = fig.add_subplot(gs[2, 1])
    #    ax4.plot(1, 1, "^", ms=7, fillstyle="none", color="red", label="Exp: vertical")
    #    ax4.plot(1, 1, "v", ms=5, fillstyle="none", color="red", label="Exp: horizontal")
    #    ax4.plot([1, 1], [2, 2], "-", color="blue", lw=2, label="MC")
    #    ax4.set_xlim(4, 5)
    #    ax4.set_ylim(4, 5)
    #    ax4.legend(loc="upper center", prop={"size":15}, ncol=3)
    #    ax4.get_xaxis().set_visible(False)
    #    ax4.get_yaxis().set_visible(False)
    #    ax4.spines['top'].set_visible(False)
    #    ax4.spines['right'].set_visible(False)
    #    ax4.spines['bottom'].set_visible(False)
    #    ax4.spines['left'].set_visible(False)

    #    plt.tight_layout()
    #    plt.savefig("rho+polar.pdf")
    #    plt.show()



    if drawPolar:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        #### experimental data
        ang = 100.
        vec_pol_angle = np.array([179, 209, 239, 269, 299, 329, 359, 29, 59, 89, 119, 149])
        R, Rerr, p0, p1, p2, p0err, p1err, p2err = ana.read_data(ang, "ver")
        dx = np.arange(0, 2*np.pi, 0.01)
        dy = []
        for i in dx:
            dy.append(func(p0, p1, 0, i))
        dy = np.array(dy)
        ## simulation
        polAngle_arr, prob_arr = pt.generator(1, 0, 0, 0.208, ang)

        norm1 = (prob_arr[90] + prob_arr[270]) / 2.
        print("MC normalization factor = %.3e" %norm1)

        #ax.plot(dx, dy/p0, "--", lw=2, color="black", label="Fitting")
        ax.errorbar(vec_pol_angle/180.*np.pi-p2, R/p0, yerr=Rerr/p0, fmt="^", ms=7, fillstyle="none", color="red", label="Exp: vertical")
        ax.plot(polAngle_arr, prob_arr/norm1, "-", lw=2, color="blue", label="MC: vertical")

        # horizontal
        R, Rerr, p0, p1, p2, p0err, p1err, p2err = ana.read_data(ang, "hor")
        dx = np.arange(0, 2*np.pi, 0.01)
        dy = []
        for i in dx:
            dy.append(func(p0, p1, 0, i))
        dy = np.array(dy)
        ## simulation
        polAngle_arr, prob_arr = pt.generator(0, 1, 0, 0.208, ang)

        norm2 = (prob_arr[0] + prob_arr[180]) / 2.
        print("MC normalization factor = %.3e" %norm2)

        #ax.plot(dx, dy/p0, "--", lw=2, color="black", label="Fitting")
        ax.errorbar(vec_pol_angle/180.*np.pi-p2+np.pi/2, R/(p0), yerr=Rerr/(p0), fmt="v", ms=7, fillstyle="none", color="red", label="Exp: horizontal")
        ax.plot(polAngle_arr, prob_arr/norm2, "-", lw=2, color="blue", label="MC: horizontal")


        #ax.set_xlabel(r"$\beta$", fontsize=15)
        ax.set_theta_zero_location("N", np.pi/2.)
        #ax.legend(bbox_to_anchor=(0.1, 0.3, 1, 1), ncol=2)
        #ax.set_yticklabels([])

        ax.text(200, 6, r"$\theta=%d$ deg"%ang, bbox=dict(facecolor='white', alpha=0.5), fontsize=15)

        plt.tight_layout()
        plt.savefig("./pdffiles/Polar_"+str(ang)+"deg.pdf")
        plt.show()

    if drawHist1D:
        costheta0, phi0, cosbeta0 = ld.loadMCdata("scat_Natural_inMom001_rhov0-0")
        costheta1, phi1, cosbeta1 = ld.loadMCdata("scat_inPol100_inMom001_rhov0-2")
        costheta2, phi2, cosbeta2 = ld.loadMCdata("scat_inPol010_inMom001_rhov0-2")
        costheta3, phi3, cosbeta3 = ld.loadMCdata("scat_Natural_inMom001_rhov0-2")

        fig = plt.figure(dpi=100,
                         constrained_layout=True,
                         figsize=(12, 6))
        gs = GridSpec(3, 2, figure=fig, width_ratios=[2., 1.5])#GridSpec将fiure分为3行3列，每行三个axes，gs为一个matplotlib.gridspec.GridSpec对象，可灵活的切片figure

        #### dist 1:
        bin_counts0_costheta0, bin_edges0 = np.histogram(costheta0, bins=50, range=(-1, 1))
        bin_err_costheta0 = np.sqrt(bin_counts0_costheta0)
        bin_counts_costheta0, bin_edges_costheta0 = np.histogram(costheta0, bins=50, range=(-1, 1), density=True)
        bin_centres_costheta0 = (bin_edges_costheta0[:-1] + bin_edges_costheta0[1:]) / 2
        bin_scale_costheta0 = bin_counts_costheta0[1] / bin_counts0_costheta0[1]
        bin_counts0_costheta1, bin_edges0 = np.histogram(costheta1, bins=50, range=(-1, 1))
        bin_err_costheta1 = np.sqrt(bin_counts0_costheta1)
        bin_counts_costheta1, bin_edges_costheta1 = np.histogram(costheta1, bins=50, range=(-1, 1), density=True)
        bin_centres_costheta1 = (bin_edges_costheta1[:-1] + bin_edges_costheta1[1:]) / 2
        bin_scale_costheta1 = bin_counts_costheta1[1] / bin_counts0_costheta1[1]
        bin_counts0_costheta2, bin_edges0 = np.histogram(costheta2, bins=50, range=(-1, 1))
        bin_err_costheta2 = np.sqrt(bin_counts0_costheta2)
        bin_counts_costheta2, bin_edges_costheta2 = np.histogram(costheta2, bins=50, range=(-1, 1), density=True)
        bin_centres_costheta2 = (bin_edges_costheta2[:-1] + bin_edges_costheta2[1:]) / 2
        bin_scale_costheta2 = bin_counts_costheta2[1] / bin_counts0_costheta2[1]
        bin_counts0_costheta3, bin_edges0 = np.histogram(costheta3, bins=50, range=(-1, 1))
        bin_err_costheta3 = np.sqrt(bin_counts0_costheta3)
        bin_counts_costheta3, bin_edges_costheta3 = np.histogram(costheta3, bins=50, range=(-1, 1), density=True)
        bin_centres_costheta3 = (bin_edges_costheta3[:-1] + bin_edges_costheta3[1:]) / 2
        bin_scale_costheta3 = bin_counts_costheta3[1] / bin_counts0_costheta3[1]
        
        ax0 = fig.add_subplot(gs[0:3, 0])
        ax0.errorbar(bin_centres_costheta1, bin_counts_costheta1, yerr=bin_err_costheta1*bin_scale_costheta1, fmt='o', ms=3, capsize=2, color="red", label="MC: vertical")
        ax0.errorbar(bin_centres_costheta2, bin_counts_costheta2, yerr=bin_err_costheta2*bin_scale_costheta2, fmt='s', ms=3, capsize=2, color="royalblue", label="MC: horizontal")
        ax0.errorbar(bin_centres_costheta3, bin_counts_costheta3, yerr=bin_err_costheta3*bin_scale_costheta3, fmt='*', ms=3, capsize=2, color="gray", label="MC: unpolarized")
        dx = np.arange(-1, 1, 0.01)
        dy = ld.theta_prediction(0.208, dx)
        ax0.plot(dx, dy, "--", color="black", label="Theory", zorder=3)
        ax0.set_xlabel(r"cos($\theta$)", fontsize=15)
        ax0.set_ylabel(r"Normalized $N_s$", fontsize=15)
        ax0.grid(True)
        l1 = ax0.legend(title=r"$\rho_v = 0.2$", title_fontsize=14, prop={"size":15})

        d1 = ax0.errorbar(bin_centres_costheta0, bin_counts_costheta0, yerr=bin_err_costheta0*bin_scale_costheta0, fmt='*', ms=3, capsize=2, color="peru")
        dx = np.arange(-1, 1, 0.01)
        dy = ld.theta_prediction(0.0, dx)
        d2, = ax0.plot(dx, dy, "-.", color="black", zorder=3)
        l2 = ax0.legend([d1, d2], ["MC: unpolarized", "Theory"], loc="center", title=r"$\rho_v = 0$", title_fontsize=14, prop={"size":15})

        plt.gca().add_artist(l1)
        plt.gca().add_artist(l2)
        

        dx = np.arange(0, 2*np.pi, 0.01)
        bin_counts0_phi1, bin_edges0 = np.histogram(phi1, bins=50, range=(0, 2*np.pi))
        bin_err_phi1 = np.sqrt(bin_counts0_phi1)
        bin_counts_phi1, bin_edges_phi1 = np.histogram(phi1, bins=50, range=(0, 2*np.pi), density=True)
        bin_centres_phi1 = (bin_edges_phi1[:-1] + bin_edges_phi1[1:]) / 2
        bin_scale_phi1 = bin_counts_phi1[1] / bin_counts0_phi1[1]
        ax1 = fig.add_subplot(gs[0, 1])#gs[0, 0:3]中0选取figure的第一行，0:3选取figure第二列和第三列
        #ax1.hist(phi1, bins=50, histtype="step", density=True, color="red", label="MC: horizontal")
        ax1.errorbar(bin_centres_phi1, bin_counts_phi1, yerr=bin_err_phi1*bin_scale_phi1, fmt='o', ms=3, capsize=2, color="red", label="MC: vertical")
        ax1.set_xlabel(r"$\phi$ [rad]", fontsize=15)
        ax1.set_ylabel(r"Normalized $N_s$", fontsize=15)
        dy1 = []
        for i in dx:
            dy1.append(phi_theory(i, 0.20))
        ax1.plot(dx, dy1, "--",  color="black", label="Theory")
        ax1.grid(True)
        #ax1.legend(prop={"size":15})

        ax2 = fig.add_subplot(gs[1, 1])#gs[0, 0:3]中0选取figure的第一行，0:3选取figure第二列和第三列
        bin_counts0_phi2, bin_edges0 = np.histogram(phi2, bins=50, range=(0, 2*np.pi))
        bin_err_phi2 = np.sqrt(bin_counts0_phi2)
        bin_counts_phi2, bin_edges_phi2 = np.histogram(phi2, bins=50, range=(0, 2*np.pi), density=True)
        bin_centres_phi2 = (bin_edges_phi2[:-1] + bin_edges_phi2[1:]) / 2
        bin_scale_phi2 = bin_counts_phi2[1] / bin_counts0_phi2[1]
        #ax2.hist(phi2, bins=50, histtype="step", density=True, color="royalblue", label="MC: vertical")
        ax2.errorbar(bin_centres_phi2, bin_counts_phi2, yerr=bin_err_phi2*bin_scale_phi2, fmt='s', ms=3, capsize=2, color="royalblue", label="MC: horizontal")
        ax2.set_xlabel(r"$\phi$ [rad]", fontsize=15)
        ax2.set_ylabel(r"Normalized $N_s$", fontsize=15)
        dy2 = []
        for i in dx:
            dy2.append(phi_theory(i+np.pi/2., 0.20))
        ax2.plot(dx, dy2, "--", color="black", label="Theory")
        #ax2.legend(prop={"size":15})
        ax2.grid(True)

        ax3 = fig.add_subplot(gs[2, 1])#gs[0, 0:3]中0选取figure的第一行，0:3选取figure第二列和第三列
        bin_counts0_phi3, bin_edges0 = np.histogram(phi3, bins=50, range=(0, 2*np.pi))
        bin_err_phi3 = np.sqrt(bin_counts0_phi3)
        bin_counts_phi3, bin_edges_phi3 = np.histogram(phi3, bins=50, range=(0, 2*np.pi), density=True)
        bin_centres_phi3 = (bin_edges_phi3[:-1] + bin_edges_phi3[1:]) / 2
        bin_scale_phi3 = bin_counts_phi3[1] / bin_counts0_phi3[1]
        ax3.errorbar(bin_centres_phi3, bin_counts_phi3, yerr=bin_err_phi3*bin_scale_phi3, fmt='*', ms=3, capsize=2, color="gray", label="MC: unpolarized")
        ax3.set_ylim(0.155, 0.165)
        #ax3.hist(phi3, bins=50, histtype="step", density=True, color="gray", label="MC: unpolarized")
        ax3.set_xlabel(r"$\phi$ [rad]", fontsize=15)
        ax3.set_ylabel(r"Normalized $N_s$", fontsize=15)
        dy3 = []
        for i in dx:
            dy3.append(1/2/np.pi)
        ax3.plot(dx, dy3, "--",  color="black", label="Theory")
        ax3.grid(True)
        #ax3.legend(prop={"size":15})

        #plt.tight_layout()
        plt.savefig("angular_dist_1D.pdf")
        plt.show()

    if drawHist2D:
        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig, ax1 = plt.subplots()
        costheta0, phi0, cosbeta0 = ld.loadMCdata("scat_Natural_inMom001_rhov0-0")
        costheta1, phi1, cosbeta1 = ld.loadMCdata("scat_inPol100_inMom001_rhov0-2")
        X, Y = np.meshgrid(np.linspace(-1, 1, num=100), np.linspace(0, 2*np.pi, num=100))

        cset = ax1.hist2d(costheta1, phi1, bins=(100, 100), range=((-1, 1), (0, 2*np.pi)), density=True, cmap="coolwarm")
        Z = f2d(X, Y, 0.20)
        CS = ax1.contour(X, Y, Z)
        ax1.clabel(CS, inline=1, fontsize=10)
        ax1.set_ylabel(r"$\phi$ [rad]", fontsize=15)
        ax1.set_xlabel(r"cos$(\theta)$", fontsize=15)
        #cbar = plt.colorbar(cset,  shrink=0.6, orientation="vertical", ax=ax)
        cbar = fig.colorbar(cset[3], ax=ax1, shrink=0.6, orientation="vertical")
        cbar.set_label(r'Normalized $N_s$',fontsize=13, labelpad=15)

        #cset1 = ax2.hist2d(costheta0, phi0, bins=(100, 100), range=((-1, 1), (0, 2*np.pi)), density=True, cmap="coolwarm")
        #Z1 = f2d(X, Y, 0.0)
        #CS1 = ax2.contour(X, Y, Z1)
        #ax2.clabel(CS1, inline=1, fontsize=10)
        #ax2.set_ylabel(r"$\phi$ [rad]", fontsize=15)
        #ax2.set_xlabel(r"cos$(\theta)$", fontsize=15)
        ##cbar = plt.colorbar(cset,  shrink=0.6, orientation="vertical", ax=ax)
        #cbar = fig.colorbar(cset1[3], ax=ax2, shrink=0.6, orientation="vertical")
        #cbar.set_label(r'Normalized $N_s$',fontsize=13, labelpad=15)
        
        plt.savefig("angular_dist_2D.pdf")
        plt.show()



    if drawHist3D :
        #fig, (ax, ax1, ax2, ax3) = plt.subplots(2, 2, subplot_kw={"projection": "3d"})
        #fig, (ax, ax1, ax2, ax3) = plt.subplots(1, 4, subplot_kw={"projection": "3d"}, figsize=(12, 10))
        fig = plt.figure(dpi=100,
                         #constrained_layout=True,
                         figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1., 1.])#GridSpec将fiure分为3行3列，每行三个axes，gs为一个matplotlib.gridspec.GridSpec对象，可灵活的切片figure

        ax = fig.add_subplot(gs[0, 0], projection="3d")
        ax1 = fig.add_subplot(gs[0, 1], projection="3d")
        ax2 = fig.add_subplot(gs[1, 0], projection="3d")
        ax3 = fig.add_subplot(gs[1, 1], projection="3d")

        costheta1, phi1, cosbeta1 = ld.loadMCdata("scat_inPol100_inMom001_rhov0-2")
        count = np.zeros((100, 100))
        value = np.zeros((100, 100))
        binwX, binwY = 2./100., 2*np.pi/100
        for i, j, k in zip(costheta1, phi1, cosbeta1):
            binNy, binNx = int((i+1)/binwX) , int(j/binwY)
            count[binNx, binNy] += 1
            value[binNx, binNy] += k

        data = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                if count[i, j] == 0:
                    data[i, j] = 0
                else:
                    data[i, j] = value[i, j] / count[i, j]


        Y, X = np.meshgrid(np.linspace(-1, 1, num=100), np.linspace(0, 2*np.pi, num=100))
        surf = ax.plot_surface(X, Y, data, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        
        ax.set_ylabel(r"cos($\theta$)", fontsize=15)
        ax.set_xlabel(r"$\phi$ [rad]", fontsize=15)
        ax.set_zlabel(r"cos($\beta$)", fontsize=15)
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, 2*np.pi)
        ax.set_zlim(-0.5, 1)
        ax.set_title("(a)", fontsize=15, y=0.97)

        cset = ax.contourf(X, Y, count/len(costheta1), zdir='z', offset=-0.5,  cmap=cm.coolwarm)
        cbar = plt.colorbar(cset,  shrink=0.6, orientation="vertical", ax=ax, anchor=(0.8, 0.6))
        cbar.ax.ticklabel_format(style='scientific',scilimits=(0, 0),useMathText=False)
        cbar.set_label(r'Normalized $N_s$',fontsize=15, labelpad=15)


        costheta1, phi1, cosbeta1 = ld.loadMCdata("scat_inPol010_inMom001_rhov0-2")
        count = np.zeros((100, 100))
        value = np.zeros((100, 100))
        binwX, binwY = 2./100., 2*np.pi/100
        for i, j, k in zip(costheta1, phi1, cosbeta1):
            binNy, binNx = int((i+1)/binwX) , int(j/binwY)
            count[binNx, binNy] += 1
            value[binNx, binNy] += k

        data = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                if count[i, j] == 0:
                    data[i, j] = 0
                else:
                    data[i, j] = value[i, j] / count[i, j]

        
        surf = ax1.plot_surface(X, Y, data, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        
        ax1.set_ylabel(r"cos($\theta$)", fontsize=15)
        ax1.set_xlabel(r"$\phi$ [rad]", fontsize=15)
        ax1.set_zlabel(r"cos($\beta$)", fontsize=15)
        ax1.set_ylim(-1, 1)
        ax1.set_xlim(0, 2*np.pi)
        ax1.set_zlim(-0.5, 1)
        ax1.set_title("(b)", fontsize=15, y=0.97)

        cset1 = ax1.contourf(X, Y, count/len(costheta1), zdir='z', offset=-0.5,   cmap=cm.coolwarm)
        cbar1 = plt.colorbar(cset1, shrink=0.6, orientation="vertical", ax=ax1, anchor=(0.8, 0.6))
        cbar1.ax.ticklabel_format(style='scientific',scilimits=(0, 0),useMathText=False)
        cbar1.set_label(r'Normalized $N_s$',fontsize=15, labelpad=15)


        costheta1, phi1, cosbeta1 = ld.loadMCdata("scat_Natural_inMom001_rhov0-2")
        count = np.zeros((100, 100))
        value = np.zeros((100, 100))
        binwX, binwY = 2./100., 2*np.pi/100
        for i, j, k in zip(costheta1, phi1, cosbeta1):
            #binNx, binNy = int((i+1)/binwX) , int(j/binwY)
            binNy, binNx = int((i+1)/binwX) , int(j/binwY)
            count[binNx, binNy] += 1
            value[binNx, binNy] += k

        data = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                if count[i, j] == 0:
                    data[i, j] = 0
                else:
                    data[i, j] = value[i, j] / count[i, j]

        
        surf = ax2.plot_surface(X, Y, data, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        
        ax2.set_ylabel(r"cos($\theta$)", fontsize=15)
        ax2.set_xlabel(r"$\phi$ [rad]", fontsize=15)
        ax2.set_zlabel(r"cos($\beta$)", fontsize=15)
        ax2.set_ylim(-1, 1)
        ax2.set_xlim(0, 2*np.pi)
        ax2.set_zlim(-0.5, 1)
        ax2.set_title("(c)", fontsize=15, y=0.97)

        cset2 = ax2.contourf(X, Y, count/len(costheta1), zdir='z', offset=-0.5, cmap=cm.coolwarm)
        cbar2 = plt.colorbar(cset2, shrink=0.6, orientation="vertical", ax=ax2, anchor=(0.8, 0.6))
        cbar2.ax.ticklabel_format(style='scientific',scilimits=(0, 0),useMathText=False)
        cbar2.set_label(r'Normalized $N_s$',fontsize=15, labelpad=15)


        costheta1, phi1, cosbeta1 = ld.loadMCdata("scat_Natural_inMom001_rhov0-0")
        count = np.zeros((100, 100))
        value = np.zeros((100, 100))
        binwX, binwY = 2./100., 2*np.pi/100
        for i, j, k in zip(costheta1, phi1, cosbeta1):
            binNy, binNx = int((i+1)/binwX) , int(j/binwY)
            count[binNx, binNy] += 1
            value[binNx, binNy] += k

        data = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                if count[i, j] == 0:
                    data[i, j] = 0
                else:
                    data[i, j] = value[i, j] / count[i, j]

        
        surf = ax3.plot_surface(X, Y, data, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        
        ax3.set_ylabel(r"cos($\theta$)", fontsize=15)
        ax3.set_xlabel(r"$\phi$ [rad]", fontsize=15)
        ax3.set_zlabel(r"cos($\beta$)", fontsize=15)
        ax3.set_ylim(-1, 1)
        ax3.set_xlim(0, 2*np.pi)
        ax3.set_zlim(-0.5, 1)
        ax3.set_title("(d)", fontsize=15, y=0.97)

        cset3 = ax3.contourf(X, Y, count/len(costheta1), zdir='z', offset=-0.5, cmap=cm.coolwarm)
        cbar3 = plt.colorbar(cset3, shrink=0.6, orientation="vertical", ax=ax3, anchor=(0.8, 0.6))
        cbar3.ax.ticklabel_format(style='scientific',scilimits=(0, 0),useMathText=False)
        cbar3.set_label(r'Normalized $N_s$',fontsize=15, labelpad=15)


        plt.tight_layout()
        plt.savefig("./pdffiles/angle3D_inMom001_rhov02.pdf")
        plt.show()

    if drawCompare:
        fig, ax = plt.subplots()
        
        dx = np.arange(-1, 1, 0.01)
        for r in [0, 0.1, 0.2, 0.3, 0.4]:
            dy = ld.theta_prediction(r , dx)
            #dy = theta_theory(r, dx)
            ax.plot(dx, dy, "-", label=r"$\rho_v$ = %.1f"%r)

        ax.legend(prop={"size":15})
        ax.set_xlabel(r"cos$(\theta)$", fontsize=15)
        ax.set_ylabel("P.D.F", fontsize=15)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig("compare_rhov_theta.pdf")
        plt.show()



