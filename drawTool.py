import numpy as np
import matplotlib.pyplot as plt

import ROOT

from Rayleigh_class import Rayleigh
from detector import detector

import sys, os

from tqdm import tqdm

import polarizerTool as pt

import analyser as ana


def func(p0, p1, p2, x) :
    return p0 + p1*np.cos(x-p2)*np.cos(x-p2)


if __name__ == "__main__" :
    
    drawPolar = True

    theta = [75, 80, 85, 90, 95, 100, 105]

    color = ["blue", "red", "black", "green", "darkviolet", "orange", "gray"]

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















