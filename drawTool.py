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
        ang = 90.
        vec_pol_angle = np.array([179, 209, 239, 269, 299, 329, 359, 29, 59, 89, 119, 149])
        R, Rerr, p0, p1, p2, p0err, p1err, p2err = ana.read_data(ang)
        dx = np.arange(0, 2*np.pi, 0.01)
        dy = []
        for i in dx:
            dy.append(func(p0, p1, p2, i))
        dy = np.array(dy)
        ## simulation
        polAngle_arr, prob_arr = pt.generator(1, 0, 0, 0.20, 90.)

        norm = (prob_arr[90] + prob_arr[270]) / 2.
        print("MC normalization factor = %.3e" %norm)

        ax.plot(dx+p2, dy/p0, "--", lw=2, color="black", label="Fitting")
        ax.plot(vec_pol_angle/180.*np.pi+p2, R/p0, "o", color="red", label="Exp")
        ax.plot(polAngle_arr, prob_arr/norm, "-", lw=2, color="blue", label="MC")
        ax.set_theta_zero_location("N", np.pi/2.)
        ax.legend()
        ax.set_yticklabels([])

        plt.tight_layout()
        plt.savefig("./pdffiles/Polar_"+str(ang)+"deg.pdf")
        plt.show()















