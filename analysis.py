import uproot as up
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplhep, hist
import sys, os
from hist import Hist
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import vector, random

def calculateAngle(vec): 
    x, y, z = vec
    len = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.abs(z)/len) * 180. / np.pi
    if z < 0:
        theta = 180 - theta

    phi = np.arctan(np.abs(y)/np.abs(x)) * 180 / np.pi
    if x <= 0 and y >=0:
        phi = 180 - phi
    if x <= 0 and y <0:
        phi = 180 + phi
    if x > 0 and y <0:
        phi = 360 - phi

    return theta, phi


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

if __name__ == "__main__" :

    if len(sys.argv) != 2:
        print("Wrong argument number ! ---> Required a input filename !")


    tt = up.open(sys.argv[1])["ray"]
    outpx = tt["outpx"].array()
    outpy = tt["outpy"].array()
    outpz = tt["outpz"].array()
    outex = tt["outex"].array()
    outey = tt["outey"].array()
    outez = tt["outez"].array()
    momTheta = tt["outMomTheta"].array()
    momPhi = tt["outMomPhi"].array()
    polAngle = tt["polAngle"].array()


    hAngle = Hist(
        hist.axis.Regular(50, -10, 190, name="polangle", label="angle [deg]", flow=False),
    )
    hCalc = Hist(
        hist.axis.Regular(50, -10, 190, name="calc", label="calc [deg]", flow=False),
    )



    
    hMom = Hist(
        hist.axis.Regular(50, -10, 190, name="momtheta", label="momtheta [deg]", flow=False),
        hist.axis.Regular(50, -10, 370, name="momphi", label="momphi [deg]", flow=False),
    )

    hPol = Hist(
        hist.axis.Regular(50, -10, 190, name="poltheta", label="poltheta [deg]", flow=False),
        hist.axis.Regular(50, -10, 370, name="polphi", label="polphi [deg]", flow=False),
    )

    hAngle = Hist(
        hist.axis.Regular(50, -10, 190, name="polangle", label="angle [deg]", flow=False),
    )

    for i in range(len(polAngle)):
        #vec1 = [outpx[i], outpy[i], outpz[i]]
        #theta1, phi1 = calculateAngle(vec1)
        theta1, phi1 = momTheta[i], momPhi[i]
        Theta = np.arccos(np.sqrt(1 - np.cos(phi1)**2*np.sin(theta1)**2))
        theta = polAngle[i]
        vec1 = [outex[i], outey[i], outez[i]]
        theta1, phi1 = calculateAngle(vec1)

        if random.uniform(0, 1) < 0.5:
            Theta = np.pi - Theta
    
        hAngle.fill(theta*180./np.pi)
        hCalc.fill(Theta * 180./np.pi)

        hPol.fill(theta1, phi1)

    hMom.fill(momTheta*180/np.pi, momPhi*180/np.pi)
    
    #fig, ax = plt.subplots()

    #mplhep.histplot(hAngle, ax=ax, color="black", label="sampling")
    #mplhep.histplot(hCalc,  ax=ax, color="red", label="calculation")
    #ax.legend(prop={"size": 15})
    #ax.set_xlabel("angle [deg]", fontsize=16)
    #plt.tight_layout()
    #plt.savefig("cosTheta.pdf")
    #plt.show()



    #plt.show()
    
    
    fig2 = plt.figure(constrained_layout=True, figsize=(16 , 4))
    spec2 = gridspec.GridSpec(ncols=3, nrows=1, figure=fig2)
    f2_ax1 = fig2.add_subplot(spec2[0, 0])
    f2_ax2 = fig2.add_subplot(spec2[0, 1])
    f2_ax3 = fig2.add_subplot(spec2[0, 2])
    #f2_ax4 = fig2.add_subplot(spec2[0, 3])


    mplhep.hist2dplot(hMom, ax=f2_ax1)
    f2_ax1.set_xlabel(r"$\theta$ [deg]", fontsize=15)
    f2_ax1.set_ylabel(r"$\phi$ [deg]", fontsize=15)
    #f2_ax2.hist(ptheta1_arr, bins=50, color="blue", edgecolor="black")
    mplhep.histplot(hMom.project("momtheta"), ax=f2_ax2, color="blue")
    f2_ax2.set_xlabel(r"$\theta$ [deg]", fontsize=15)
    #f2_ax3.hist(pphi1_arr, bins=50, color="blue", edgecolor="black")
    mplhep.histplot(hMom.project("momphi"), ax=f2_ax3, color="red")
    f2_ax3.set_xlabel(r"$\phi$ [deg]", fontsize=15)
    #mplhep.histplot(hAngle, ax=f2_ax4, color="black")
    #f2_ax4.set_xlabel(r"$\phi$ [deg]", fontsize=15)

    plt.tight_layout()
    plt.savefig("sample_mom.pdf")
    plt.show()

    

    
    """

    fig = plt.figure()
    #ax = plt.axes(projection='3d')
    ax = plt.gca(projection='3d') 
    ax._axis3don = False

    a = Arrow3D([0, 0], [0, 0], [-1, 0], mutation_scale=20, 
                lw=3, arrowstyle="-|>", color="r")

    ax.add_artist(a)
    a1 = Arrow3D([-0.3, 0.3], [0, 0], [-0.5, -0.5], mutation_scale=20, 
                lw=2, arrowstyle="-|>", color="gray")
    a2 = Arrow3D([0.3, -0.3], [0, 0], [-0.5, -0.5], mutation_scale=20, 
                lw=2, arrowstyle="-|>", color="gray")

    ax.add_artist(a1)
    ax.add_artist(a2)

    a = Arrow3D([0, 0], [0, 1], [0, 0], mutation_scale=20, 
                lw=3, arrowstyle="-|>", color="blue")

    ax.text(0, 1.1, 0, "y", fontsize=15)
    ax.add_artist(a)

    a = Arrow3D([0, 0], [0, 0], [-1.1, 1.1], mutation_scale=20, 
                lw=2, arrowstyle="-|>", color="black")

    ax.text(0, 0, 1.2, "z", fontsize=15)
    ax.add_artist(a)    
    a = Arrow3D([0, 0], [0, 1.1], [0, 0], mutation_scale=20, 
                lw=2, arrowstyle="-|>", color="black")

    ax.add_artist(a)    
    a = Arrow3D([-1, 1], [0, 0], [0, 0], mutation_scale=20, 
                lw=3, arrowstyle="-|>", color="black")

    ax.text(1.1, 0, 0, "x", fontsize=15)
    ax.add_artist(a)

    ax.text(0, 0.1, -0.7, "incident photon", fontsize=15, color="crimson")
    ax.text(0, 0.7, 0.4, "scattered photon", fontsize=15,  color="blue")

    for i in range(20):
        theta = ltheta[i]
        phi = lphi[i]
        x = 0.3*np.cos(phi/180*np.pi) * np.sin(theta/180*np.pi)
        z = 0.3*np.cos(theta/180*np.pi)
        y = 0.3

        a = Arrow3D([0, x], [y, y], [0, z], mutation_scale=20, 
                    lw=2, arrowstyle="-|>", color="gray")

        ax.add_artist(a)


    ax.set_ylim(0, 1.1)
    ax.set_xlim(-1, 1)
    ax.set_zlim(-1.1, 1.1)

    plt.tight_layout()
    #plt.savefig("sampling_scattered90theta90phi.pdf")
    plt.show()

    """






