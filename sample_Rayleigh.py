import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import uproot3
import hist
from hist import Hist
import mplhep
from Rayleigh_class import Rayleigh
import sys, os


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


if __name__ == "__main__" :

    if len(sys.argv) == 1:
        outfile = "./rootfiles/tmp.root"
    else:
        outfile = sys.argv[1]


    # Histo Saving
    #hMom = Hist(
    #hist.axis.Regular(50, 0, 180, name="momtheta", label=r"$\theta$ [degree]", flow=False),
    #hist.axis.Regular(50, 0, 360, name="momphi",   label=r"$\phi$ [degree]", flow=False),
    #)

    #hPol = Hist(
    #hist.axis.Regular(50, 0, 180, name="poltheta", label=r"$\theta$ [degree]", flow=False),
    #hist.axis.Regular(50, 0, 360, name="polphi",   label=r"$\phi$ [degree]", flow=False),
    #)

    #hBeta = Hist(
    #hist.axis.Regular(50, 0, 360, name="polbeta", label=r"$\beta$ [degree]", flow=False),        
    #)

    ray = Rayleigh()
    outpx, outpy, outpz = [], [], []
    outMomTheta, outMomPhi = [], []
    outex, outey, outez = [], [], []
    polAngle = []
    beta = []

    ray.set_inMom(0, 0, 0.1)
    ray.set_inPol(1, 0, 0)
    #ray.set_outMomTheta(np.pi/2.)
    #ray.set_outMomPhi(np.pi*1/2.)

    nPhoton = 50000
    for i in range(nPhoton) :
        if i % 100 == 0 :
            print(i)
        ray.SampleFromTheta()
        #ray.sampleLocally()
        #ray.sampleSeconderies()   # Geant4 OpRayleigh sampling
        #ray.set_outMomPhi(ray.GeneratePhi())
        #ray.GetPhotonPolarisation()
        outMom = ray.get_outMom()
        outPol = ray.get_outPol()
        outPTheta = ray.get_outMomTheta()
        outPPhi = ray.get_outMomPhi()

        outpx.append(outMom[0])
        outpy.append(outMom[1])
        outpz.append(outMom[2])
        outMomTheta.append(outPTheta)
        outMomPhi.append(outPPhi)
        outex.append(outPol[0])
        outey.append(outPol[1])
        outez.append(outPol[2])
        polAngle.append(ray.get_polAngle())
        beta.append(ray.get_outPolBeta())


        ### Angle calculation
        #momTheta, momPhi = calculateAngle(outMom)
        #polTheta, polPhi = calculateAngle(outPol)
        
        #ptheta1_arr.append(momTheta)
        #pphi1_arr.append(momPhi)
        #ltheta1_arr.append(polTheta)
        #lphi1_arr.append(polPhi)
        #lbeta_arr.append(ray.get_outPolBeta() * 180./np.pi)
        #lpolAngle_arr.append(ray.get_polAngle() * 180 / np.pi)

        #hMom.fill(momTheta, momPhi)
        #hPol.fill(polTheta, polPhi)
        #hBeta.fill(ray.get_outPolBeta() * 180 / np.pi)
    
    with uproot3.recreate("./rootfiles/"+outfile) as f:
        f["ray"] = uproot3.newtree({"outpx":"float64", "outpy":"float64", "outpz":"float64",
                    "outMomTheta":"float64", "outMomPhi":"float64",
                    "outex":"float64", "outey":"float64", "outez":"float64", "polAngle":"float64", "beta":"float64"})
        f["ray"].extend({"outpx":outpx, "outpy":outpy, "outpz":outpz, 
                    "outMomTheta":outMomTheta, "outMomPhi":outMomPhi,
                    "outex":outex, "outey":outey, "outez":outez, "polAngle":polAngle, "beta":beta})

