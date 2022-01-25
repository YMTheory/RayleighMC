import numpy as np
import matplotlib.pyplot as plt

import uproot as up
from tqdm import tqdm

if __name__ == "__main__" :

    kx, ky, kz, px, py, pz = [], [], [], [], [], []
    for i in tqdm(range(100)):
        filename = "./rootfiles/scat_inPol010_inMom001_"+str(i)+".root"
        ff = up.open(filename)
        tmpkx = ff["Ray"]["photon_momx"].array()
        tmpky = ff["Ray"]["photon_momy"].array()
        tmpkz = ff["Ray"]["photon_momz"].array()
        tmppx = ff["Ray"]["photon_polx"].array()
        tmppy = ff["Ray"]["photon_poly"].array()
        tmppz = ff["Ray"]["photon_polz"].array()

        kx.extend(tmpkx)
        ky.extend(tmpky)
        kz.extend(tmpkz)
        px.extend(tmppx)
        py.extend(tmppy)
        pz.extend(tmppz)

    px = np.array(px)
    py = np.array(py)
    pz = np.array(pz)
    kx = np.array(kx)
    ky = np.array(ky)
    kz = np.array(kz)
    
    costheta = kz / np.sqrt(kx**2 + ky**2 + kz**2)
    phi = []
    for i, j in zip(kx, ky):
        tmpphi = np.arctan(np.abs(j/i))
        if i > 0 and j > 0:
            phi.append(tmpphi)
        elif i <= 0 and j>0:
            phi.append(np.pi-tmpphi)
        elif i<=0 and j<=0 :
            phi.append(np.pi + tmpphi)
        else:
            phi.append(2*np.pi - tmpphi)

    #fig, (ax1, ax2) = plt.subplots(2, 1)
    #ax1.hist(costheta, bins=100)
    #ax1.set_xlabel(r"cos($\theta$)", fontsize=15)
    #ax1.tick_params(axis='both', which='major', labelsize=15)

    #ax2.hist(phi, bins=100)
    #ax2.set_xlabel(r"$\phi$", fontsize=15)
    #ax2.tick_params(axis='both', which='major', labelsize=15)

    fig, ax = plt.subplots()
    h = ax.hist2d(costheta, phi, bins=(100, 100))
    ax.set_xlabel(r"cos($\Theta$)", fontsize=15)
    ax.set_ylabel(r"$\phi$", fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    fig.colorbar(h[3], ax=ax)

    plt.tight_layout()
    plt.savefig("./pdffiles/scatAng2D_inPol010_inMom001.pdf")
    #plt.savefig("./pdffiles/scatAng_inPol010_inMom001.pdf")
    plt.show()





