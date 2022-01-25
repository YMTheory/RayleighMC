import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm

import uproot as up
from tqdm import tqdm

if __name__ == "__main__" :

    draw1D = False
    draw2D = False
    draw3D = True


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


    phi = np.array(phi)
    cosbeta = 1*py 


    if draw1D:

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.hist(costheta, bins=100)
        ax1.set_xlabel(r"cos($\theta$)", fontsize=15)
        ax1.tick_params(axis='both', which='major', labelsize=15)

        ax2.hist(phi, bins=100)
        ax2.set_xlabel(r"$\phi$", fontsize=15)
        ax2.tick_params(axis='both', which='major', labelsize=15)
        plt.tight_layout()
        plt.savefig("./pdffiles/scatAng_inPol010_inMom001.pdf")

    if draw2D:
        fig, ax = plt.subplots()
        h = ax.hist2d(costheta, phi, bins=(100, 100))
        ax.set_xlabel(r"cos($\Theta$)", fontsize=15)
        ax.set_ylabel(r"$\phi$", fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        fig.colorbar(h[3], ax=ax)

        plt.tight_layout()
        plt.savefig("./pdffiles/scatAng2D_inPol010_inMom001.pdf")

    if draw3D:
        count = np.zeros((100, 100))
        value = np.zeros((100, 100))
        binwX, binwY = 2./100., 2*np.pi/100
        for i, j, k in zip(costheta, phi, cosbeta):
            binNx, binNy = int((i+1)/binwX) , int(j/binwY)
            count[binNx, binNy] += 1
            value[binNx, binNy] += k

        data = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                if count[i, j] == 0:
                    data[i, j] = 0
                else:
                    data[i, j] = value[i, j] / count[i, j]


        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X, Y = np.meshgrid(np.linspace(-1, 1, num=100), np.linspace(0, 2*np.pi, num=100))
        # Plot the surface.

        surf = ax.plot_surface(X, Y, data, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        
        ax.set_xlabel(r"cos($\theta$)", fontsize=13)
        ax.set_ylabel(r"$\phi$", fontsize=13)
        ax.set_zlabel(r"cos($\beta$)", fontsize=13)

        # Customize the z axis.
        #ax.set_zlim(-1.01, 1.01)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        #ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        plt.tight_layout()
        plt.savefig("./pdffiles/scatAng3D_inPol010_inMom001.pdf")
    
    plt.show()





