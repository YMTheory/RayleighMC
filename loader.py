import numpy as np
#from tqdm import tqdm
import uproot as up

def func(p0, p1, p2, x):
    return p0 + p1*np.cos(x-p2)**2



def theta_prediction(rhov, x):
    rhou = 2*rhov / (1 + rhov)
    scale = 1 / (2+2./3*(1-rhou)/(1+rhou))
    #return 1 + (1-rhou)/(1+rhou) * np.cos(x)**2
    return scale*(1 + (1-rhou)/(1+rhou) * x**2)



def loadMCdata(fn):
    ix, iy, iz = [], [], []
    kx, ky, kz, px, py, pz = [], [], [], [], [], []
    if fn == "scat_Natural_inMom001_rhov0-0":
        Ns = 100
    else:
        Ns = 1000
    Ns = 10
    for i in range(Ns):
    #for i in tqdm(range(Ns)):
        #filename = "./rootfiles/"+fn+str(i)+".root"
        filename = "./rootfiles/"+fn+str(i)+".root"
        print(filename)
        #if filename ==  './rootfiles/scat_inPol100_inMom001_106.root' or filename == "./rootfiles/scat_inPol010_inMom001_106.root" or filename == "./rootfiles/scat_Natural_inMom001_106.root" or filename=='./rootfiles/scat_Natural_inMom001_346.root':
        #    continue
        ff = up.open(filename)
        tmpix = ff["Ray"]["in_polx"].array()
        tmpiy = ff["Ray"]["in_poly"].array()
        tmpiz = ff["Ray"]["in_polz"].array()
        tmpkx = ff["Ray"]["photon_momx"].array()
        tmpky = ff["Ray"]["photon_momy"].array()
        tmpkz = ff["Ray"]["photon_momz"].array()
        tmppx = ff["Ray"]["photon_polx"].array()
        tmppy = ff["Ray"]["photon_poly"].array()
        tmppz = ff["Ray"]["photon_polz"].array()

        ix.extend(tmpix)
        iy.extend(tmpiy)
        iz.extend(tmpiz)
        kx.extend(tmpkx)
        ky.extend(tmpky)
        kz.extend(tmpkz)
        px.extend(tmppx)
        py.extend(tmppy)
        pz.extend(tmppz)

    ix = np.array(ix)
    iy = np.array(iy)
    iz = np.array(iz)
    px = np.array(px)
    py = np.array(py)
    pz = np.array(pz)
    kx = np.array(kx)
    ky = np.array(ky)
    kz = np.array(kz)

    costheta = kz / np.sqrt(kx**2 + ky**2 + kz**2)
    theta = np.arccos(costheta)
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

    cosbeta = ix*px + iy*py + iz*pz

    return costheta, phi, cosbeta

