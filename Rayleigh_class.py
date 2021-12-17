import random

from matplotlib.pyplot import pie
import numpy as np
import vector


class Rayleigh(object):

    def __init__(self) -> None:
        self.inMom  = [0, 0, 1]
        self.inPol  = [1, 0, 0]
        self.outMom = [0, 0, 0]
        self.outPol = [0, 0, 0]
        self.outPol1 = [0, 0, 0]    # perp
        self.outPol2 = [0, 0, 0]    # parallel
        self.inE = 1

        self.outMomTheta = 0
        self.outMomPhi   = 0
        self.outPolTheta = 0
        self.outPolPhi   = 0
        self.outPolBeta  = 0
        self.inMomTheta = 0
        self.inMomPhi   = 0
        self.inPolTheta = 0
        self.inPolPhi   = 0
        self.polAngle   = 0

    def get_inE(self):
        return self.inE


    def set_inMom(self, x, y, z):
        self.inMom = x, y, z
        self.inE = np.sqrt(x**2+y**2+z**2)

    def set_inPol(self, x, y, z):
        self.inPol = x, y, z

    def set_outMom(self, x, y, z):
        self.outMom = x, y, z
    
    def set_outPol1(self, x, y, z):
        self.outPol1 = x, y, z

    def set_outPol2(self, x, y, z):
        self.outPol2 = x, y, z

    def set_outPol(self, x, y, z):
        self.outPol = x, y, z

    def set_nPhotons(self, N):
        self.nPhoton = N

    def get_outPol(self):
        return self.outPol

    def get_outPol1(self):
        return self.outPol1

    def get_outPol2(self):
        return self.outPol2

    def get_polAngle(self):
        return self.polAngle

    def set_polAngle(self, angle):
        self.polAngle = angle


    def rotateUz(self, u1, u2, u3, dx, dy, dz):
        up = u1*u1 + u2*u2;
    
        if up > 0:
            up = np.sqrt(up)
            px = dx;  py = dy;  pz = dz;
            dx = (u1*u3*px - u2*py)/up + u1*pz;
            dy = (u2*u3*px + u1*py)/up + u2*pz;
            dz =    -up*px +             u3*pz;
    
        elif u3 < 0.: 
            dx = -dx; dz = -dz;      
        else:
            pass

        return dx, dy, dz


    def normalize(self, dx, dy, dz):
        dd = np.sqrt(dx**2 + dy**2 + dz**2)
        nx = dx / dd
        ny = dy / dd
        nz = dz / dd

        return nx, ny, nz


    def get_inMom(self):
        return self.inMom

    def get_inPol(self):
        return self.inPol

    def get_inMomTheta(self):
        return self.inMomTheta

    def get_inMomPhi(self):
        return self.inMomPhi    

    def get_inPolTheta(self):
        return self.inPolTheta

    def get_inPolPhi(self):
        return self.inPolPhi    


    def get_outMom(self):
        return self.outMom

    def get_outPol(self):
        return self.outPol

    def get_outMomTheta(self):
        return self.outMomTheta

    def set_outMomTheta(self, theta):
        self.outMomTheta = theta

    def get_outMomPhi(self):
        return self.outMomPhi    

    def set_outMomPhi(self, phi):
        self.outMomPhi = phi

    def get_outPolTheta(self):
        return self.outPolTheta

    def get_outPolPhi(self):
        return self.outPolPhi    

    def get_outPolBeta(self):
        return self.outPolBeta

    def set_outPolBeta(self, beta):
        self.outPolBeta = beta

    def sampleLocally(self):
        self.set_inMom(0, 0, 1)
        self.set_inPol(1, 0, 0)

        flag = True
        while flag:
            CosTheta = random.uniform(0, 1)
            SinTheta = np.sqrt(1-CosTheta**2)
            if random.uniform(0, 1) < 0.5 :
                CosTheta = -CosTheta
            
            rand = 2*np.pi * random.uniform(0, 1)
            SinPhi = np.sin(rand)
            CosPhi = np.cos(rand)
 
            self.set_outMomTheta(np.arccos(CosTheta))
            self.set_outMomPhi(rand)

            energy = 1
            px1, py1, pz1 = energy*SinPhi*CosTheta, energy*SinPhi*SinTheta, energy*CosTheta
            self.set_outMom(px1, py1, pz1)
            px1n, py1n, pz1n = CosPhi*SinTheta, SinPhi*SinTheta, CosTheta

            ex0, ey0, ez0 = self.inPol[0], self.inPol[1], self.inPol[2]

            c = -1. / (px1n*ex0 + py1n*ey0 + pz1n*ez0)
            ex1, ey1, ez1 = px1n + c*ex0, py1n+c*ey0, pz1n+c*ez0
            ex1n, ey1n, ez1n = self.normalize(ex1, ey1, ez1)
            self.set_outPol(ex1n, ey1n, ez1n)

            if np.abs(ex1n) < 1e-5 and np.abs(ey1n)<1e-5 and np.abs(ez1n) < 1e-5 :   # special case
                rand = np.pi * 2 *random.uniform(0, 1)
                y, z = np.cos(rand), np.sin(rand)
                ex1n, ey1n, ez1n = 0, y, z
                self.set_outPol(0, y, z)


            cosTheta = ex1n * ex0 + ey1n * ey0 + ez1n * ez0

            if cosTheta**2 > random.uniform(0, 1):
                flag = False

        self.set_polAngle(np.arccos(cosTheta))




    def sampleSeconderies(self):   # implementations in G4OpRayleigh class
        flag = True
        while flag:
            CosTheta = random.uniform(0, 1)
            SinTheta = np.sqrt(1-CosTheta**2)
            if random.uniform(0, 1) < 0.5 :
                CosTheta = -CosTheta
            
            rand = 2*np.pi * random.uniform(0, 1)
            SinPhi = np.sin(rand)
            CosPhi = np.cos(rand)

            tmp_outMomTheta = np.arccos(CosTheta) 
            self.set_outMomTheta(tmp_outMomTheta)
            self.set_outMomPhi(rand)

            E = self.inE
            px0n, py0n, pz0n = self.normalize(self.inMom[0], self.inMom[1], self.inMom[2])
            self.set_outMom(E*SinTheta*CosPhi, E*SinTheta*SinPhi, E*CosTheta)
            px1, py1, pz1 = self.rotateUz(px0n, py0n, pz0n, E*SinTheta*CosPhi, E*SinTheta*SinPhi, E*CosTheta)
            px1n, py1n, pz1n = self.normalize(px1, py1, pz1)

            newMomDirection = vector.obj(x=px1n, y=py1n, z=pz1n)
            oldPolDirection = vector.obj(x=self.inPol[0], y=self.inPol[1], z=self.inPol[2])

            ### Polarisation
            C = -1. / (newMomDirection.dot(oldPolDirection))            
            newPol = newMomDirection + C * oldPolDirection
            e1, e2, e3 = self.normalize(newPol.x, newPol.y, newPol.z)
            newPolDirection = vector.obj(x=e1, y=e2, z=e3)

            if newPolDirection.mag == 0:
                rand = random.uniform(0, 1) * 2 * np.pi
                tmp_ex1, tmp_ey1, tmp_ez1 = self.rotateUz(px1n, py1n, pz1n, np.cos(rand), np.sin(rand), 0)
                newPolDirection = vector.obj(x=tmp_ex1, y=tmp_ey1, z=tmp_ez1) 

            else:
                if random.uniform(0, 1) < 0.5:
                    newPolDirection = -newPolDirection

            cosTheta = newPolDirection.dot(oldPolDirection)

            if cosTheta**2 > random.uniform(0, 1):
                flag = False


        self.set_outPol(newPolDirection.x, newPolDirection.y, newPolDirection.z)
        self.set_polAngle(np.arccos(cosTheta))


        ex0n, ey0n, ez0n = self.normalize(self.inPol[0], self.inPol[1], self.inPol[2])
        oldPolDirection = vector.obj(x=ex0n, y=ey0n, z=ez0n)
        #print("--------------")
        #print(oldPolDirection)
        #print(newMomDirection)
        #print(newPolDirection)
        #print(newPolDirection.dot(newMomDirection))

    ### From the NIMA Paper

    # Generate Phi :
    def GeneratePhi(self):
        phi = 0
        flag = True
        while flag:
            m_phi = random.uniform(0, 1) * 2 * np.pi
            m_Pr = random.uniform(0, 1) * 1
            m_Pphir = 1 - np.cos(m_phi)**2
            if m_Pr <= m_Pphir :  # accept
                phi = m_phi
                flag = False

        return phi


    ## Generate beta
    def GeneratePolarization(self):
        beta = 0
        flag = True
        N = np.sqrt(1-np.sin(self.outMomTheta)**2*np.cos(self.outMomPhi)**2)
        while flag:
            b = 4 * N**2
            m_beta = random.uniform(0, 1) * 2 * np.pi
            m_Pb = random.uniform(0, 1)
            m_Pbeta = np.cos(m_beta)**2
            if m_Pb <= m_Pbeta :
                beta = m_beta
                flag = False
        
        return beta


    def LocalSystem(self):
        self.set_outMomPhi(self.GeneratePhi())

        E = self.inE
        self.set_outMom(E*np.sin(self.outMomTheta)*np.cos(self.outMomPhi), E*np.sin(self.outMomTheta)*np.sin(self.outMomPhi), E*np.cos(self.outMomTheta))

        N = np.sqrt(1-np.sin(self.outMomTheta)**2*np.cos(self.outMomPhi)**2)

        self.set_outPolBeta(self.GeneratePolarization())
        self.set_outPol1(0, np.cos(self.outMomTheta)*np.sin(self.outPolBeta)/N, -np.sin(self.outMomTheta)*np.sin(self.outMomPhi)*np.sin(self.outPolBeta)/N )
        self.set_outPol2(np.cos(self.outPolBeta)*N, -np.sin(self.outMomTheta)**2*np.sin(self.outMomPhi)*np.cos(self.outMomPhi)*np.cos(self.outPolBeta)/N, -np.sin(self.outMomTheta)*np.cos(self.outMomTheta)*np.cos(self.outMomPhi)*np.cos(self.outPolBeta)/N)

        self.set_outPol(self.outPol1[0] + self.outPol2[0] , self.outPol1[1] + self.outPol2[1] , self.outPol1[2] + self.outPol2[2] )

        print(self.outPol)


    # from Geant4 codes :
    def GetPhotonPolarisation(self):
        tmp_ex0, tmp_ey0, tmp_ez0 = self.inPol
        tmp_inPol = vector.obj(x=tmp_ex0, y=tmp_ey0, z=tmp_ez0)

        E = self.inE
        self.set_outMom(E*np.sin(self.outMomTheta)*np.cos(self.outMomPhi), E*np.sin(self.outMomTheta)*np.sin(self.outMomPhi), E*np.cos(self.outMomTheta))
        tmp_px1, tmp_py1, tmp_pz1 = self.outMom
        tmp_outMom = vector.obj(x=tmp_px1, y=tmp_py1, z=tmp_pz1)

        if tmp_inPol.is_parallel(tmp_outMom) or tmp_inPol.is_antiparallel(tmp_outMom) :
            angle = random.uniform(0, 1) * 2*np.pi
            tmp_outPol = vector.obj(x=np.cos(angle), y=np.sin(angle), z=0)
            self.set_outPol(tmp_outPol.x, tmp_outPol.y, tmp_outPol.z)

        elif not tmp_inPol.is_perpendicular(tmp_outMom):
            self.set_outPol(1, 0, 0)
            if random.uniform(0, 1) < 0.5 :
                self.set_outPol(-1, 0, 0)

        else:
            self.set_outPol(tmp_ex0, tmp_ey0, tmp_ez0)   















