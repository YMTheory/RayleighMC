import random
import numpy as np
from scipy.spatial.transform import Rotation as R
import calculateRotationMatrix as rot
import vector_method as vm


class Rayleigh(object):

    def __init__(self) -> None:
        self.inMom  = np.array([0., 0., 1.])
        self.inPol  = np.array([1., 0., 0.])
        self.outMom = np.array([0., 0., 0.])
        self.outPol = np.array([0., 0., 0.])
        self.inE = 1
        self.midPol = np.array([1., 0., 0.])

        self.nPhoton = 1

        self.outMomTheta = 0
        self.outMomPhi   = 0
        self.outPolTheta = 0
        self.outPolPhi   = 0
        self.inMomTheta = 0
        self.inMomPhi   = 0
        self.inPolTheta = 0
        self.inPolPhi   = 0
        self.polAngle   = 0

        self.rhou = 0
        self.rhov = 0
        self.alpha = 0.
        self.beta = 0.

        self.scale = 1.
        self.scatProb = 1.

    def get_inE(self):
        return self.inE


    def set_inMom(self, x, y, z):
        self.inMom[0] = x
        self.inMom[1] = y
        self.inMom[2] = z
        self.inE = np.sqrt(x**2+y**2+z**2)


    def set_midPol(self, x, y, z):
        self.midPol[0] = x
        self.midPol[1] = y
        self.midPol[2] = z

    def set_inPol(self, x, y, z):
        self.inPol[0] = x
        self.inPol[1] = y
        self.inPol[2] = z

    def set_outMom(self, x, y, z):
        self.outMom[0] = x
        self.outMom[1] = y
        self.outMom[2] = z
    
    def set_outPol(self, x, y, z):
        self.outPol[0] = x
        self.outPol[1] = y
        self.outPol[2] = z


    def set_nPhotons(self, N):
        self.nPhoton = N


    def get_outPol(self):
        return self.outPol


    def get_polAngle(self):
        return self.polAngle

    def set_polAngle(self, angle):
        self.polAngle = angle


    def set_rhou(self, delta):
        self.rhou = delta
        self.rhov = delta / (2 - delta) 

    def get_rhou(self):
        return self.rhou

    def set_rhov(self, delta):
        self.rhov = delta
        self.rhou = 2*self.rhov / (1 + self.rhov)

    def get_rhov(self):
        return self.rhov

    
    def get_alpha(self):
        return self.alpha

    def get_beta(self):
        return self.beta

    def get_scatProb(self):
        return self.scatProb

    def set_scatPron(self, prob):
        self.scatProb = prob



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

    def get_midPol(self):
        return self.midPol

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


    def calcTensor(self):
        # On the main axes
        alpha = 1
        rho = self.rhov
        beta = (np.sqrt(45*rho)/3 - np.sqrt(3-4*rho)) / (-np.sqrt(3-4*rho) - 2/3*np.sqrt(45*rho))
        #beta = (np.sqrt(45*rho)/3 + np.sqrt(3-4*rho)) / (np.sqrt(3-4*rho) -2/3*np.sqrt(45*rho))

        #self.alpha = np.sqrt(alpha)
        #self.beta = np.sqrt(beta)
        self.alpha = alpha
        self.beta = beta


    def rotate_inPol(self):
        # My own implementations based on symmetry
        #cosThetaLoc = random.uniform(-1, 1)
        #sinThetaLoc = np.sqrt(1 - cosThetaLoc**2)
        #PhiLoc   = random.uniform(0, 2*np.pi)
        #cosPhiLoc = np.cos(PhiLoc)
        #sinPhiLoc = np.sin(PhiLoc)
        #newX1 = np.array([sinThetaLoc*cosPhiLoc, sinThetaLoc*sinPhiLoc, cosThetaLoc])
        #tmp_newX2 = vm.perpendicular_vector(newX1)
        #tmp_newX3 = np.cross(newX1, tmp_newX2)

        #ksi = random.uniform(0, 2*np.pi)
        #newX2 = tmp_newX2 * np.cos(ksi) + tmp_newX3 * np.sin(ksi)
        #newX3 = np.cross(newX1, newX2)

        #crd1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        #crd2 = np.vstack((newX1, newX2, newX3))
        #rotM = np.matmul(crd1, crd2)

        # based on scipy random transform
        p = R.random(1).as_matrix()[0]
        rotM = R.from_matrix(p) 
        rotM_inv = rotM.inv()
        p_inv = R.as_matrix(rotM_inv)

        # check with fixed rotation
        #rotM = R.from_euler('zyx', [[60, 30, 30]], degrees=True)
        #rotM_inv = rotM.inv()
        #p = R.as_matrix(rotM)[0]
        #p_inv = R.as_matrix(rotM_inv)[0]

        T = np.array([[self.alpha, 0, 0], [0, self.beta, 0], [0, 0, self.beta]])
        T_new = np.matmul(np.matmul(p, T), p_inv)

        pol_new = np.matmul(T_new, self.inPol)
        inpx, inpy, inpz = self.normalize(pol_new[0], pol_new[1], pol_new[2])
        cosTheta2 = (inpx*self.inPol[0] + inpy*self.inPol[1] + inpz*self.inPol[2])**2
        #print("cosTheta2 = ", cosTheta2)
        self.set_inPol(inpx, inpy, inpz)

        #self.scale = cosTheta2
        self.scale = (pol_new.dot(pol_new))
        #print("scale factor = ", self.scale)



        ## energy loss in re-polarisation ????
        #inP = self.get_inPol()
        #inM = self.get_inMom()
        #vanish_prob = inP.dot(inM)
        #sample = random.uniform(0, 1)
        #if sample < vanish_prob:
        #    self.scatProb = 0



    def rotate_inPol_twice(self):
    
        scale = 1.

        # 1st rotation 
        p = R.random(1).as_matrix()[0]
        rotM = R.from_matrix(p) 
        rotM_inv = rotM.inv()
        p_inv = R.as_matrix(rotM_inv)

        T = np.array([[self.alpha, 0, 0], [0, self.beta, 0], [0, 0, self.beta]])
        T_new = np.matmul(np.matmul(p, T), p_inv)

        pol_new = np.matmul(T_new, self.inPol)
        inpx, inpy, inpz = self.normalize(pol_new[0], pol_new[1], pol_new[2])
        self.set_midPol(inpx, inpy, inpz)
        inPol1 = self.inPol.reshape(3, 1)

        scale *= (np.matmul(self.midPol, np.matmul(T_new, inPol1)))**2
        
        # 2nd rotation 
        p = R.random(1).as_matrix()[0]
        rotM = R.from_matrix(p) 
        rotM_inv = rotM.inv()
        p_inv = R.as_matrix(rotM_inv)

        T = np.array([[self.alpha, 0, 0], [0, self.beta, 0], [0, 0, self.beta]])
        T_new = np.matmul(np.matmul(p, T), p_inv)

        pol_new = np.matmul(T_new, self.inPol)
        inpx, inpy, inpz = self.normalize(pol_new[0], pol_new[1], pol_new[2])
        self.set_midPol(inpx, inpy, inpz)

        # sample scattering light
        if self.scatProb == 0:
            self.scatProb = 0
            self.set_outPol(1, 0, 0)     # temporary
        else:
            CosTheta = np.cos(self.outMomTheta)
            SinTheta = np.sin(self.outMomTheta)
            CosPhi = np.cos(self.outMomPhi)
            SinPhi = np.sin(self.outMomPhi)

            px1n, py1n, pz1n = CosPhi*SinTheta, SinPhi*SinTheta, CosTheta
            
            pol0 = self.get_midPol()
            k = np.array([px1n, py1n, pz1n])
            cosAng = pol0.dot(k) / np.sqrt(k.dot(k)) / np.sqrt(pol0.dot(pol0))
            if  np.abs(cosAng-1)<1e-5:
                l1 = vm.perpendicular_vector(pol0)
                l2 = np.cross(pol0, l1)
                beta = random.uniform(0, 2*np.pi)
                tmp = np.cos(beta) * l1 + np.sin(beta) * l2
                pol1 = self.normalize(tmp[0], tmp[1], tmp[2])
            else:
                tmp = pol0 - np.sqrt(pol0.dot(pol0)) * cosAng * k
                pol1 = self.normalize(tmp[0], tmp[1], tmp[2])

            self.set_outPol(pol1[0], pol1[1], pol1[2]) 
            

            #scale *= (self.midPol.dot(self.outPol))**2

            scale *= (np.matmul(self.outPol, np.matmul(T_new, inPol1)))**2

        self.scatProb = scale
    


    
    def calculatePol_locally(self):

        self.rotate_inPol()

        vec = np.array([1, 0, 0])
        ang, ax = rot.rotationAngle(self.inPol, vec), rot.rotationAxis(self.inPol, vec)
        mat = rot.RotationMatrix(ang, ax)
        #print("Rotation matrix", mat)

        local_outMom = np.matmul(mat, self.outMom)
        #print("local outMom : ", local_outMom)

        CosTheta = local_outMom[2] / np.sqrt(local_outMom.dot(local_outMom))
        SinTheta = np.sqrt(1 - CosTheta**2)
        CosPhi = local_outMom[0] / np.sqrt(local_outMom[0]**2 + local_outMom[1]**2)
        SinPhi = local_outMom[1] / np.sqrt(local_outMom[0]**2 + local_outMom[1]**2)

        px1n, py1n, pz1n = CosPhi*SinTheta, SinPhi*SinTheta, CosTheta

        if py1n == 0 and pz1n == 0 :
            beta = random.uniform(0, 1) * 2 * np.pi
            SinBeta, CosBeta = np.sin(beta), np.cos(beta)

        else:
            beta = 0   # required at the plane 
            if random.uniform(0, 1) < 0.5:
                beta = np.pi
            SinBeta, CosBeta = np.sin(beta), np.cos(beta)


        N = np.sqrt(1 - SinTheta**2*CosPhi**2)
        ex1, ey1, ez1 = N * CosBeta, SinTheta**2*SinPhi*CosPhi/N*CosBeta, SinTheta*CosTheta*CosPhi/N*CosBeta
        ex1n, ey1n, ez1n = self.normalize(ex1, ey1, ez1)
        local_outPol = np.array([ex1n, ey1n, ez1n])
        global_outPol = np.matmul(np.linalg.inv(mat), local_outPol)

        self.set_outPol(global_outPol[0], global_outPol[1], global_outPol[2])
        



    def calculatePol(self):
        
        if self.scatProb == 0:
            self.scatProb = 0
            self.set_outPol(1, 0, 0)     # temporary
        else:
            CosTheta = np.cos(self.outMomTheta)
            SinTheta = np.sin(self.outMomTheta)
            CosPhi = np.cos(self.outMomPhi)
            SinPhi = np.sin(self.outMomPhi)

            px1n, py1n, pz1n = CosPhi*SinTheta, SinPhi*SinTheta, CosTheta
            
            # This method is based on the paper description 
            #if py1n == 0 and pz1n == 0 :
            #    beta = random.uniform(0, 1) * 2 * np.pi
            #    SinBeta, CosBeta = np.sin(beta), np.cos(beta)

            #else:
            #    beta = 0   # required at the plane 
            #    if random.uniform(0, 1) < 0.5:
            #        beta = np.pi
            #    SinBeta, CosBeta = np.sin(beta), np.cos(beta)


            #N = np.sqrt(1 - SinTheta**2*CosPhi**2)
            #ex1, ey1, ez1 = N * CosBeta, SinTheta**2*SinPhi*CosPhi/N*CosBeta, SinTheta*CosTheta*CosPhi/N*CosBeta
            #ex1n, ey1n, ez1n = self.normalize(ex1, ey1, ez1)
            #self.set_outPol(ex1n, ey1n, ez1n)

            
            # personal implementations
            pol0 = self.get_inPol()
            k = np.array([px1n, py1n, pz1n])
            cosAng = pol0.dot(k) / np.sqrt(k.dot(k)) / np.sqrt(pol0.dot(pol0))
            if  np.abs(cosAng-1)<1e-5:
                l1 = vm.perpendicular_vector(pol0)
                l2 = np.cross(pol0, l1)
                beta = random.uniform(0, 2*np.pi)
                tmp = np.cos(beta) * l1 + np.sin(beta) * l2
                pol1 = self.normalize(tmp[0], tmp[1], tmp[2])
            else:
                tmp = pol0 - np.sqrt(pol0.dot(pol0)) * cosAng * k
                pol1 = self.normalize(tmp[0], tmp[1], tmp[2])

            self.set_outPol(pol1[0], pol1[1], pol1[2]) 

            self.scatProb = self.inPol.dot(self.outPol)**2
            self.scatProb *= self.scale



    def sampleMomPol(self):

        flag = True
        while flag:
            # sample momentum randomly
            CosTheta = random.uniform(0, 1)
            SinTheta = np.sqrt(1-CosTheta**2)
            if random.uniform(0, 1) < 0.5 :
                CosTheta = -CosTheta
            
            rand = 2*np.pi * random.uniform(0, 1)
            SinPhi = np.sin(rand)
            CosPhi = np.cos(rand)
 
            self.set_outMomTheta(np.arccos(CosTheta))
            self.set_outMomPhi(rand)

            energy = self.inE
            px1, py1, pz1 = energy*SinPhi*CosTheta, energy*SinPhi*SinTheta, energy*CosTheta
            self.set_outMom(px1, py1, pz1)
            px1n, py1n, pz1n = CosPhi*SinTheta, SinPhi*SinTheta, CosTheta
            ex0, ey0, ez0 = self.inPol[0], self.inPol[1], self.inPol[2]


            if py1n == 0 and pz1n == 0 :
                beta = random.uniform(0, 1) * 2 * np.pi
                SinBeta, CosBeta = np.sin(beta), np.cos(beta)

            else:
                beta = 0   # required at the plane 
                if random.uniform(0, 1) < 0.5:
                    beta = np.pi
                SinBeta, CosBeta = np.sin(beta), np.cos(beta)

            N = np.sqrt(1 - SinTheta**2*CosPhi**2)
            ex1_per, ey1_per, ez1_per = 0, CosTheta*SinBeta/N, -SinTheta*SinPhi/N
            ex1_par, ey1_par, ez1_par = N*CosBeta, -SinTheta**2*CosBeta*SinPhi*CosPhi/N, -SinTheta*CosTheta*CosPhi*CosBeta/N
            ex1, ey1, ez1 = ex1_par+ex1_per, ey1_par+ey1_per, ez1_par+ez1_per
            ex1n, ey1n, ez1n = self.normalize(ex1, ey1, ez1)
            self.set_outPol(ex1n, ey1n, ez1n)

            cos2Theta = CosBeta**2 * (1 - SinTheta**2*CosPhi**2)
            cosTheta = np.sqrt(cos2Theta)
            if ex1n*ex0+ey1n*ey0+ez1n*ez0 < 0:
                cosTheta = -cosTheta

            ### Sampling with CosTheta^2 Law
            if cos2Theta >= random.uniform(0, 1):
                self.set_outPolBeta(beta)
                self.set_polAngle(np.arccos(cosTheta))
                flag = False

    

