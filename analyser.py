import ROOT
import numpy as np

def func(p0, p1, p2, x) :
    return p0 + p1*np.cos(x-p2)*np.cos(x-p2)

def read_data(theta, pol):
    vec_pol_angle = [179, 209, 239, 269, 299, 329, 359, 29, 59, 89, 119, 149]
    data = np.loadtxt("./exp/"+pol+str(int(theta))+"_1.txt")

    R    = data[:, 0]
    Rerr = data[:, 1]

    g1 = ROOT.TGraphErrors()
    for k in range(len(vec_pol_angle)):
        g1.SetPoint(k, vec_pol_angle[k]/180.*np.pi, R[k])
        g1.SetPointError(k, 0, Rerr[k])

    f1 = ROOT.TF1("f1", "[0] + [1]*cos(x-[2])*cos(x-[2])", 0, np.pi*2)
    g1.Fit(f1, "RE")
    p0, p1, p2 = f1.GetParameter(0), f1.GetParameter(1), f1.GetParameter(2)
    p0err, p1err, p2err = f1.GetParError(0), f1.GetParError(1), f1.GetParError(2)

    return R, Rerr, p0, p1, p2, p0err, p1err, p2err




def analyse_verData():
    theta = [75, 80, 85, 90, 95, 100, 105]
    vec_pol_angle = [179, 209, 239, 269, 299, 329, 359, 29, 59, 89, 119, 149]

    rhov, rhov_err = [], []
    Vv, Vverr, Hv, Hverr = [], [], [], []

    vec_pol_angle = [179, 209, 239, 269, 299, 329, 359, 29, 59, 89, 119, 149]
    for n, i in enumerate(theta):

        data = np.loadtxt("./exp/ver"+str(i)+"_1.txt")

        R    = data[:, 0]
        Rerr = data[:, 1]

        g1 = ROOT.TGraphErrors()
        for k in range(len(vec_pol_angle)):
            g1.SetPoint(k, vec_pol_angle[k]/180.*np.pi, R[k])
            g1.SetPointError(k, 0, Rerr[k])

        f1 = ROOT.TF1("f1", "[0] + [1]*cos(x-[2])*cos(x-[2])", 0, np.pi*2)
        g1.Fit(f1, "RE")
        p0, p1, p2 = f1.GetParameter(0), f1.GetParameter(1), f1.GetParameter(2)
        p0err, p1err, p2err = f1.GetParError(0), f1.GetParError(1), f1.GetParError(2)


        Vv.append(func(p0, p1, p2, p2))
        Vvmax = func(p0+p0err, p1+p1err, p2, p2)
        Vvmin = func(p0-p0err, p1-p1err, p2, p2)
        Vverr.append((Vvmax-Vvmin)/2.)

        Hv.append(func(p0, p1, p2, p2+np.pi/2.))
        Hvmax = func(p0+p0err, p1+p1err, p2, p2+np.pi/2.)
        Hvmin = func(p0-p0err, p1-p1err, p2, p2+np.pi/2.)
        Hverr.append((Hvmax-Hvmin)/2.)

        rhov.append(f1.Eval(f1.GetParameter(2)+np.pi/2) / f1.Eval(f1.GetParameter(2)))
        rhov_min = func(p0-p0err, p1-p1err, p2, p2+np.pi/2) / func(p0+p0err, p1+p1err, p2, p2)
        rhov_max = func(p0+p0err, p1+p1err, p2, p2+np.pi/2) / func(p0-p0err, p1-p1err, p2, p2)
        rhov_err.append((rhov_max - rhov_min)/2.)


    return Hv, Hverr, Vv, Vverr, rhov, rhov_err



def analyse_horData():

    theta = [75, 80, 85, 90, 95, 100, 105]
    vec_pol_angle = [179, 209, 239, 269, 299, 329, 359, 29, 59, 89, 119, 149]

    rhoh, rhoh_err = [], []
    Vh, Vherr, Hh, Hherr = [], [], [], []

    vec_pol_angle = [179, 209, 239, 269, 299, 329, 359, 29, 59, 89, 119, 149]
    for n, i in enumerate(theta):

        data = np.loadtxt("./exp/hor"+str(i)+"_1.txt")

        R    = data[:, 0]
        Rerr = data[:, 1]

        g1 = ROOT.TGraphErrors()
        for k in range(len(vec_pol_angle)):
            g1.SetPoint(k, vec_pol_angle[k]/180.*np.pi, R[k])
            g1.SetPointError(k, 0, Rerr[k])

        f1 = ROOT.TF1("f1", "[0] + [1]*cos(x-[2])*cos(x-[2])", 0, np.pi*2)
        g1.Fit(f1, "RE")
        p0, p1, p2 = f1.GetParameter(0), f1.GetParameter(1), f1.GetParameter(2)
        p0err, p1err, p2err = f1.GetParError(0), f1.GetParError(1), f1.GetParError(2)

        if p1 < 0:
            Vh.append(p0+p1)
            Vhmax = p0 + p1 + p0err + p1err
            Vhmin = p0 + p1 - p0err - p1err
            Vherr.append((Vhmax-Vhmin)/2.)

            Hh.append(p0)
            Hhmax = p0 + p0err
            Hhmin = p0 - p0err
            Hherr.append((Hhmax-Hhmin)/2.)

            print("Vh = %.3f, Hh = %.3f" %(Vh[-1], Hh[-1]))
    
        if p1 >= 0:
            Hh.append(p0+p1)
            Hhmax = p0 + p1 + p0err + p1err
            Hhmin = p0 + p1 - p0err - p1err
            Hherr.append((Hhmax-Hhmin)/2.)

            Vh.append(p0)
            Vhmax = p0 + p0err
            Vhmin = p0 - p0err
            Vherr.append((Vhmax-Vhmin)/2.)

            print("Vh = %.3f, Hh = %.3f" %(Vh[-1], Hh[-1]))

        rhoh.append(Hh[-1]/Vh[-1])
        rhoh_max = Hhmax / Vhmin
        rhoh_min = Hhmin / Vhmax
        rhoh_err.append((rhoh_max - rhoh_min)/2.)

    return Hh, Hherr, Vh, Vherr, rhoh, rhoh_err

