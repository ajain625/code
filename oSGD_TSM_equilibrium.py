import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

"""New, relevant code"""
# Useful quantities
def time_constants(d, lr, mean, mixing, varpos, varneg):
    varmix = mixing*varpos + (1-mixing)*varneg
    varsquaremix = mixing*varpos**2 + (1-mixing)*varneg**2
    N = np.dot(mean, mean)/d
    
    t1 = d/lr/(varmix + N)
    t2 = t1/2
    t3 = d/lr/varmix
    t4 = d/lr/(2*varmix - lr*varsquaremix)
    
    return t1, t2, t3, t4

def unicluster_egQ(var, lr):
    """Convenience function
    asymptotics can do this and more
    (0 mean only!)"""
    T = np.sqrt(2/np.pi/var)
    Q  = (lr*var*(4/np.pi - 1) - 4/np.pi)/(lr*var-2)/var
    eg  = (1-2/np.pi)/(1-(lr*var/2))
    return eg, Q, T

def asymptotics(d, lr, mean, mixing, varpos, varneg, teachpos, teachneg, biaspos, biasneg):
    # These order parameters don't change
    Mpos = np.dot(teachpos, teachpos)/d
    Mneg = np.dot(teachneg, teachneg)/d
    P = np.dot(teachpos, teachneg)/d
    Spos = np.dot(teachpos, mean)/d
    Sneg = np.dot(teachneg, mean)/d
    N = np.dot(mean, mean)/d

    varmix = mixing*varpos + (1-mixing)*varneg
    varsquaremix = mixing*varpos**2 + (1-mixing)*varneg**2

    # Constants
    betapos = np.sqrt(2*varpos/Mpos/np.pi)*np.exp(-(Spos+biaspos)**2/(2*Mpos*varpos))
    alphapos = 1 - 2*norm.cdf(-(Spos+biaspos)/np.sqrt(Mpos*varpos))
    betaneg = np.sqrt(2*varneg/Mneg/np.pi)*np.exp(-(-Sneg+biasneg)**2/(2*Mneg*varneg))
    alphaneg = 1 - 2*norm.cdf(-(-Sneg+biasneg)/np.sqrt(Mneg*varneg))

    
    Rinf = (mixing*(N*alphapos + Spos*betapos) + (1-mixing)*(-N*alphaneg + Sneg*betaneg))/(N+varmix)    
    Tposinf = (Spos*(mixing*alphapos - (1-mixing)*alphaneg - Rinf) + (mixing*betapos + P*(1-mixing)*betaneg))/(varmix)
    Tneginf = (Sneg*(mixing*alphapos - (1-mixing)*alphaneg - Rinf) + (P*mixing*betapos + (1-mixing)*betaneg))/(varmix)

    k1 = -2*varmix + lr*varsquaremix
    k2 = lr*varmix - 2
    k3 = 2*mixing*alphapos*(1-lr*varpos) - 2*(1-mixing)*alphaneg*(1-lr*varneg)
    k4 = 2*mixing*betapos*(1-lr*varpos)
    k5 = 2*(1-mixing)*betaneg*(1-lr*varneg)

    Qinf = -(Tposinf*k4 + Tneginf*k5 + lr*varmix)/k1 - Rinf*(Rinf*k2 + k3)/k1
    
    egposinf = 1 + Qinf*varpos + Rinf**2 - 2*(Rinf*alphapos + Tposinf*betapos)
    egneginf = 1 + Qinf*varneg + Rinf**2 - 2*(-Rinf*alphaneg + Tneginf*betaneg)
    eginf = mixing*egposinf + (1-mixing)*egneginf
    
    return eginf, egposinf, egneginf, Qinf, Tposinf, Tneginf, Rinf

def critical_mixing(d, lr, mean, mixing, varpos, varneg, teachpos, teachneg, biaspos, biasneg):
    def egposminusegneg(mixing, d, lr, mean, varpos, varneg, teachpos, teachneg, biaspos, biasneg):
        _, egposinf, egneginf, _, _, _, _ = asymptotics(d, lr, mean, mixing, varpos, varneg, teachpos, teachneg, biaspos, biasneg)
        return egposinf-egneginf
    mixing = fsolve(egposminusegneg, 0.5, args = (d, lr, mean, varpos, varneg, teachpos, teachneg, biaspos, biasneg))
    return mixing

def critical_lr(mixing, varpos, varneg):
    varmix = mixing*varpos + (1-mixing)*varneg
    varsquaremix = mixing*varpos**2 + (1-mixing)*varneg**2
    return 2*varmix/varsquaremix

def dip_time(d, mixing, varpos, varneg, P, lr, Q0=0, Tpos0=0 , Tneg0=0):
    """0 mean only!"""
    varmix = mixing*varpos + (1-mixing)*varneg
    varsquaremix = mixing*varpos**2 + (1-mixing)*varneg**2
    assert lr < 2*varmix/varsquaremix
    
    k2pos = mixing*np.sqrt(2*varpos/np.pi)*(2-2*varpos*lr)
    k2neg = (1-mixing)*np.sqrt(2*varneg/np.pi)*(2-2*varneg*lr)
    
    Tposinf = np.sqrt(2/np.pi)*(mixing*np.sqrt(varpos) + P*(1-mixing)*np.sqrt(varneg))/varmix
    Tneginf = np.sqrt(2/np.pi)*(P*mixing*np.sqrt(varpos) + (1-mixing)*np.sqrt(varneg))/varmix
    
    Qinf = (lr*varmix + Tposinf*k2pos + Tneginf*k2neg)/(2*varmix - lr*varsquaremix)
    alpha = (k2pos*(Tpos0 - Tposinf) + k2neg*(Tneg0 - Tneginf))/(lr*varsquaremix - varmix)
    
    t1pos = (2-lr*varsquaremix/varmix)*(Qinf - Q0 - alpha)/(np.sqrt(8/np.pi/varpos)*(Tposinf - Tpos0) - alpha)
    t1neg = (2-lr*varsquaremix/varmix)*(Qinf - Q0 - alpha)/(np.sqrt(8/np.pi/varneg)*(Tneginf - Tneg0) - alpha)

    if t1pos > 0 and t1neg > 0:
        print("both positive and negative")
        return d/lr/(varmix - lr*varsquaremix)*np.log(t1pos), d/lr/(varmix - lr*varsquaremix)*np.log(t1neg)
    elif t1pos > 0:
        print('positive')
        return d/lr/(varmix - lr*varsquaremix)*np.log(t1pos)
    elif t1neg > 0:
        print('negative')
        return d/lr/(varmix - lr*varsquaremix)*np.log(t1neg)
    else:
        print("no dip")
        
def time_constants(d, lr, mean, mixing, varpos, varneg):
    varmix = mixing*varpos + (1-mixing)*varneg
    varsquaremix = mixing*varpos**2 + (1-mixing)*varneg**2
    N = np.dot(mean, mean)/d
    
    t1 = d/lr/(varmix + N)
    t2 = t1/2
    t3 = d/lr/varmix
    t4 = d/lr/(2*varmix - lr*varsquaremix)
    
    return t1, t2, t3, t4

"""Old (probably) irrelevant code"""
# def stable_T(mixing, varpos, varneg):
#     varmix = mixing*varpos + (1-mixing)*varneg
#     return np.sqrt(2/np.pi)*(mixing*np.sqrt(varpos) - (1-mixing)*np.sqrt(varneg))/varmix

# def stable_Q(mixing, varpos, varneg, lr=None):
#     varmix = mixing*varpos + (1-mixing)*varneg
#     sigmadiff = mixing*np.sqrt(varpos) - (1-mixing)*np.sqrt(varneg)
#     sigmacubediff = mixing*np.sqrt(varpos**3) - (1-mixing)*np.sqrt(varneg**3)
#     varsquaremix = mixing*varpos**2 + (1-mixing)*varneg**2
#     if lr:
#         return (lr*varmix**2 - (4/np.pi)*(lr*sigmacubediff*sigmadiff - sigmadiff**2))/(2*varmix**2 - lr*varmix*varsquaremix)
#     else:
#         return (2/np.pi)*(sigmadiff)**2/sigmacubediff**2
    
# def stable_egpos(varpos, Q, T):
#     return 1 + varpos*Q - 2*T*np.sqrt(2*varpos/np.pi)

# def stable_egneg(varneg, Q, T):
#     return 1 + varneg*Q + 2*T*np.sqrt(2*varneg/np.pi)

# def equierror_Q(mixing, varpos, varneg):
#     varmix = mixing*varpos + (1-mixing)*varneg
#     sigmadiff = mixing*np.sqrt(varpos) - (1-mixing)*np.sqrt(varneg)
#     if varpos != varneg:
#         return (4/np.pi)*(np.sqrt(varpos) + np.sqrt(varneg))*sigmadiff/(varpos-varneg)/varmix
#     else:
#         return 0

# def stable_T_general_P(mixing, varpos, varneg, P):
#     varmix = mixing*varpos + (1-mixing)*varneg
#     Tpos = np.sqrt(2/np.pi)*(mixing*np.sqrt(varpos) + P*(1-mixing)*np.sqrt(varneg))/varmix
#     Tneg = np.sqrt(2/np.pi)*(P*mixing*np.sqrt(varpos) + (1-mixing)*np.sqrt(varneg))/varmix
#     return Tpos, Tneg

# def stable_Q_general_P(mixing, varpos, varneg, lr, P):
#     varmix = mixing*varpos + (1-mixing)*varneg
#     varsquaremix = mixing*varpos**2 + (1-mixing)*varneg**2
#     Tpos, Tneg = stable_T_general_P(mixing, varpos, varneg, P)
#     I1 = Tpos*np.sqrt(2*varpos/np.pi)
#     I2 = Tneg*np.sqrt(2*varneg/np.pi)
#     Q = (lr*varmix - 2*mixing*I1*(lr*varpos-1) -2*(1-mixing)*I2*(lr*varneg -1))/(2*varmix - lr*varsquaremix)
#     return Q

# def critical_lr_finder(start, stop, n, mixing, varpos, varneg, P, log=True):
#     if log:
#         lrs = np.logspace(start, stop, n)
#     else:
#         lrs = np.linspace(start, stop, n)
#     for lr in lrs:
#         Tpos, Tneg = stable_T_general_P(mixing, varpos, varneg, P)
#         Q = stable_Q_general_P(mixing, varpos, varneg, lr, P)
#         egpos = stable_egpos(varpos, Q, Tpos)
#         egneg = stable_egneg(varneg, Q, -Tneg)
#         if Q < 0 or egpos < 0 or egneg<0:
#             #print(f"Q={Q}, egpos={egpos}, egneg={egneg}")
#             return lr
#     print("not in given range")
#     return -1        

# def asymptotics(mixing, varpos, varneg, lr, P):
#     Tpos, Tneg = stable_T_general_P(mixing, varpos, varneg, P)
#     Q = stable_Q_general_P(mixing, varpos, varneg, lr, P)
#     egpos = stable_egpos(varpos, Q, Tpos)
#     egneg = stable_egneg(varneg, Q,  -Tneg)
#     return egpos, egneg, Q, Tpos, Tneg

# def critical_mixing(varpos, varneg, P, lr):
#     def egposminusegneg(mixing, varpos, varneg, lr, P):
#         Tpos, Tneg = stable_T_general_P(mixing, varpos, varneg, P)
#         Q = stable_Q_general_P(mixing, varpos, varneg, lr, P)
#         egpos = stable_egpos(varpos, Q, Tpos)
#         egneg = stable_egneg(varneg, Q,  -Tneg)
#         return egpos-egneg

#     mixing = fsolve(egposminusegneg, 0.8, args = (varpos, varneg, lr, P))
#     return mixing

# def unicluster_egQ(var, lr):
#     T = np.sqrt(2/np.pi/var)
#     Q  = (lr*var*(4/np.pi - 1) - 4/np.pi)/(lr*var-2)/var
#     eg  = (1-2/np.pi)/(1-(lr*var/2))
#     return eg, Q, T

# def check_critical_lr():
#     save_path = "/nfs/ghome/live/ajain/checkpoints/oSGD/bicluster_critical_lr.csv"
#     varposs = np.logspace(-3, 3, 30)
#     varnegs = np.logspace(-3, 3, 30)
#     mixings = np.linspace(0.1, 0.9, 9)
#     Ps = np.linspace(-1, 1, 21)
#     results = []

#     for varpos in varposs:
#         for varneg in varnegs:
#             for mixing in mixings:
#                 lr_predicted = 2*(mixing*varpos + (1-mixing)*varneg)/(mixing*varpos**2 + (1-mixing)*varneg**2)
#                 for P in Ps:
#                     lr = critical_lr_finder(0.8*lr_predicted, 1.2*lr_predicted, 50, mixing, varpos, varneg, P, log=False)
#                     lr_diff = np.abs(lr-lr_predicted)/lr_predicted
#                     results.append([varpos, varneg, mixing, P, lr_predicted, lr, lr_diff])
#             np.savetxt(save_path, results, delimiter=',', header='varpos, varneg, mixing, P, lr_predicted, lr, lr_diff')