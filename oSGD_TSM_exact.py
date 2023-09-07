import numpy as np 
from scipy.stats import norm

def TSMexact(T, d, lr, classifier0, mean, mixing, varpos, varneg, teachpos, teachneg, biaspos, biasneg):
    Q0 = np.dot(classifier0, classifier0)/d
    R0 = np.dot(classifier0, mean)/d
    Tpos0 = np.dot(classifier0, teachpos)/d
    Tneg0 = np.dot(classifier0, teachneg)/d

    # These order parameters don't change
    Mpos = np.dot(teachpos, teachpos)/d
    Mneg = np.dot(teachneg, teachneg)/d
    P = np.dot(teachpos, teachneg)/d
    Spos = np.dot(teachpos, mean)/d
    Sneg = np.dot(teachneg, mean)/d
    N = np.dot(mean, mean)/d
    print(f'P: {P}, Mpos: {Mpos}, Mneg: {Mneg}, Spos: {Spos}, Sneg: {Sneg}, N: {N}')

    varmix = mixing*varpos + (1-mixing)*varneg
    varsquaremix = mixing*varpos**2 + (1-mixing)*varneg**2

    # Constants
    betapos = np.sqrt(2*varpos/Mpos/np.pi)*np.exp(-(Spos+biaspos)**2/(2*Mpos*varpos))
    alphapos = 1 - 2*norm.cdf(-(Spos+biaspos)/np.sqrt(Mpos*varpos))
    betaneg = np.sqrt(2*varneg/Mneg/np.pi)*np.exp(-(-Sneg+biasneg)**2/(2*Mneg*varneg))
    alphaneg = 1 - 2*norm.cdf(-(-Sneg+biasneg)/np.sqrt(Mneg*varneg))

    c1 = lr*(N+varmix)/d
    exp1 = np.exp(-c1*np.arange(T+1))
    exp1squared = np.exp(-2*c1*np.arange(T+1))
    
    Rinf = (mixing*(N*alphapos + Spos*betapos) + (1-mixing)*(-N*alphaneg + Sneg*betaneg))/(N+varmix)
    Rs = Rinf*(1-exp1) + R0*exp1

    c2 = lr*varmix/d
    exp2 = np.exp(-c2*np.arange(T+1))
    
    Tposinf = (Spos*(mixing*alphapos - (1-mixing)*alphaneg - Rinf) + (mixing*betapos + P*(1-mixing)*betaneg))/(varmix)
    Tneginf = (Sneg*(mixing*alphapos - (1-mixing)*alphaneg - Rinf) + (P*mixing*betapos + (1-mixing)*betaneg))/(varmix)
    if N != 0:
        Tpostrans = Spos/N*(Rinf-R0)
        Tnegtrans = Sneg/N*(Rinf-R0)
    else:
        Tpostrans = 0
        Tnegtrans = 0
    
    Tposs = Tpos0*exp2 + Tposinf*(1-exp2) + Tpostrans*(exp2 - exp1)
    Tnegs = Tneg0*exp2 + Tneginf*(1-exp2) + Tnegtrans*(exp2 - exp1)

    k1 = -2*varmix + lr*varsquaremix
    k2 = lr*varmix - 2
    k3 = 2*mixing*alphapos*(1-lr*varpos) - 2*(1-mixing)*alphaneg*(1-lr*varneg)
    k4 = 2*mixing*betapos*(1-lr*varpos)
    k5 = 2*(1-mixing)*betaneg*(1-lr*varneg)

    c3 = lr/d*(-k1)
    exp3 = np.exp(-c3*np.arange(T+1))

    Qinf = -(Tposinf*k4 + Tneginf*k5 + lr*varmix)/k1 - Rinf*(Rinf*k2 + k3)/k1
    
    Qterm1 = Q0*exp3
    Qterm2 = Qinf*(1-exp3)
    Qterm3 = ((R0-Rinf)*k3 + 2*Rinf*k2*(R0 - Rinf) - Tpostrans*k4 - Tnegtrans*k5)*(exp3-exp1)/(N+varmix+k1)
    Qterm4 = (k2*(R0 - Rinf)**2)/(2*(N+varmix) + k1)*(exp3 - exp1squared)
    Qterm5 = ((Tpos0-Tposinf)*k4 + (Tneg0-Tneginf)*k5 + Tpostrans*k4 + Tnegtrans*k5)*(exp3-exp2)/(varmix+k1)
    
    Qs = Qterm1 + Qterm2 + Qterm3 + Qterm4 + Qterm5
    
    egposs = 1 + Qs*varpos + Rs**2 - 2*(Rs*alphapos + Tposs*betapos)
    egnegs = 1 + Qs*varneg + Rs**2 - 2*(-Rs*alphaneg + Tnegs*betaneg)
    egs = mixing*egposs + (1-mixing)*egnegs
    
    return egs, egposs, egnegs, Qs, Tposs, Tnegs, Rs