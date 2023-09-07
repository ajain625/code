import numpy as np

def oSGD(T, d, lr, classifier0, mean, mixing, varpos, varneg, teachpos, teachneg, biaspos, biasneg):
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

    # Initialise generalisation error
    egpos0 = (varpos*(Mpos+Q0-2*Tpos0) + (R0-Spos)**2)/2
    egneg0 = (varneg*(Mneg+Q0-2*Tneg0) + (R0-Sneg)**2)/2
    eg0 = mixing*egpos0 + (1-mixing)*egneg0

    # Create arrays to store order parameters and initialise them
    Q = Q0
    R = R0
    Tpos = Tpos0
    Tneg = Tneg0
    eg = eg0 
    egpos = egpos0
    egneg = egneg0

    egs = np.zeros(T+1)
    egs[0] = eg

    egposs = np.zeros(T+1)
    egposs[0] = egpos

    egnegs = np.zeros(T+1)
    egnegs[0] = egneg

    Qs = np.zeros(T+1)
    Qs[0] = Q
    Rs = np.zeros(T+1)
    Rs[0] = R
    Tposs = np.zeros(T+1)
    Tposs[0]  = Tpos
    Tnegs = np.zeros(T+1)
    Tnegs[0] = Tneg  


    for i in range(1, T+1):

        Q += 2*lr/d*(mixing*(Spos*Rs[i-1] + varpos*Tposs[i-1] - varpos*Qs[i-1] - Rs[i-1]**2) + (1-mixing)*(Sneg*Rs[i-1] + varneg*Tnegs[i-1] - varneg*Qs[i-1] -Rs[i-1]**2)) + lr**2/d*(mixing*varpos*(varpos*(Mpos+Qs[i-1]-2*Tposs[i-1]) + (Rs[i-1] - Spos)**2) +(1-mixing)*varneg*(varneg*(Mneg + Qs[i-1] - 2*Tnegs[i-1]) +(Rs[i-1]-Sneg)**2))
        
        Tpos += lr/d*(mixing*(varpos*Mpos + Spos**2 - Spos*Rs[i-1] - varpos*Tposs[i-1]) + (1-mixing)*(Spos*Sneg + varneg*P - Spos*Rs[i-1] - varneg*Tposs[i-1]))
        Tneg += lr/d*((1-mixing)*(varneg*Mneg + Sneg**2 - Sneg*Rs[i-1] - varneg*Tnegs[i-1]) + mixing*(Spos*Sneg + varpos*P - Sneg*Rs[i-1] - varpos*Tnegs[i-1]))

        R += lr/d*(mixing*(Spos*N + varpos*Spos - N*Rs[i-1] - varpos*Rs[i-1]) + (1-mixing)*(Sneg*N + varneg*Sneg - N*Rs[i-1] - varneg*Rs[i-1]))
        
        Qs[i] = Q
        Rs[i] = R
        Tposs[i] = Tpos
        Tnegs[i] = Tneg
        
        egpos = (varpos*(Mpos+Qs[i]-2*Tposs[i]) + (Rs[i]-Spos)**2)/2
        egneg = (varneg*(Mneg+Qs[i]-2*Tnegs[i]) + (Rs[i]-Sneg)**2)/2
        eg = mixing*egpos + (1-mixing)*egneg

        egs[i] = eg
        egposs[i] = egpos
        egnegs[i] = egneg

    return egs, egposs, egnegs, Qs, Rs, Tposs, Tnegs