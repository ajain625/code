import numpy as np
from scipy.stats import norm

def get_vector(vector, similarity):
    """given a vector and a desired cosine similarity, 
    returns another vector that has the desired cosine similarity
    (samples uniformly from space of possible vectors)"""
    target_norm = np.linalg.norm(vector)
    vector = vector/target_norm
    t = similarity
    x = np.random.normal(size=len(vector))
    u = np.sqrt(1-t**2)*(x-(np.dot(x,vector)*vector))/np.linalg.norm(x-(np.dot(x,vector)*vector))
    return (t*vector + u)*target_norm

def generate_data(T, d, mean, varpos, varneg, mixing, seed=0):
    np.random.seed(seed)
    Npos = int(np.round(mixing*T))
    Nneg = int(np.round((1-mixing)*T))
    posdata = np.random.multivariate_normal(mean=mean/np.sqrt(d) , cov=varpos*np.eye(d), size=Npos) 
    negdata = np.random.multivariate_normal(mean=-mean/np.sqrt(d) , cov=varneg*np.eye(d), size=Nneg)
    return posdata, negdata

def label_data(posdata, negdata, cosine_similarity, biaspos, biasneg):
    d = posdata.shape[1]
    data = np.vstack([posdata, negdata])
    T = data.shape[0]
    teachpos = np.random.normal(size=d)
    teachneg = get_vector(teachpos, cosine_similarity)
    poslabels = (posdata@teachpos/(d**0.5) + biaspos) >= 0
    neglabels = (negdata@teachneg/(d**0.5) + biasneg) >= 0
    labels = np.hstack([poslabels, neglabels])
    shuffle = np.random.permutation(T)
    data = data[shuffle]
    labels = labels[shuffle]
    ytrue = -np.ones(len(labels))
    ytrue[labels] = 1
    return data, ytrue

def oSGD(T, d, lr, classifier0, mean, mixing, varpos, varneg, teachpos, teachneg, biaspos, biasneg, seed=0):
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

    # Constants that appear frequently while computing integrals of the form <yW.x>
    k1 = np.sqrt(2*varpos/Mpos/np.pi)*np.exp(-(Spos+biaspos)**2/(2*Mpos*varpos))
    k2 = 1 - 2*norm.cdf(-(Spos+biaspos)/np.sqrt(Mpos*varpos))
    k3 = np.sqrt(2*varneg/Mneg/np.pi)*np.exp(-(-Sneg+biasneg)**2/(2*Mneg*varneg))
    k4 = 1 - 2*norm.cdf(-(-Sneg+biasneg)/np.sqrt(Mneg*varneg))

    # Integrals that don't change and are hence computed outside the for loop

    # I3, I4, I5, I5 appear in evolution of Tpos and Tneg
    I3 = Mpos*k1 + Spos*k2
    I4 = -Spos*(1-2*norm.cdf(-(-Sneg+biasneg)/np.sqrt(Mneg*varneg))) + P*np.sqrt(2*varneg/(Mneg*np.pi))*np.exp(-(-Sneg+biasneg)**2/(2*Mneg*varneg))

    I5 = Sneg*(1-2*norm.cdf(-(Spos+biaspos)/np.sqrt(Mpos*varpos))) + P*np.sqrt(2*varpos/(Mpos*np.pi))*np.exp(-(Spos+biaspos)**2/(2*Mpos*varpos))
    I6 = Mneg*k3 - Sneg*k4

    # Weighted sum of the first two integral terms involved in R
    I7 = mixing*(N*k2 + Spos*k1) + (1-mixing)*(-N*k4 + Sneg*k3)

    # Analytical results

    # Initialise generalisation error
    eg0 = 1 + varmix*Q0 + R0**2 - 2*mixing*(R0*k2 + Tpos0*k1) - 2*(1-mixing)*(-R0*k4 + Tneg0*k3)
    egpos0 = 1 + varpos*Q0 + R0**2 - 2*(R0*k2 + Tpos0*k1)
    egneg0 = 1 + varneg*Q0 + R0**2 - 2*(-R0*k4 + Tneg0*k3)

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
        # Needed for generalisation error and Q
        I1 = Rs[i-1]*k2 + Tposs[i-1]*k1
        I2 = -Rs[i-1]*k4 + Tnegs[i-1]*k3

        Q += lr/d*(1-egs[i-1] - Rs[i-1]**2 - Qs[i-1]*varmix) + lr**2/d*((1+Rs[i-1]**2)*varmix + Qs[i-1]*(mixing*varpos**2 + (1-mixing)*varneg**2) -2*mixing*varpos*I1 + -2*(1-mixing)*varneg*I2)
        
        Tpos += lr/d*(mixing*I3 + (1-mixing)*I4 - Rs[i-1]*Spos - Tposs[i-1]*varmix)
        Tneg += lr/d*(mixing*I5 + (1-mixing)*I6 - Rs[i-1]*Sneg - Tnegs[i-1]*varmix)
        
        R += lr/d*(I7 - Rs[i-1]*(N+varmix))
        
        egpos = 1 + varpos*(Qs[i-1]) + Rs[i-1]**2 - 2*I1
        egneg = 1 + varneg*(Qs[i-1]) + Rs[i-1]**2 - 2*I2
        eg = mixing*egpos + (1-mixing)*egneg

        egs[i] = eg
        egposs[i] = egpos
        egnegs[i] = egneg
        Qs[i] = Q
        Rs[i] = R
        Tposs[i] = Tpos
        Tnegs[i] = Tneg
    
    return egs, egposs, egnegs, Qs, Rs, Tposs, Tnegs

def error_analysis(egposs, egnegs):
    egposmin = np.min(egposs)
    egnegmin = np.min(egnegs)
    egposminind = np.argmin(egposs)
    egnegminind = np.argmin(egnegs)
    egposmax = np.max(egposs)
    egnegmax = np.max(egnegs)
    egposmaxind = np.argmax(egposs)
    egnegmaxind = np.argmax(egnegs)

    dis = egposs/egnegs

    dismin = np.min(dis)
    disminind = np.argmin(dis)
    dismax = np.max(dis)
    dismaxind = np.argmax(dis)
    di_start = dis[0]
    di_end = dis[-1]

    return egposmin, egposmax, egposminind, egposmaxind, egnegmin, egnegmax, egnegminind, egnegmaxind, dismin, dismax, disminind, dismaxind, di_start, di_end

def complete_analysis():
    np.random.seed(0)
    save_path = '/nfs/ghome/live/ajain/checkpoints/oSGD/first_results.csv'
    mixings = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    varposs = np.array([0.1, 0.5, 1])
    var_ratios = np.array([0.1, 0.5, 1, 2, 5, 10])
    cosine_similarities = np.array([-1, -0.5, -0.75, -0.1, 0, 0.1, 0.5, 0.75, 0.9])
    lrs = np.array([0.01, 0.05, 0.1, 0.2, 0.5])

    T = 50000
    d = 1000
    mean = np.ones(d)
    classifier0 = np.random.normal(size=d)
    biaspos = 0
    biasneg = 0

    results = []

    for mixing in mixings:
        for varpos in varposs:
            for var_ratio in var_ratios:
                varneg = varpos*var_ratio
                for cosine_similarity in cosine_similarities:
                    teachpos = np.random.normal(size=d)
                    teachneg = get_vector(teachpos, cosine_similarity)
                    for lr in lrs:
                        print('...')
                        _, egposs, egnegs, _, _, Tposs, Tnegs = oSGD(T, d, lr, classifier0, mean, mixing, varpos, varneg, teachpos, teachneg, biaspos, biasneg)
                        egposmin, egposmax, egposminind, egposmaxind, egnegmin, egnegmax, egnegminind, egnegmaxind, dismin, dismax, disminind, dismaxind, di_start, di_end = error_analysis(egposs, egnegs)
                        print(f"mixing: {mixing}, varpos: {varpos}, varneg: {varneg}, cosine_similarity: {cosine_similarity}, lr: {lr}, egposmin: {egposmin}, egposmax: {egposmax}, egposminind: {egposminind}, egposmaxind: {egposmaxind}, egnegmin: {egnegmin}, egnegmax: {egnegmax}, egnegminind: {egnegminind}, egnegmaxind: {egnegmaxind}, dismin: {dismin}, dismax: {dismax}, disminind: {disminind}, dismaxind: {dismaxind}, di_start: {di_start}, di_end: {di_end}, Tpos_end: {Tposs[-1]}, Tneg_end: {Tnegs[-1]}")
                        results.append([mixing, varpos, varneg, cosine_similarity, lr, egposmin, egposmax, egposminind, egposmaxind, egnegmin, egnegmax, egnegminind, egnegmaxind, egposs[0], egposs[-1], egnegs[0], egnegs[-1], dismin, dismax, disminind, dismaxind, di_start, di_end, Tposs[0], Tposs[-1], Tnegs[0], Tnegs[-1]])
                        np.savetxt(save_path, results, delimiter=',', header='mixing, varpos, varneg, cosine_similarity, lr, egposmin, egposmax, egposminind, egposmaxind, egnegmin, egnegmax, egnegminind, egnegmaxind, egposs[0], egposs[-1], egnegs[0], egnegs[-1], dismin, dismax, disminind, dismaxind, di_start, di_end, Tposs_start, Tposs_end, Tnegs_start, Tnegs_end', fmt='%f')

if __name__ == '__main__':
    complete_analysis()