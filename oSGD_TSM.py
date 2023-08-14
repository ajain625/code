import numpy as np
from scipy.stats import norm
from scipy.stats.mvn import mvnun
import copy

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

def get_conf_matrix(mean, cov):
    """expects 2d parameters. the first corresponds to the prediction and
    the second to the ground truth"""
    TN, _ = mvnun(np.array([-np.inf, -np.inf]), np.array([0, 0]), mean, cov)
    TP, _  = mvnun(np.array([0, 0]), np.array([np.inf, np.inf]), mean, cov)
    FN, _ = mvnun(np.array([-np.inf, 0]), np.array([0, np.inf]), mean, cov)
    FP = 1 - TP - TN - FN
    return np.array([[TP, FN], [FP, TN]])

def oSGD(T, d, lr, classifier0, mean, mixing, varpos, varneg, teachpos, teachneg, biaspos, biasneg, seed=0, get_conf=False):
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
    
    if get_conf:
        conf_matrices = np.zeros((T+1, 2, 2, 2))
        conf_matrices[0, 0, :, :] = get_conf_matrix(np.array([R, Spos + biaspos]), np.array([[Q, Tpos], [Tpos, Mpos]]))
        conf_matrices[0, 1, :, :] = get_conf_matrix(np.array([-R, -Sneg + biasneg]), np.array([[Q, Tneg], [Tneg, Mneg]]))

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
        if get_conf:
            conf_matrices[i, 0, :, :] = get_conf_matrix(np.array([Rs[i-1], Spos + biaspos]), np.array([[Qs[i-1], Tposs[i-1]], [Tposs[i-1], Mpos]]))
            conf_matrices[i, 1, :, :] = get_conf_matrix(np.array([-Rs[i-1], -Sneg + biasneg]), np.array([[Qs[i-1], Tnegs[i-1]], [Tnegs[i-1], Mneg]]))
    
    if get_conf:
        return egs, egposs, egnegs, Qs, Rs, Tposs, Tnegs, conf_matrices
    return egs, egposs, egnegs, Qs, Rs, Tposs, Tnegs

def oSGD_lr_reweighed(T, d, lr, classifier0, mean, mixing, varpos, varneg, teachpos, teachneg, biaspos, biasneg, seed=0, get_conf=False):
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

    # Integral terms involved in R
    I8 = N*k2 + Spos*k1
    I9 = -N*k4 + Sneg*k3

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
    
    if get_conf:
        conf_matrices = np.zeros((T+1, 2, 2, 2))
        conf_matrices[0, 0, :, :] = get_conf_matrix(np.array([R, Spos + biaspos]), np.array([[Q, Tpos], [Tpos, Mpos]]))
        conf_matrices[0, 1, :, :] = get_conf_matrix(np.array([-R, -Sneg + biasneg]), np.array([[Q, Tneg], [Tneg, Mneg]]))
    
    lrpos = lr/varpos
    lrneg = lr/varneg

    for i in range(1, T+1):
        # Needed for generalisation error and Q
        I1 = Rs[i-1]*k2 + Tposs[i-1]*k1
        I2 = -Rs[i-1]*k4 + Tnegs[i-1]*k3

        Q += 2*lrpos*mixing/d*(I1 - R**2 - varpos*Q) + 2*lrneg*(1-mixing)/d*(I2 - R**2 - varneg*Q) + lrpos**2*mixing/d*(varpos + Q*varpos**2 + R**2*varpos - 2*varpos*I1) + lrneg**2*(1-mixing)/d*(varneg + Q*varneg**2 + R**2*varneg - 2*varneg*I2)
        #Q += lr/d*(1-egs[i-1] - Rs[i-1]**2 - Qs[i-1]*varmix) + lr**2/d*((1+Rs[i-1]**2)*varmix + Qs[i-1]*(mixing*varpos**2 + (1-mixing)*varneg**2) -2*mixing*varpos*I1 + -2*(1-mixing)*varneg*I2)
        Tpos += 1/d*(lrpos*mixing*I3 + lrneg*(1-mixing)*I4 - lrpos*mixing*Rs[i-1]*Spos - lrneg*(1-mixing)*Rs[i-1]*Spos - lrpos*mixing*Tposs[i-1]*varpos - lrneg*(1-mixing)*Tposs[i-1]*varneg)
        Tneg += 1/d*(lrpos*mixing*I5 + lrneg*(1-mixing)*I6 - lrpos*mixing*Rs[i-1]*Sneg - lrneg*(1-mixing)*Rs[i-1]*Sneg - lrpos*mixing*Tnegs[i-1]*varpos - lrneg*(1-mixing)*Tnegs[i-1]*varneg)
        
        R += 1/d*(lrpos*mixing*I8 + lrneg*(1-mixing)*I9 - lrpos*mixing*Rs[i-1]*N - lrneg*(1-mixing)*Rs[i-1]*N - lrpos*mixing*Rs[i-1]*varpos - lrneg*(1-mixing)*Rs[i-1]*varneg)
        
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
        if get_conf:
            conf_matrices[i, 0, :, :] = get_conf_matrix(np.array([Rs[i-1], Spos + biaspos]), np.array([[Qs[i-1], Tposs[i-1]], [Tposs[i-1], Mpos]]))
            conf_matrices[i, 1, :, :] = get_conf_matrix(np.array([-Rs[i-1], -Sneg + biasneg]), np.array([[Qs[i-1], Tnegs[i-1]], [Tnegs[i-1], Mneg]]))
    
    if get_conf:
        return egs, egposs, egnegs, Qs, Rs, Tposs, Tnegs, conf_matrices
    return egs, egposs, egnegs, Qs, Rs, Tposs, Tnegs

def generate_data(T, d, mean, varpos, varneg, mixing, seed=0):
    np.random.seed(seed)
    Npos = int(np.round(mixing*T))
    Nneg = int(np.round((1-mixing)*T))
    posdata = np.random.multivariate_normal(mean=mean/np.sqrt(d) , cov=varpos*np.eye(d), size=Npos) 
    negdata = np.random.multivariate_normal(mean=-mean/np.sqrt(d) , cov=varneg*np.eye(d), size=Nneg)
    return posdata, negdata

def label_data(posdata, negdata, teachpos, teachneg, biaspos, biasneg, get_cluster_labels = False):
    d = posdata.shape[1]
    data = np.vstack([posdata, negdata])
    T = data.shape[0]
    #teachpos = np.random.normal(size=d)
    #teachneg = get_vector(teachpos, cosine_similarity)
    poslabels = (posdata@teachpos/(d**0.5) + biaspos) >= 0
    neglabels = (negdata@teachneg/(d**0.5) + biasneg) >= 0
    labels = np.hstack([poslabels, neglabels])
    shuffle = np.random.permutation(T) 
    data = data[shuffle]
    labels = labels[shuffle]
    ytrue = -np.ones(len(labels))
    ytrue[labels] = 1
    if get_cluster_labels:
        cluster_labels = np.hstack([np.ones(len(poslabels)), -np.ones(len(neglabels))])
        cluster_labels = cluster_labels[shuffle]
        return data, ytrue, cluster_labels
    return data, ytrue

def simulate_oSGD(lr, data, ytrue, classifier0, k1, k2, k3, k4, mixing, varpos, varneg, mean, teachpos, teachneg):
    d = len(classifier0)
    classifier = copy.deepcopy(classifier0)
    classifiers = np.zeros(data.shape)
    errors = np.zeros(len(data))
    for i in range(len(data)):
        error = (ytrue[i] - np.dot(classifier, data[i])/np.sqrt(d))**2
        errors[i] = error
        classifier += data[i]*lr*(ytrue[i] - np.dot(classifier, data[i])/np.sqrt(d))/np.sqrt(d)
        classifiers[i, :] = classifier
    
    simQs = np.zeros(len(classifiers))
    simRs = np.zeros(len(classifiers))
    simTposs = np.zeros(len(classifiers))
    simTnegs = np.zeros(len(classifiers))
    simegs = np.zeros(len(classifiers))
    simegposs = np.zeros(len(classifiers))
    simegnegs = np.zeros(len(classifiers))
    
    varmix = mixing*varpos + (1-mixing)*varneg

    for i in range(len(classifiers)):
        simQs[i] = np.dot(classifiers[i], classifiers[i])/d
        simRs[i] = np.dot(classifiers[i], mean)/d
        simTposs[i] = np.dot(classifiers[i], teachpos)/d
        simTnegs[i] = np.dot(classifiers[i], teachneg)/d
        simegposs[i] = 1 + varpos*simQs[i] + simRs[i]**2 - 2*(simRs[i]*k2 + simTposs[i]*k1)
        simegnegs[i] = 1 + varneg*simQs[i] + simRs[i]**2 - 2*(-simRs[i]*k4 + simTnegs[i]*k3)
        simegs[i] = 1 + varmix*simQs[i] + simRs[i]**2 - 2*mixing*(simRs[i]*k2 + simTposs[i]*k1) - 2*(1-mixing)*(-simRs[i]*k4 + simTnegs[i]*k3)
    
    return simegs, simegposs, simegnegs, simQs, simRs, simTposs, simTnegs

def simulate_oSGD_lr_reweighed(lr, data, ytrue, cluster_labels, classifier0, k1, k2, k3, k4, mixing, varpos, varneg, mean, teachpos, teachneg):
    d = len(classifier0)
    classifier = copy.deepcopy(classifier0)
    classifiers = np.zeros(data.shape)
    errors = np.zeros(len(data))
    varmix = mixing*varpos + (1-mixing)*varneg
    print(lr)
    
    for i in range(len(data)):
        error = (ytrue[i] - np.dot(classifier, data[i])/np.sqrt(d))**2
        errors[i] = error
        if cluster_labels[i] == 1:
            classifier += data[i]*(lr/varpos)*(ytrue[i] - np.dot(classifier, data[i])/np.sqrt(d))/(np.sqrt(d))
        elif cluster_labels[i] == -1:
            classifier += data[i]*(lr/varneg)*(ytrue[i] - np.dot(classifier, data[i])/np.sqrt(d))/(np.sqrt(d))
        else:
            raise ValueError
        #classifier += data[i]*(lr/((varpos+varneg)/2 + ytrue[i]/2*(varpos-varneg)))*(ytrue[i] - np.dot(classifier, data[i])/np.sqrt(d))/(np.sqrt(d))
        classifiers[i, :] = classifier
    
    simQs = np.zeros(len(classifiers))
    simRs = np.zeros(len(classifiers))
    simTposs = np.zeros(len(classifiers))
    simTnegs = np.zeros(len(classifiers))
    simegs = np.zeros(len(classifiers))
    simegposs = np.zeros(len(classifiers))
    simegnegs = np.zeros(len(classifiers))
    

    for i in range(len(classifiers)):
        simQs[i] = np.dot(classifiers[i], classifiers[i])/d
        simRs[i] = np.dot(classifiers[i], mean)/d
        simTposs[i] = np.dot(classifiers[i], teachpos)/d
        simTnegs[i] = np.dot(classifiers[i], teachneg)/d
        simegposs[i] = 1 + varpos*simQs[i] + simRs[i]**2 - 2*(simRs[i]*k2 + simTposs[i]*k1)
        simegnegs[i] = 1 + varneg*simQs[i] + simRs[i]**2 - 2*(-simRs[i]*k4 + simTnegs[i]*k3)
        simegs[i] = 1 + varmix*simQs[i] + simRs[i]**2 - 2*mixing*(simRs[i]*k2 + simTposs[i]*k1) - 2*(1-mixing)*(-simRs[i]*k4 + simTnegs[i]*k3)
    
    return simegs, simegposs, simegnegs, simQs, simRs, simTposs, simTnegs

def complete_simulation(T, d, lr, classifier0, mean, mixing, varpos, varneg, teachpos, teachneg, biaspos, biasneg, reweighed=False):
    posdata, negdata = generate_data(T, d, mean, varpos, varneg, mixing)
    if reweighed:
        data, ytrue, cluster_labels = label_data(posdata, negdata, teachpos, teachneg, biaspos, biasneg, get_cluster_labels=True)
    else:
        data, ytrue = label_data(posdata, negdata, teachpos, teachneg, biaspos, biasneg, get_cluster_labels=False)
    del posdata, negdata

    # These order parameters don't change
    Mpos = np.dot(teachpos, teachpos)/d
    Mneg = np.dot(teachneg, teachneg)/d
    Spos = np.dot(teachpos, mean)/d
    Sneg = np.dot(teachneg, mean)/d

    # Constants that appear frequently while computing integrals of the form <yW.x>
    k1 = np.sqrt(2*varpos/Mpos/np.pi)*np.exp(-(Spos+biaspos)**2/(2*Mpos*varpos))
    k2 = 1 - 2*norm.cdf(-(Spos+biaspos)/np.sqrt(Mpos*varpos))
    k3 = np.sqrt(2*varneg/Mneg/np.pi)*np.exp(-(-Sneg+biasneg)**2/(2*Mneg*varneg))
    k4 = 1 - 2*norm.cdf(-(-Sneg+biasneg)/np.sqrt(Mneg*varneg))
    
    if reweighed:
        simegs, simegposs, simegnegs, simQs, simRs, simTposs, simTnegs = simulate_oSGD_lr_reweighed(lr, data, ytrue, cluster_labels, classifier0, k1, k2, k3, k4, mixing, varpos, varneg, mean, teachpos, teachneg)
    else:
        simegs, simegposs, simegnegs, simQs, simRs, simTposs, simTnegs = simulate_oSGD(lr, data, ytrue, classifier0, k1, k2, k3, k4, mixing, varpos, varneg, mean, teachpos, teachneg)
    
    
    return simegs, simegposs, simegnegs, simQs, simRs, simTposs, simTnegs

def error_analysis(egposs, egnegs, conf_matrices = None):
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

    if conf_matrices is not None:
        fnposmin = np.min(conf_matrices[:, 0, 0, 1])
        fnposminind = np.argmin(conf_matrices[:, 0, 0, 1])
        fnposmax = np.max(conf_matrices[:, 0, 0, 1])
        fnposmaxind = np.argmax(conf_matrices[:, 0, 0, 1])
        fnnegmin = np.min(conf_matrices[:, 1, 0, 1])
        fnnegminind = np.argmin(conf_matrices[:, 1, 0, 1])
        fnnegmax = np.max(conf_matrices[:, 1, 0, 1])
        fnnegmaxind = np.argmax(conf_matrices[:, 1, 0, 1])

        fpposmin = np.min(conf_matrices[:, 0, 1, 0])
        fpposminind = np.argmin(conf_matrices[:, 0, 1, 0])
        fpposmax = np.max(conf_matrices[:, 0, 1, 0])
        fpposmaxind = np.argmax(conf_matrices[:, 0, 1, 0])
        fpnegmin = np.min(conf_matrices[:, 1, 1, 0])
        fpnegminind = np.argmin(conf_matrices[:, 1, 1, 0])
        fpnegmax = np.max(conf_matrices[:, 1, 1, 0])
        fpnegmaxind = np.argmax(conf_matrices[:, 1, 1, 0])

        fnpos_start = conf_matrices[0, 0, 0, 1]
        fnpos_end = conf_matrices[-1, 0, 0, 1]
        fnneg_start = conf_matrices[0, 1, 0, 1]
        fnneg_end = conf_matrices[-1, 1, 0, 1]

        fppos_start = conf_matrices[0, 0, 1, 0]
        fppos_end = conf_matrices[-1, 0, 1, 0]
        fpneg_start = conf_matrices[0, 1, 1, 0]
        fpneg_end = conf_matrices[-1, 1, 1, 0]

        return egposmin, egposmax, egposminind, egposmaxind, egnegmin, egnegmax, egnegminind, egnegmaxind, dismin, dismax, disminind, dismaxind, di_start, di_end, fnposmin, fnposminind, fnposmax, fnposmaxind, fnnegmin, fnnegminind, fnnegmax, fnnegmaxind, fpposmin, fpposminind, fpposmax, fpposmaxind, fpnegmin, fpnegminind, fpnegmax, fpnegmaxind, fnpos_start, fnpos_end, fnneg_start, fnneg_end, fppos_start, fppos_end, fpneg_start, fpneg_end
    
    return egposmin, egposmax, egposminind, egposmaxind, egnegmin, egnegmax, egnegminind, egnegmaxind, dismin, dismax, disminind, dismaxind, di_start, di_end

def complete_analysis():
    np.random.seed(0)
    save_path = '/nfs/ghome/live/ajain/checkpoints/oSGD/first_results_conf_matrices.csv'
    mixings = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    varposs = np.array([0.05, 0.1, 0.5, 1])
    var_ratios = np.array([0.1, 0.5, 1, 2, 5, 10])
    cosine_similarities = np.array([-1, -0.75, 0, 0.5, 0.7, 0.8, 0.9])
    #lrs = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
    lrs = np.array([0.2])

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
                        _, egposs, egnegs, _, _, Tposs, Tnegs, conf_matrices = oSGD(T, d, lr, classifier0, mean, mixing, varpos, varneg, teachpos, teachneg, biaspos, biasneg, get_conf=True)
                        egposmin, egposmax, egposminind, egposmaxind, egnegmin, egnegmax, egnegminind, egnegmaxind, dismin, dismax, disminind, dismaxind, di_start, di_end, fnposmin, fnposminind, fnposmax, fnposmaxind, fnnegmin, fnnegminind, fnnegmax, fnnegmaxind, fpposmin, fpposminind, fpposmax, fpposmaxind, fpnegmin, fpnegminind, fpnegmax, fpnegmaxind, fnpos_start, fnpos_end, fnneg_start, fnneg_end, fppos_start, fppos_end, fpneg_start, fpneg_end = error_analysis(egposs, egnegs, conf_matrices)
                        print(f"mixing: {mixing}, varpos: {varpos}, varneg: {varneg}, cosine_similarity: {cosine_similarity}, lr: {lr}, egposmin: {egposmin}, egposmax: {egposmax}, egposminind: {egposminind}, egposmaxind: {egposmaxind}, egnegmin: {egnegmin}, egnegmax: {egnegmax}, egnegminind: {egnegminind}, egnegmaxind: {egnegmaxind}, dismin: {dismin}, dismax: {dismax}, disminind: {disminind}, dismaxind: {dismaxind}, di_start: {di_start}, di_end: {di_end}, Tpos_end: {Tposs[-1]}, Tneg_end: {Tnegs[-1]}, fnposmin : {fnposmin}, fnposminind: {fnposminind}, fnposmax: {fnposmax}, fnposmaxind: {fnposmaxind}, fnnegmin: {fnnegmin}, fnnegminind: {fnnegminind}, fnnegmax: {fnnegmax}, fnnegmaxind: {fnnegmaxind}, fpposmin: {fpposmin}, fpposminind: {fpposminind}, fpposmax: {fpposmax}, fpposmaxind: {fpposmaxind}, fpnegmin: {fpnegmin}, fpnegminind: {fpnegminind}, fpnegmax: {fpnegmax}, fpnegmaxind: {fpnegmaxind}, fnpos_start: {fnpos_start}, fnpos_end: {fnpos_end}, fnneg_start: {fnneg_start}, fnneg_end: {fnneg_end}, fppos_start: {fppos_start}, fppos_end: {fppos_end}, fpneg_start: {fpneg_start}, fpneg_end: {fpneg_end}")
                        results.append([mixing, varpos, varneg, cosine_similarity, lr, egposmin, egposmax, egposminind, egposmaxind, egnegmin, egnegmax, egnegminind, egnegmaxind, egposs[0], egposs[-1], egnegs[0], egnegs[-1], dismin, dismax, disminind, dismaxind, di_start, di_end, Tposs[0], Tposs[-1], Tnegs[0], Tnegs[-1], fnposmin, fnposminind, fnposmax, fnposmaxind, fnnegmin, fnnegminind, fnnegmax, fnnegmaxind, fpposmin, fpposminind, fpposmax, fpposmaxind, fpnegmin, fpnegminind, fpnegmax, fpnegmaxind, fnpos_start, fnpos_end, fnneg_start, fnneg_end, fppos_start, fppos_end, fpneg_start, fpneg_end])
                        np.savetxt(save_path, results, delimiter=',', header='mixing, varpos, varneg, cosine_similarity, lr, egposmin, egposmax, egposminind, egposmaxind, egnegmin, egnegmax, egnegminind, egnegmaxind, egposs[0], egposs[-1], egnegs[0], egnegs[-1], dismin, dismax, disminind, dismaxind, di_start, di_end, Tposs_start, Tposs_end, Tnegs_start, Tnegs_end, fnposmin, fnposmindind, fnposmax, fnposmaxind, fnnegmin, fnnegminind, fnnegmax, fnnegmaxind, fpposmin, fpposminind, fpposmax, fpposmaxind, fpnegmin, fpnegminind, fpnegmax, fpnegmaxind, fnpos_start, fnpos_end, fnneg_start, fnneg_end, fppos_start, fppos_end, fpneg_start, fpneg_end ', fmt='%f')

if __name__ == '__main__':
    complete_analysis()