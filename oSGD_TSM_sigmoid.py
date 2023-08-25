import numpy as np 
import matplotlib.pyplot as plt
import copy
import math

def get_conf_matrix(A, B, D):
    """expects 2d parameters. the first corresponds to the prediction and
    the second to the ground truth"""
    TP = 1/(2*np.pi)*(np.pi/2 + np.arctan(D/np.sqrt(A*B-D**2)))
    return np.array([[TP, 1/2 - TP], [1/2 - TP, TP]])
    
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

def oSGD(T, d, lr, classifier0, mixing, varpos, varneg, teachpos, teachneg, get_conf = False):
    Q0 = np.dot(classifier0, classifier0)/d
    Tpos0 = np.dot(classifier0, teachpos)/d
    Tneg0 = np.dot(classifier0, teachneg)/d

    # These order parameters don't change
    Mpos = np.dot(teachpos, teachpos)/d
    Mneg = np.dot(teachneg, teachneg)/d
    P = np.dot(teachpos, teachneg)/d
    print(f'P: {P}, Mpos: {Mpos}, Mneg: {Mneg}')
    
    #Useful Constants
    k1 = 2/np.pi
    k2 = 4/np.pi**2

    # Initialise generalisation error
    egpos0 = 1/2 + k1/2*np.arcsin(varpos*Q0/(1+varpos*Q0)) - k1*np.arcsin(varpos*Tpos0/np.sqrt(varpos*Mpos*(1+varpos*Q0)))
    egneg0 = 1/2 + k1/2*np.arcsin(varneg*Q0/(1+varneg*Q0)) - k1*np.arcsin(varneg*Tneg0/np.sqrt(varneg*Mneg*(1+varneg*Q0)))
    eg0 = mixing*egpos0 + (1-mixing)*egneg0


    # Create arrays to store order parameters and initialise them
    Q = Q0
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
    Tposs = np.zeros(T+1)
    Tposs[0]  = Tpos
    Tnegs = np.zeros(T+1)
    Tnegs[0] = Tneg  
    
    if get_conf:
        conf_matrices = np.zeros((T+1, 2, 2, 2))
        conf_matrices[0, 0, :, :] = get_conf_matrix(varpos*Q, varpos*Mpos, varpos*Tpos)
        conf_matrices[0, 1, :, :] = get_conf_matrix(varneg*Q, varneg*Mpos, varneg*Tneg)

    for i in range(1, T+1):
        
        # Useful Transients
        k3pos = 1+varpos*Q
        k3neg = 1+varneg*Q

        k4pos = np.sqrt(1+2*varpos*Q)
        k4neg = np.sqrt(1+2*varneg*Q)

        k5pos = 1+3*varpos*Q
        k5neg = 1+3*varneg*Q
        
        # Terms involved in order parameter updates
        t1 = k1*np.sqrt(Mpos*varpos*(k3pos) - varpos**2*Tpos**2)/(1+Q*varpos)
        t2 = k1*varpos*Tpos/(k3pos)/k4pos
        t3 = k1*((Tpos*Mneg - P*Tneg)/(Q*Mneg - Tneg**2)*(varneg*Tneg)*np.sqrt(varneg*Q)/(k3neg)/np.sqrt((varneg**2*Mneg*Q - Tneg**2*varneg**2)*(k3neg) + Tneg**2*varneg**2) + (P*Q - Tpos*Tneg)/(Q*Mneg - Tneg**2)*np.sqrt(varneg*Mneg*(k3neg) -Tneg**2*varneg**2)/(k3neg))
        t4 = k1*varneg*Tpos/(k3neg)/k4neg
        
        t7 = k1*np.sqrt(Mneg*varneg*(k3neg) - varneg**2*Tneg**2)/(1+Q*varneg)
        t6 = k1*varpos*Tneg/(k3pos)/k4pos
        t5 = k1*((Tneg*Mpos - P*Tpos)/(Q*Mpos - Tpos**2)*(varpos*Tpos)*np.sqrt(varpos*Q)/(k3pos)/np.sqrt((varpos**2*Mpos*Q - Tpos**2*varpos**2)*(k3pos) + Tpos**2*varpos**2) + (P*Q - Tneg*Tpos)/(Q*Mpos - Tpos**2)*np.sqrt(varpos*Mpos*(k3pos) -Tpos**2*varpos**2)/(k3pos))
        t8 = k1*varneg*Tneg/(k3neg)/k4neg
        
        t9 = k1*varpos*Tpos*np.sqrt(varpos*Q)/(k3pos)/np.sqrt((varpos*Mpos*varpos*Q - varpos**2*Tpos**2)*(k3pos) + varpos**2*Tpos**2)
        t10 = k1*varpos*Q/(k3pos)/k4pos
        
        t11 = k1*varneg*Tneg*np.sqrt(varneg*Q)/(k3neg)/np.sqrt((varneg*Mneg*varneg*Q - varneg**2*Tneg**2)*(k3neg) + varneg**2*Tneg**2)
        t12 = k1*varneg*Q/(k3neg)/k4neg
        
        t13 = k1/k4pos
        t14 = k2/k4pos*np.arcsin(varpos*Q/(k5pos))
        t15 = k2/k4pos*np.arcsin(varpos*Tpos*np.sqrt(varpos*Q)/np.sqrt(k5pos)/np.sqrt((1+2*varpos*Q)*(varpos*Mpos*varpos*Q - varpos**2*Tpos**2) + varpos**2*Tpos**2))
        
        t16 = k1/k4neg
        t17 = k2/k4neg*np.arcsin(varneg*Q/(k5neg))
        t18 = k2/k4neg*np.arcsin(varneg*Tneg*np.sqrt(varneg*Q)/np.sqrt(k5neg)/np.sqrt((1+2*varneg*Q)*(varneg*Mneg*varneg*Q - varneg**2*Tneg**2) + varneg**2*Tneg**2))

        # Order Parameter Updates
        Q += 2*lr/d*(mixing*(t9-t10) + (1-mixing)*(t11-t12)) + lr**2/d*(mixing*varpos*(t13+t14-2*t15) + (1-mixing)*varneg*(t16 + t17 -2*t18))

        Tpos += lr/d*(mixing*(t1-t2) + (1-mixing)*(t3-t4))
        Tneg += lr/d*(mixing*(t5-t6) + (1-mixing)*(t7-t8))
        
        # Generalisation Error
        egpos = 1/2 + k1/2*np.arcsin(varpos*Q/(k3pos)) - k1*np.arcsin(varpos*Tpos/np.sqrt(varpos*Mpos*(k3pos)))
        egneg = 1/2 + k1/2*np.arcsin(varneg*Q/(k3neg)) - k1*np.arcsin(varneg*Tneg/np.sqrt(varneg*Mneg*(k3neg)))
        eg = mixing*egpos + (1-mixing)*egneg

        egs[i] = eg
        egposs[i] = egpos
        egnegs[i] = egneg
        Qs[i] = Q
        Tposs[i] = Tpos
        Tnegs[i] = Tneg
        if get_conf:
            conf_matrices[i, 0, :, :] = get_conf_matrix(varpos*Q, varpos*Mpos, varpos*Tpos)
            conf_matrices[i, 1, :, :] = get_conf_matrix(varneg*Q, varneg*Mpos, varneg*Tneg)
    
    if get_conf:
        return egs, egposs, egnegs, Qs, Tposs, Tnegs, conf_matrices
    return egs, egposs, egnegs, Qs, Tposs, Tnegs 

def generate_data(T, d, varpos, varneg, mixing, seed=0):
    np.random.seed(seed)
    Npos = int(np.round(mixing*T))
    Nneg = int(np.round((1-mixing)*T))
    posdata = np.random.multivariate_normal(mean=np.zeros(d) , cov=varpos*np.eye(d), size=Npos) 
    negdata = np.random.multivariate_normal(mean=np.zeros(d) , cov=varneg*np.eye(d), size=Nneg)
    return posdata, negdata

def label_data(posdata, negdata, teachpos, teachneg, get_cluster_labels = False):
    d = posdata.shape[1]
    data = np.vstack([posdata, negdata])
    T = data.shape[0]
    #teachpos = np.random.normal(size=d)
    #teachneg = get_vector(teachpos, cosine_similarity)
    poslabels = (posdata@teachpos/(d**0.5)) >= 0
    neglabels = (negdata@teachneg/(d**0.5)) >= 0
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

def simulate_oSGD(lr, data, ytrue, classifier0, mixing, varpos, varneg, teachpos, teachneg):
    d = len(classifier0)
    classifier = copy.deepcopy(classifier0)
    classifiers = np.zeros(data.shape)
    errors = np.zeros(len(data))
    for i in range(len(data)):
        error = ytrue[i] - math.erf(np.dot(classifier, data[i])/np.sqrt(2*d))
        errors[i] = error**2/2
        classifier += lr/np.sqrt(d)*error*(np.sqrt(2/np.pi)*np.exp(-np.dot(classifier, data[i])**2/(2*d)))*data[i]
        classifiers[i, :] = classifier
    
    simQs = np.zeros(len(classifiers))
    simTposs = np.zeros(len(classifiers))
    simTnegs = np.zeros(len(classifiers))
    simegs = np.zeros(len(classifiers))
    simegposs = np.zeros(len(classifiers))
    simegnegs = np.zeros(len(classifiers))
    
    Mpos = np.linalg.norm(teachpos)**2/d
    Mneg = np.linalg.norm(teachneg)**2/d

    for i in range(len(classifiers)):
        simQs[i] = np.dot(classifiers[i], classifiers[i])/d
        simTposs[i] = np.dot(classifiers[i], teachpos)/d
        simTnegs[i] = np.dot(classifiers[i], teachneg)/d
        simegposs[i] = 1/2 + 1/np.pi*np.arcsin(varpos*simQs[i]/(1+varpos*simQs[i])) - 2/np.pi*np.arcsin(varpos*simTposs[i]/np.sqrt(varpos*Mpos*(1+varpos*simQs[i])))
        simegnegs[i] = 1/2 + 1/np.pi*np.arcsin(varneg*simQs[i]/(1+varneg*simQs[i])) - 2/np.pi*np.arcsin(varneg*simTnegs[i]/np.sqrt(varneg*Mneg*(1+varneg*simQs[i])))
        simegs[i] = mixing*simegposs[i] + (1-mixing)*simegnegs[i]
    
    return simegs, simegposs, simegnegs, simQs, simTposs, simTnegs

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

def complete_simulation(T, d, lr, classifier0, mixing, varpos, varneg, teachpos, teachneg, reweighed=False):
    posdata, negdata = generate_data(T, d, varpos, varneg, mixing)
    if reweighed:
        data, ytrue, cluster_labels = label_data(posdata, negdata, teachpos, teachneg, get_cluster_labels=True)
    else:
        data, ytrue = label_data(posdata, negdata, teachpos, teachneg, get_cluster_labels=False)
    del posdata, negdata

    
    if reweighed:
        simegs, simegposs, simegnegs, simQs, simTposs, simTnegs = simulate_oSGD_lr_reweighed(lr, data, ytrue, cluster_labels, classifier0, mixing, varpos, varneg, teachpos, teachneg)
    else:
        simegs, simegposs, simegnegs, simQs, simTposs, simTnegs = simulate_oSGD(lr, data, ytrue, classifier0, mixing, varpos, varneg, teachpos, teachneg)
    
    return simegs, simegposs, simegnegs, simQs, simTposs, simTnegs
