from oSGD_TSM import *

def stable_T(mixing, varpos, varneg):
    varmix = mixing*varpos + (1-mixing)*varneg
    return np.sqrt(2/np.pi)*(mixing*np.sqrt(varpos) - (1-mixing)*np.sqrt(varneg))/varmix

def stable_Q(mixing, varpos, varneg, lr=None):
    varmix = mixing*varpos + (1-mixing)*varneg
    sigmadiff = mixing*np.sqrt(varpos) - (1-mixing)*np.sqrt(varneg)
    sigmacubediff = mixing*np.sqrt(varpos**3) - (1-mixing)*np.sqrt(varneg**3)
    varsquaremix = mixing*varpos**2 + (1-mixing)*varneg**2
    if lr:
        return (lr*varmix**2 - (4/np.pi)*(lr*sigmacubediff*sigmadiff - sigmadiff**2))/(2*varmix**2 - lr*varmix*varsquaremix)
    else:
        return (2/np.pi)*(sigmadiff)**2/sigmacubediff**2
    
def stable_egpos(varpos, Q, T):
    return 1 + varpos*Q - 2*T*np.sqrt(2*varpos/np.pi)

def stable_egneg(varneg, Q, T):
    return 1 + varneg*Q + 2*T*np.sqrt(2*varneg/np.pi)

def equilibria_parameters():
    save_path = '/nfs/ghome/live/ajain/checkpoints/oSGD/equilibrias.csv'
    mixings = np.linspace(0,1,25)
    varposs = np.logspace(-2,2,25)
    varnegs = np.logspace(-2,2,25)
    lrs = np.logspace(-3,1,25)

    results = []

    for mixing in mixings:
        print(mixing)
        for varpos in varposs:
            print(varpos)
            for varneg in varnegs:
                print(varneg)
                stableT = stable_T(mixing, varpos, varneg)
                for lr in lrs:
                    print(lr)
                    stableQ = stable_Q(mixing, varpos, varneg, lr)
                    egpos = stable_egpos(varpos, stableQ, stableT)
                    egneg = stable_egneg(varneg, stableQ, stableT)
                    results.append([mixing, varpos, varneg, lr, stableT, stableQ, egpos, egneg])
                    np.savetxt(save_path, results, header='mixing, varpos, varneg, lr, stableT, stableQ, egpos, egneg', fmt='%s', delimiter=',')

if __name__ == '__main__':
    equilibria_parameters()
