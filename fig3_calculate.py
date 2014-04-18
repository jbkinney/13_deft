'''
fig3_calculate.py 

Written by by Justin B. Kinney, Cold Spring Harbor Laboratory

Last updated on 15 December 2013 

Description:
    Simulates data and performs density estimates for Fig. 3 of
    Kinney, 2013. Takes about 30 min to execute on my laptop computer.   

Dependencies:
    scipy
    numpy
    sklearn
    time
    deft
    kinney2013_utils

Loads:
    None.
    
Saves:
    dists.npy    
            
Reference: 
    Kinney, J.B. (2013) Practical estimation of probability densities using scale-free field theories. arxiv preprint
'''

import scipy as sp
import numpy as np
from sklearn import mixture
from kinney2013_utils import make_gaussian_mix, draw_from_gaussian_mix, geo_dist
from scipy.stats import gaussian_kde
from deft import deft_1d
import time

start_time = time.clock()

# Number of grid points to use for DEFT calculation
G = 100

# Specify Ns and alphas
Ns = [100, 1000, 10000]
num_Ns = len(Ns)
num_gaussians = 5
plot_grid_size = 1000

# Number of trials to run
num_trials = 100
dists = np.zeros([num_trials,5,num_Ns])
  
for i, N in enumerate(Ns):
    print 'Performing trials for N = %d'%N
    for trial_num in range(num_trials):
        # User feedback
        print 'Running N = %d, trial_num = %d...'%(N, trial_num)
    
        # Choose mixture of gaussians
        gaussians = make_gaussian_mix(num_gaussians)
    
        # Draw data from mixture
        [xis, xgrid, Q_true, other] = draw_from_gaussian_mix(N=N, Nx=G, gaussians=gaussians)    

        # Compute data range and grid for fine-graned analysis
        xmin = min(xgrid)
        xmax = max(xgrid)
        xint = [xmin, xmax]
        xs = sp.linspace(xmin, xmax, plot_grid_size)
        dx = xs[1]-xs[0]
        
        # Interpolate Q_true for plotting
        Q_true_vals = Q_true(xs)
    
        # Perform DEFT density estimation
        Q_star1_vals = deft_1d(xis, xint, alpha=1, G=G, verbose=False)(xs)
        Q_star2_vals = deft_1d(xis, xint, alpha=2, G=G, verbose=False)(xs)        
        Q_star3_vals = deft_1d(xis, xint, alpha=3, G=G, verbose=False)(xs)
        
        # Perform GKDE denstiy estimation
        gkde = gaussian_kde(xis)
        Q_gkde_vals = gkde(xs)/sum(gkde(xs)*dx)
            
        # Perform GMM denstiy estimation using BIC 
        max_K = 10
        bic_values = sp.zeros([max_K]);
        Qs_gmm = sp.zeros([max_K,plot_grid_size])
        for k in sp.arange(1,max_K+1):
            gmm = mixture.GMM(int(k))
            gmm.fit(xis)
            Qgmm = lambda(x): sp.exp(gmm.score(x))/sum(sp.exp(gmm.score(xs))*dx)
            Qs_gmm[k-1,:] = Qgmm(xs)/sum(Qgmm(xs)*dx)
            bic_values[k-1] = gmm.bic(sp.array(xis)) 
        
        # Choose distribution with lowest BIC
        i_best = sp.argmin(bic_values)
        Q_gmm_vals = Qs_gmm[i_best,:]
        
        # Compute distances
        dists[trial_num,0, i] = geo_dist(Q_true_vals, Q_star1_vals, dx)
        dists[trial_num,1, i] = geo_dist(Q_true_vals, Q_star2_vals, dx)
        dists[trial_num,2, i] = geo_dist(Q_true_vals, Q_star3_vals, dx)
        dists[trial_num,3, i] = geo_dist(Q_true_vals, Q_gmm_vals, dx)
        dists[trial_num,4, i] = geo_dist(Q_true_vals, Q_gkde_vals, dx)
        
#np.save('dists.npy', dists)

print 'fig3_calculate.py took %.2f seconds to execute'%(time.clock()-start_time) 

