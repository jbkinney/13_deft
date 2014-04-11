'''
fig1_fig2_calculate.py 

Written by by Justin B. Kinney, Cold Spring Harbor Laboratory

Last updated on 15 December 2013 

Description:
    Simulates data and performs density estimation reported in Figs. 1 and 2 of
    Kinney, 2013. Takes about 1 min to execute on my laptop computer.  

Dependencies:
    scipy
    time
    deft
    kinney2013_utils

Loads:
    None.
    
Saves:
    things_1d.pk    
            
Reference: 
    Kinney, J.B. (2013) Practical estimation of probability densities using scale-free field theories. arxiv preprint
'''

# Makes a simple figure demonstrating the inference proces. 
import scipy as sp
from kinney2013_utils import make_gaussian_mix, draw_from_gaussian_mix, save_object
from deft import deft_1d
import time

start_time = time.clock()

# Design data distribution
num_gaussians = 2
mus = [0, 10]
sigmas = [2, 5]
weights = [1, 1]
gaussians = make_gaussian_mix(mus=mus, sigmas=sigmas, weights=weights)

# Number of grid points to use for DEFT calculation
G = 100

# Specify N and alphas
N = 100
alpha = 2
num_samples = 20

# Draw data, rescaled to give an xint of length L = 10, centered on 0
xint = sp.array([-5.0, 5.0])
[xis, xint, Q_true, Q_true_details] = draw_from_gaussian_mix(N=N, Nx=G, gaussians=gaussians, xint=xint)    
xmid = sp.mean(xint)
xspan = xint - xmid
gaussians = Q_true_details.gaussians

# Perform DEFT density estimation
Q_star, Q_star_details = deft_1d(xis, xmid+xspan, alpha=alpha, G=G, details=True, num_samples=20, verbose=True)
Q_star_wide3, Q_star_wide3_details = deft_1d(xis, xmid+3*xspan, alpha=alpha, G=3*G, details=True, verbose=True)
Q_star_wide10, Q_star_wide10_details = deft_1d(xis, xmid+10*xint, alpha=alpha, G=10*G, details=True, verbose=True)
Q_star_fine, Q_star_fine_details = deft_1d(xis, xmid+xspan, alpha=alpha, G=3*G, details=True, verbose=True)
Q_star_coarse, Q_star_coarse_details = deft_1d(xis, xmid+xspan, alpha=alpha, G=int(G/3), details=True, verbose=True)
Q_star_alpha1, Q_star_alpha1_details = deft_1d(xis, xmid+xspan, alpha=1, G=G, details=True, verbose=True)
Q_star_alpha3, Q_star_alpha3_details = deft_1d(xis, xmid+xspan, alpha=3, G=G, details=True, verbose=True)

# Design plotting grid
xs = sp.linspace(xint[0], xint[1], 1000)

# Save everything
things = {}
things['Q_star_details'] = Q_star_details
things['Q_true_details'] = Q_true_details
things['Q_star_wide3_details'] = Q_star_wide3_details
things['Q_star_wide10_details'] = Q_star_wide10_details
things['Q_star_fine_details'] = Q_star_fine_details
things['Q_star_coarse_details'] = Q_star_coarse_details
things['Q_star_alpha1_details'] = Q_star_alpha1_details
things['Q_star_alpha3_details'] = Q_star_alpha3_details
things['xs'] = xs
things['xis'] = xis
things['xint'] = xint
things['xspan'] = xspan
things['xmid'] = xmid
things['N'] = N
things['G'] = G
things['gaussians'] = gaussians

# Save everything
save_object(things, 'things_1d.pk')
    
print 'fig1_fig2_calculate.py took %.2f seconds to execute'%(time.clock()-start_time) 
