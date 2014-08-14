'''
demo.py 

Written by by Justin B. Kinney, Cold Spring Harbor Laboratory

Last updated on 24 December 2013 

Description:
    Provides a simple demonstration of DEFT in 1D 

Dependencies:
    scipy
    time
    deft
            
Reference: 
    Kinney, J.B. (2013) "Practical estimation of probability densities using 
    scale-free field theories" arXiv:1312.6661 [physics.data-an]
'''

# Makes a simple figure demonstrating the inference proces. 
import scipy as sp
import matplotlib.pyplot as plt
from deft import deft_1d
import time 

###
### Synthesize data 
###

# Synthesize data (use mixture of two Gaussians)
N = 50
x1s = 1.0*sp.random.randn(int(N/2)) + 4.0
x2s = 1.0*sp.random.randn(N - int(N/2))
xs = sp.concatenate((x1s, x2s))
Q_true = lambda x: (sp.exp(-(x**2)/2.0)/(2*sp.sqrt(2*sp.pi))) + (sp.exp(-((x-4.0)**2)/2.0)/(2*sp.sqrt(2*sp.pi)))

###
### Perform density estimation using deft_1d
###

# Set number of grid points used by DEFT
G = 100

# Set power of derivative to constrain
alpha = 3

# Set bounding box used by DEFT
bbox = [-5.0, 10.0]

# Perform density estimation using DEFT
start_time = time.clock()

#
# DO THIS If all you want is the distribution function
#
#Q_star = deft_1d(xs, G=G, alpha=alpha, bbox=bbox)

#
# DO THIS If all you want details of the computation
#
Q_star, results = deft_1d(xs, G=G, alpha=alpha, bbox=bbox, details=True)

s =  'deft_1d with G=%d and alpha=%d took %.2f sec'%(G,alpha,time.clock()-start_time) 

###
### Plot results
###

# Close existing figure and create new figure
plt.close('all')
plt.figure()

# Plot histogram
plt.hist(xs, G, normed=1, histtype='stepfilled', edgecolor='none', facecolor='gray') 

# Plot estimated density and true density
xgrid = sp.linspace(bbox[0], bbox[1], 1000)
plt.plot(xgrid, Q_true(xgrid), linewidth=2, color='black', label='Q_true')
plt.plot(xgrid, Q_star(xgrid), linewidth=4, color=[0,.3,1], label='Q_star')
plt.title(s)
plt.ylim([0, 2*max(Q_true(xgrid))])

plt.legend()
# Show plot
plt.show()