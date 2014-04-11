'''
fig3_draw.py 

Written by by Justin B. Kinney, Cold Spring Harbor Laboratory

Last updated on 15 December 2013 

Description:
    Draws Fig. 3 of Kinney, 2013. Takes less than 10 sec to execute on 
    my laptop computer.   

Dependencies:
    scipy
    numpy
    matplotlib
    time

Loads:
    dists.npy 
    
Saves:
    fig3.pdf   
            
Reference: 
    Kinney, J.B. (2013) Practical estimation of probability densities using scale-free field theories. arxiv preprint
'''

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import mquantiles
import time

start_time = time.clock()

# Close all open figures
plt.close('all')

# Set default plotting parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=8)
colwidth = 3.375

# Define colors for plotting
orange = [1, .5, 0]
lightblue = [0, .5, 1]	
lightgray = [0.7, 0.7, 0.7]
magenta = [.8, .2, .8]	
green = [.2, .8, .2]
panel_label_size = 14

plt.figure(figsize=[1*colwidth, 2.25])

# Load results file
dists = np.load('dists.npy')
    
# Plot
colors=[lightblue, magenta, green]

num_trials = dists.shape[0]
num_Ns = dists.shape[2]

# Plot 1D DEFT trials
hs = []
Ns = [100, 1000, 10000]
for j, a in enumerate([0, 1, 2, 4, 3]):
    for i in range(num_Ns):
        x0 = 0.2*(i-1)
        
        xs = x0+(j+1)*sp.ones([num_trials])#+ .3*(sp.random.rand(num_trials)-.5)
        ys = dists[:,a,i]

        # Plot mean of points
        quantiles = mquantiles(ys,[0.05, 0.25, 0.5, 0.75, 0.95])
        lolo_y = quantiles[0]
        lo_y = quantiles[1]
        mid_y = quantiles[2]
        hi_y = quantiles[3]
        hihi_y = quantiles[4]
        meanxs = sp.array([j+1-.15, j+1+.15])
        plt.semilogy(x0+meanxs, sp.array([mid_y, mid_y]), color=colors[i], linewidth=1, label='_nolegend_')
        plt.semilogy(x0+sp.array([j+1, j+1]), sp.array([lo_y, hi_y]), color=colors[i], linewidth=6, label='_nolegend_')
        plt.semilogy(x0+sp.array([j+1, j+1]), sp.array([lolo_y, hihi_y]), color=colors[i], linewidth=2, label='_nolegend_')
        if j == 0:
            h, = plt.semilogy(x0+sp.array([j+1, j+1]), sp.array([lolo_y, hihi_y]), color=colors[i], linewidth=2, label=r'$N = %d$'%Ns[i]) 
            hs.append(h)

#plt.legend(hs)

plt.legend(hs, ['$N=100$', '$N=1000$', '$N=10000$'], fontsize=7, loc=2, ncol=3, mode="expand", borderpad=.5, frameon=False)
#plt.gca().add_artist(leg)

plt.xticks([1,2,3,4,5], ['$\\alpha = 1$', '$\\alpha = 2$', '$\\alpha = 3$', 'KDE',  'GMM'])
plt.ylabel('$D(Q_{true},Q^*)$')
plt.ylim([1E-2, 6E-1])
plt.xlim([.5, 5.5])
#plt.title('$n = 1$')
       
    
plt.subplots_adjust(hspace=0.15, wspace=0.2, left=0.2, right=0.95, top=.85, bottom=.15)
plt.show()
plt.savefig('fig3.pdf')

print 'fig3_draw.py took %.2f seconds to execute'%(time.clock()-start_time) 

