'''
fig4_draw.py 

Written by by Justin B. Kinney, Cold Spring Harbor Laboratory

Last updated on 15 December 2013 

Description:
    Draws Fig. 4 of Kinney, 2013. Takes less than 10 seconds execute on my laptop computer.  

Dependencies:
    scipy
    matplotlib
    time
    deft
    kinney2013_utils

Loads:
    things_2d.pk 
    
Saves:
    fig4.pdf  
            
Reference: 
    Kinney, J.B. (2013) Practical estimation of probability densities using scale-free field theories. arxiv preprint
'''

import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from kinney2013_utils import load_object
from scipy.interpolate import RectBivariateSpline
import time

start_time = time.clock()

# Set plotting prameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=8)
cmap = cm.gray

# Close all current figures
plt.close('all')
colwidth = 3.375
plt.figure(figsize=[1*colwidth, 2.75])


things = load_object('things_2d.pk')
Ns = things['Ns']
G = things['G']
xedges = things['xedges']
yedges = things['yedges']
Q_true = things['Q_true']
Rs = things['Rs']
Q_star2s = things['Q_star2s']
Q_star3s = things['Q_star3s']
Q_star4s = things['Q_star4s']
Q_gmms = things['Q_gmms']
Q_kdes = things['Q_kdes']

dx = xedges[1]-xedges[0]
dy = yedges[1]-yedges[0]
xcenters = xedges[:-1]+dx/2
ycenters = yedges[:-1]+dy/2
bbox = [min(xedges), max(xedges), min(yedges), max(yedges)]

M = 100
xs = sp.linspace(min(xedges), max(xedges), M)
ys = sp.linspace(min(yedges), max(yedges), M)

# Simulation parameters
cols = 18
rows = 13
V = G**2
panel_label_size = 14
clim = [0, max(Q_true.flat[:])] 

# Plot Q_true
ax = plt.subplot2grid((rows,cols),(0, 0),rowspan=3,colspan=3)
Q_func = RectBivariateSpline(xcenters, ycenters, Q_true, bbox=bbox)
plt.imshow(Q_func(xs, ys), interpolation='nearest', cmap=cmap)
plt.title('$Q_\mathrm{true}$', fontsize=8)
plt.xticks([])
plt.yticks([])
plt.clim(clim)

# Iterate through different Ns
for n,N in enumerate(Ns):
    
    # Show 2d histogram of points
    ax = plt.subplot2grid((rows,cols),(4+3*n, 0),rowspan=3,colspan=3)
    plt.imshow(Rs[n], interpolation='nearest',cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('$N=%d$'%N, fontsize=8)
    plt.clim(clim)
    if n==0:
        plt.title('$R$', fontsize=8)
    
    # Plot DEFT estimate of Q, alpha=2
    ax = plt.subplot2grid((rows,cols),(4+3*n, 3),rowspan=3,colspan=3)
    Q_func = RectBivariateSpline(xcenters, ycenters, Q_star2s[n], bbox=bbox)
    plt.imshow(Q_func(xs, ys), interpolation='nearest', cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.clim(clim)
    if n==0:
        plt.title('$\\alpha = 2$', fontsize=8)
        
    # Plot DEFT estimate of Q, alpha=3
    ax = plt.subplot2grid((rows,cols),(4+3*n, 6),rowspan=3,colspan=3)
    Q_func = RectBivariateSpline(xcenters, ycenters, Q_star3s[n], bbox=bbox)
    plt.imshow(Q_func(xs, ys), interpolation='nearest', cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.clim(clim)
    if n==0:
        plt.title('$\\alpha = 3$', fontsize=8)
        
    # Plot DEFT estimate of Q
    ax = plt.subplot2grid((rows,cols),(4+3*n, 9),rowspan=3,colspan=3)
    Q_func = RectBivariateSpline(xcenters, ycenters, Q_star4s[n], bbox=bbox)
    plt.imshow(Q_func(xs, ys), interpolation='nearest', cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.clim(clim)
    if n==0:
        plt.title('$\\alpha = 4$', fontsize=8)
    
    # Plot KDE estimate
    ax = plt.subplot2grid((rows,cols),(4+3*n, 12),rowspan=3,colspan=3)
    Q_func = RectBivariateSpline(xcenters, ycenters, Q_kdes[n], bbox=bbox)
    plt.imshow(Q_func(xs, ys), interpolation='nearest', cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.clim(clim)
    if n==0:
        plt.title('KDE', fontsize=8)
    
    # Plot GMM estimate
    ax = plt.subplot2grid((rows,cols),(4+3*n, 15),rowspan=3,colspan=3)
    Q_func = RectBivariateSpline(xcenters, ycenters, Q_gmms[n], bbox=bbox)
    plt.imshow(Q_func(xs, ys), interpolation='nearest', cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.clim(clim)
    if n==0:
        plt.title('GMM', fontsize=8)
        

# Show plots. Annoying that I have to do this at all.
plt.subplots_adjust(hspace=0.2, wspace=0.2, left=0.1, right=0.95)
plt.show()
plt.savefig('fig4.pdf')

print 'fig4_draw.py took %.2f seconds to execute'%(time.clock()-start_time) 