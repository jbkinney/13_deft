'''
fig4_calculate.py 

Written by by Justin B. Kinney, Cold Spring Harbor Laboratory

Last updated on 15 December 2013 

Description:
    Simulates data and performs density estimation reported in Fig. 4 of
    Kinney, 2013. Takes about 2 min to execute on my laptop computer.  

Dependencies:
    scipy
    sklearn
    matplotlib
    time
    deft
    kinney2013_utils

Loads:
    None.
    
Saves:
    things_2d.pk   
            
Reference: 
    Kinney, J.B. (2013) Practical estimation of probability densities using scale-free field theories. arxiv preprint
'''

import scipy as sp
from scipy.stats import gaussian_kde
from sklearn import mixture
import matplotlib.pyplot as plt
from matplotlib import cm
from kinney2013_utils import save_object
from deft import deft_2d
import time

def midpoints(grid):
    return grid[:-1] + 0.5*(grid[1] - grid[0])

start_time = time.clock()

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=9)
cmap = cm.gray

# Close all current figures
plt.close('all')
plt.figure(figsize=[ 11.625,   6.35 ])

# Simulation parameters
Ns = [30, 300, 3000]
num_Ns = len(Ns)
G = 35 # 20x20 grid

# Define grid
xmin = -5
xmax = 5
ymin = -5
ymax = 5
xedges = sp.linspace(xmin, xmax, G+1)
yedges = sp.linspace(ymin, ymax, G+1)
xgrid = midpoints(xedges)
ygrid = midpoints(yedges)
Xgrid, Ygrid = sp.meshgrid(xgrid, ygrid)
dx = xgrid[1]-xgrid[0]
dy = ygrid[1]-ygrid[0]
bbox = [xmin, xmax, ymin, ymax]

xfine = sp.linspace(xmin, xmax, 100)
yfine = sp.linspace(ymin, ymax, 100)

# Compute Q_true
R1 = (Xgrid+2.0)**2 + (Ygrid+2.0)**2
R2 = (Xgrid-1.0)**2/4 + (Ygrid-2.0)**2
Q_true = sp.exp(-R1/2) + sp.exp(-R2/2)
Q_true = Q_true.T/(dx*dy*sum(sum(Q_true)))
clim = [0, max(Q_true.flat[:])] 

def draw_from_Q_true(N, bbox):
    # Draw xs and ys from a normal distribution
    vis = sp.random.randn(2*N,2)
    
    # Create bimodal distribution
    ncut = int(sp.floor(2*N/3))
    xis = vis[:,0]
    yis = vis[:,1]
    yis[:ncut] -= 2.0
    yis[ncut:] += 2.0
    xis[:ncut] -= 2.0
    xis[ncut:] *= 2.0
    xis[ncut:] += 1.0
    
    # Shuffle xis and yis
    indices = sp.arange(len(vis))
    sp.random.shuffle(indices)
    xis = xis[indices]
    yis = yis[indices]
    
    # Select exactly N data points
    indices = (xis > bbox[0]) & (xis < bbox[1]) & (yis > bbox[2]) & (yis < bbox[3])
    xis = xis[indices]
    xis = xis[:N]
    yis = yis[indices]
    yis = yis[:N]
    
    return xis, yis

# Plot Q_true
plt.subplot(num_Ns,7,1)
plt.imshow(Q_true, interpolation='nearest', cmap=cmap)
plt.title('$Q_{true}$')
plt.xticks([])
plt.yticks([])
plt.clim(clim)

# Save everything
things = {}
things['Ns'] = Ns
things['G'] = G
things['xedges'] = xedges
things['yedges'] = yedges
things['Q_true'] = Q_true
things['Rs'] = []
things['Q_star2s'] = []
things['Q_star3s'] = []
things['Q_star4s'] = []
things['Q_gmms'] = []
things['Q_kdes'] = []
things['data'] = []

# Iterate through different Ns
for n,N in enumerate(Ns):

    # Draw data from distribution
    xis, yis = draw_from_Q_true(N, bbox)

    # Define grid
    [R, xxx, yyy] = sp.histogram2d(xis, yis, [xedges, yedges], normed='True')
    
    # Flatten H into R matrix
    things['Rs'].append(R)
    plt.subplot(num_Ns,7,2+7*n)
    plt.imshow(R, interpolation='nearest', cmap=cmap)
    plt.clim(clim)
    plt.axis('off')
    plt.title(r'R, N=%d'%N)
    
    # Do DEFT calculation (alpha = 2)
    Q_star_func2, results2 = deft_2d(xis, yis, bbox, G=G, alpha=2.0, details=True, verbose=True)
    Q_star2 = results2.Q_star
    things['Q_star2s'].append(Q_star2)
    plt.subplot(num_Ns,7,3+7*n)
    plt.imshow(Q_star_func2(xfine,yfine), interpolation='nearest', cmap=cmap)
    plt.clim(clim)
    plt.axis('off')
    plt.title(r'Q, $\alpha$=%d, N=%d'%(2,N))
    
    # Do DEFT caclulation (alpha = 3)
    Q_star_func3, results3 = deft_2d(xis, yis, bbox, G=G, alpha=3.0, details=True, verbose=True)
    Q_star3 = results3.Q_star
    things['Q_star3s'].append(Q_star3)
    plt.subplot(num_Ns,7,4+7*n)
    plt.imshow(Q_star_func3(xfine,yfine), interpolation='nearest', cmap=cmap)
    plt.clim(clim)
    plt.axis('off')
    plt.title(r'Q, $\alpha$=%d, N=%d'%(3,N))
    
    # Do DEFT caclulation (alpha = 4)
    Q_star_func4, results4 = deft_2d(xis, yis, bbox, G=G, alpha=4.0, details=True, verbose=True)
    Q_star4 = results4.Q_star
    things['Q_star4s'].append(Q_star4)
    plt.subplot(num_Ns,7,5+7*n)
    plt.imshow(Q_star_func4(xfine,yfine), interpolation='nearest', cmap=cmap)
    plt.clim(clim)
    plt.axis('off')
    plt.title(r'Q, $\alpha$=%d, N=%d'%(4,N))
    
    # Compute KDE density estimate
    Vs = sp.zeros([G**2,2])
    Vs[:,0] = Xgrid.flat
    Vs[:,1] = Ygrid.flat
    vis = sp.zeros([N,2])
    vis[:,0] = xis
    vis[:,1] = yis
    kde = gaussian_kde(vis.T)
    Q_kde = sp.reshape(kde(Vs.T), [G, G]).T
    things['Q_kdes'].append(Q_kde)
    plt.subplot(num_Ns,7,6+7*n)
    plt.imshow(Q_kde, interpolation='nearest', cmap=cmap)
    plt.clim(clim)
    plt.axis('off')
    plt.title(r'KDE, N=%d'%N)
    
    # Compute GMM density estimate using BIC
    max_K = 10
    bic_values = sp.zeros([max_K]);
    Qs_gmm = sp.zeros([max_K,G**2])
    for k in sp.arange(1,max_K+1):
        gmm = mixture.GMM(int(k))
        gmm.fit(vis)
        Qgmm = lambda(x): sp.exp(gmm.score(x))
        Qs_gmm[k-1,:] = Qgmm(Vs)#/sum(Qgmm(Vs))
        bic_values[k-1] = gmm.bic(vis) 
    
    # Choose distribution with lowest BIC
    i_best = sp.argmin(bic_values)
    Q_gmm = sp.reshape(Qs_gmm[i_best,:], [G,G]).T
    things['Q_gmms'].append(Q_gmm)
    
    plt.subplot(num_Ns,7,7+7*n)
    plt.imshow(Q_gmm, interpolation='nearest', cmap=cmap)
    plt.clim(clim)
    plt.axis('off')
    plt.title('GMM, k=%d, N=%d'%(i_best,N))
    
    # Save data for later if needed
    things['data'].append(vis)

# Save everything
#save_object(things, 'things_2d.pk')    
plt.show()

print 'fig4_calculate.py took %.2f seconds to execute'%(time.clock()-start_time) 

