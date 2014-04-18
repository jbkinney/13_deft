'''
fig1_fig2_draw.py 

Written by by Justin B. Kinney, Cold Spring Harbor Laboratory

Last updated on 16 April 2014 

Description:
    Plots Figs. 1 and 2 of Kinney, 2013. Takes about 15 sec to execute on my
    laptop computer 

Dependencies:
    scipy
    time
    pylab
    matplotlib
    kinney2013_utils

Loads:
    things_1d.pk 
    
Saves:
   fig1.pdf
   fig2.pdf 
'''

# Makes a simple figure demonstrating the inference proces. 
import scipy as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pylab import cm
from kinney2013_utils import load_object, get_dist_func_from_details
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
panel_label_x = -0.15
panel_label_y = 1.3

###
### Load results
###

# Load results 
things = load_object('things_1d.pk')
Q_star = get_dist_func_from_details(things['Q_star_details'])
Q_true = get_dist_func_from_details(things['Q_true_details'])
Q_star_wide3 = get_dist_func_from_details(things['Q_star_wide3_details'])
Q_star_wide10 = get_dist_func_from_details(things['Q_star_wide10_details'])
Q_star_fine = get_dist_func_from_details(things['Q_star_fine_details'])
Q_star_coarse = get_dist_func_from_details(things['Q_star_coarse_details'])
Q_star_alpha1 = get_dist_func_from_details(things['Q_star_alpha1_details'], 'cubic')
Q_star_alpha3 = get_dist_func_from_details(things['Q_star_alpha3_details'])
xs = things['xs']
xis = things['xis']
xint = things['xint']
xspan = things['xspan']
xmid = things['xmid']
N = things['N']
G = things['G']
gaussians = things['gaussians']

Q_star_details = things['Q_star_details']
Q_true_details = things['Q_true_details']
yint = [0, 1.5*max(Q_true_details.extended_Q_star)]

###
### Figure 1
###

plt.figure(figsize=[1*colwidth, 5])

#
# Panel A: Histogram
#
ax = plt.subplot2grid((4,1),(0, 0))
ax.text(panel_label_x, panel_label_y, '(a)', transform=ax.transAxes, family='sans-serif',
      fontsize=panel_label_size, fontweight='bold', va='top', ha='right')

# Histogram data
n, bins, patches = plt.hist(xis, G, normed=1, histtype='stepfilled') 
plt.setp(patches, 'facecolor', lightgray, 'edgecolor', 'none')

# Plot true distribution
plt.plot(xs, Q_true(xs), '-', color='k', linewidth=1)
  
plt.yticks([])    
#plt.xticks([])
plt.ylim(yint)         
plt.xlim(xint)
plt.xlabel('$x$')
plt.ylabel('$Q$')
ax.xaxis.set_label_coords(.5, -0.25)

#
# Panel B: Field as a function of ell
#
ax = plt.subplot2grid((4,1),(1, 0))
ax.text(panel_label_x, panel_label_y,  '(b)', transform=ax.transAxes, 
      fontsize=panel_label_size, fontweight='bold', va='top', ha='right')

# THIS IS WHAT NEEDS TO BE REPLACED. NEED TO INTERPOLATE Qs at selected
# set of ells. 
#Qs_image = Q_star_details.Qs

# Get ell range
log10_ells_raw = sp.log10(Q_star_details.ells)[::-1]
log10_ell_i = max(log10_ells_raw)
log10_ell_f = min(log10_ells_raw)

# Get x range
G = Q_star_details.Qs.shape[1]
xmin = -5.0
xmax = 5.0

# Set extent for plotting
extent = [xmin, xmax, log10_ell_f, log10_ell_i]

# Interploate Qs at 300 different ells
K = 300
Qs_raw = Q_star_details.Qs[::-1,:]
log10_ells_grid = sp.linspace(log10_ell_f, log10_ell_i, K)
Qs_interp_func = interp1d(log10_ells_raw, Qs_raw, axis=0, kind='cubic')
Qs_image = Qs_interp_func(log10_ells_grid)[::-1]

plt.imshow(Qs_image, aspect='auto', cmap=cm.gray, interpolation='nearest', extent=extent)
plt.clim([0, max(Q_star_details.Q_star)])
plt.yticks([-1, 0, 1], ['$10^{-1}$', '$10^{0\ }$', '$10^{1\ }$'])
[t.set_color('w') for t in ax.yaxis.get_ticklines()]
[t.set_color('w') for t in ax.xaxis.get_ticklines()]

num_ells = len(Q_star_details.ells)
plt.xlim([xmin, xmax])
plt.xlabel('$x$')
plt.ylabel('$\ell$')
ax.xaxis.set_label_coords(.5, -0.25)

#
# Panel C: evidence as a function of ell
#
ax = plt.subplot2grid((4,1),(2, 0))
ax.text(panel_label_x, panel_label_y,  '(c)', transform=ax.transAxes,
      fontsize=panel_label_size, fontweight='bold', va='top', ha='right')
L = xspan[1]-xspan[0]
ys = Q_star_details.log_evidence
ys = ys - max(ys)
uniform = - N*sp.log(L)
xl = [min(Q_star_details.ells), max(Q_star_details.ells)]
#plt.semilogx(xl, [uniform, uniform], '--r')
plt.semilogx(Q_star_details.ells, ys, color='k', linewidth=1)
plt.xlim(xl)
plt.xlabel('$\ell$')
ax.xaxis.set_label_coords(.5, -0.25)
plt.ylabel('$\ln\ p(\ell|$data$)$\t')
yspan = max(ys) - uniform
ymax = max(ys) + 0.05*yspan #ymax = uniform + 1.2*yspan
ymin = max(ys) - 100 #min(ys) - 0.05*yspan #ymin = uniform - 1.2*yspan

# Compute limits on ell
num_samples = Q_star_details.num_samples
for k in range(num_samples):
    ell = Q_star_details.ells_sampled[k]
    plt.semilogx(ell*sp.array([1, 1]), [ymin, ymax], color=orange, linewidth=1, alpha=0.3)

plt.semilogx(Q_star_details.ell_star*sp.array([1, 1]), [ymin, ymax], color=lightblue, linewidth=2)
plt.ylim([ymin, ymax])

#
# Panel D: MAP and posterior sampling
#
ax = plt.subplot2grid((4,1),(3, 0))
ax.text(panel_label_x, panel_label_y,  '(d)', transform=ax.transAxes,
      fontsize=panel_label_size, fontweight='bold', va='top', ha='right')

# Histogram data
n, bins, patches = plt.hist(xis, G, normed=1, histtype='stepfilled') 
plt.setp(patches, 'facecolor', lightgray, 'edgecolor', 'none')

# Plot true distribution
plt.plot(xs, Q_true(xs), '-', color='k', linewidth=1)

# Plot samples drawn from posterior
#plt.plot(Q_star_details.xgrid, Q_star_details.Qs_sampled.T, '-', color=orange, linewidth=1, alpha=0.3)
num_samples = Q_star_details.num_samples
extended_xgrid = Q_star_details.extended_xgrid
for k in range(num_samples):
    
    # This is very annoying. Have to extend Q on either side by 1 grid point 
    # to do interpolation
    Q = Q_star_details.Qs_sampled[k,:]
    extended_Q = sp.zeros(G+2)
    extended_Q[1:-1] = Q
    end_Q = 0.5*(Q[0]+Q[-1])
    extended_Q[0] = end_Q
    extended_Q[-1] = end_Q
    Q_sampled_func = interp1d(extended_xgrid, extended_Q, kind='cubic')

    # Plot sampled Q
    plt.plot(xs, Q_sampled_func(xs), '-', color=orange, linewidth=1, alpha=0.3)

# Plot Q_star distribution
plt.plot(xs, Q_star(xs), '-', color=lightblue, linewidth=2)

plt.yticks([])    
#plt.xticks([])
plt.ylim(yint)         
plt.xlim(xint)
plt.xlabel('$x$')
plt.ylabel('$Q$')
ax.xaxis.set_label_coords(.5, -0.25)

# Show Figure 1
plt.subplots_adjust(hspace=0.6, wspace=0.5, left=0.2, right=0.95)
plt.show()
plt.savefig('fig1.pdf')

###
### Figure 2
###

plt.figure(figsize=[1*colwidth, 3.6])
panel_label_y = 1.0
#
# Panel A
#

ax = plt.subplot2grid((3,1),(0, 0))
ax.text(panel_label_x, panel_label_y,  '(a)', transform=ax.transAxes,
      fontsize=panel_label_size, fontweight='bold', va='top', ha='right')

# Histogram data and plot in gray
n, bins, patches = plt.hist(xis, G, normed=1, histtype='stepfilled') 
plt.setp(patches, 'facecolor', lightgray, 'edgecolor', 'none') 

# Draw MAP distribution
plt.plot(xs, Q_true(xs), linewidth=1, color='k')
plt.plot(xs, Q_star(xs), linewidth=2, label='$L = 10$', color=lightblue)
plt.plot(xs, Q_star_wide3(xs), linewidth=2, label='$L = 30$', color=magenta)
plt.plot(xs, Q_star_wide10(xs), linewidth=2, label='$L = 100$', color=green)
ax.text(.5,0.80,r'$\alpha=2$, $h = 0.1$',
        horizontalalignment='center',
        transform=ax.transAxes)

plt.legend(fontsize=8, frameon=False)
plt.yticks([])    
plt.xlim(xint)
plt.ylim(yint) 
plt.xlabel('$x$')
plt.ylabel('$Q$')
ax.xaxis.set_label_coords(.5, -0.25)

#
# Panel B
#

ax = plt.subplot2grid((3,1),(1, 0))
ax.text(panel_label_x, panel_label_y,  '(b)', transform=ax.transAxes,
      fontsize=panel_label_size, fontweight='bold', va='top', ha='right')

# Histogram data and plot in gray
n, bins, patches = plt.hist(xis, G, normed=1, histtype='stepfilled') 
plt.setp(patches, 'facecolor', lightgray, 'edgecolor', 'none') 

# Draw MAP distribution
plt.plot(xs, Q_true(xs), linewidth=1, color='k')
plt.plot(xs, Q_star_coarse(xs), linewidth=2, label='$h = 0.3$', color=magenta)
plt.plot(xs, Q_star(xs), linewidth=2, label='$h = 0.1$', color=lightblue)
plt.plot(xs, Q_star_fine(xs), linewidth=2, label='$h = 0.03$', color=green)
ax.text(.5,0.80,r'$\alpha=2$, $L = 10$',
        horizontalalignment='center',
        transform=ax.transAxes)

plt.legend(fontsize=8, frameon=False)
plt.yticks([])    
plt.xlim(xint)
plt.ylim(yint) 
plt.xlabel('$x$')
plt.ylabel('$Q$')
ax.xaxis.set_label_coords(.5, -0.25)

#
# Panel C
#

ax = plt.subplot2grid((3,1),(2, 0))
ax.text(panel_label_x, panel_label_y,  '(c)', transform=ax.transAxes,
      fontsize=panel_label_size, fontweight='bold', va='top', ha='right')

# Histogram data and plot in gray
n, bins, patches = plt.hist(xis, G, normed=1, histtype='stepfilled') 
plt.setp(patches, 'facecolor', lightgray, 'edgecolor', 'none')

plt.plot(xs, Q_true(xs), linewidth=1, color='k')
plt.plot(xs, Q_star_alpha1(xs), linewidth=2, label=r'$\alpha=1$', color=magenta)
plt.plot(xs, Q_star(xs), linewidth=2, label=r'$\alpha=2$', color=lightblue)
plt.plot(xs, Q_star_alpha3(xs), linewidth=2, label=r'$\alpha=3$', color=green)
ax.text(.5,0.80,r'$L = 10$, $h = 0.1$',
        horizontalalignment='center',
        transform=ax.transAxes)

plt.legend(fontsize=8, frameon=False)
plt.yticks([])    
plt.xlim(xint)
plt.ylim(yint) 
plt.xlabel('$x$')
plt.ylabel('$Q$')
ax.xaxis.set_label_coords(.5, -0.25)

# Show Figure 2
plt.subplots_adjust(hspace=0.6, wspace=0.5, left=0.2, right=0.95)
plt.show()
plt.savefig('fig2.pdf')
 
print 'fig1_fig2_plot.py took %.2f seconds to execute'%(time.clock()-start_time) 
