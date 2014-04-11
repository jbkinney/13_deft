'''
kinney2013_utils.py 

Written by by Justin B. Kinney, Cold Spring Harbor Laboratory

Last updated on 15 December 2013 

Description:
    Defines various functions useful in the computations of Kinney, 2013. 

Dependencies:
    cPickle
    scipy

Loads:
    Nothing
    
Saves:
    Nothing
            
Reference: 
    Kinney, J.B. (2013) Practical estimation of probability densities using scale-free field theories. arxiv preprint
'''

import cPickle as pickle
import scipy as sp
from scipy.stats import beta
from scipy.interpolate import interp1d

class Details: pass;

class Gaussian: pass;

def get_dist_func(xgrid, Q_on_xgrid, kind='cubic'):
    phi_func = interp1d(xgrid, -sp.log(Q_on_xgrid), kind=kind)
    return lambda x: sp.exp(-phi_func(x))
    
def get_dist_func_from_details(details, kind='cubic'):
    xgrid = details.extended_xgrid
    Q_on_xgrid = details.extended_Q_star
    return get_dist_func(xgrid, Q_on_xgrid, kind=kind)

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    input = open(filename, 'rb')
    return pickle.load(input)

# Extrapolates out from ends of fit function
# Posted by sastanin on Stackoverflow on Apr 30 '10 at 15:07
def extrap1d(interpolator):
    xs = interpolator.x
    #ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return sp.Inf
            #return ys[0]
            #return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return sp.Inf
            #return ys[-1]
            #return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return sp.array(map(pointwise, sp.array(xs)))

    return ufunclike

def geo_dist(Q1, Q2, dx):
	return sp.arccos(sum(dx*sp.sqrt(Q1*Q2)))
#
def l2_dist(Q1, Q2, dx):
	return sum(dx*(Q1 - Q2)**2)

def draw_from_gaussian_mix(N, Nx, gaussians, xint=[]):
        weights = sp.array([g.weight for g in gaussians])
        mus = sp.array([g.mu for g in gaussians])
        sigmas = sp.array([g.sigma for g in gaussians])
        Ns = sp.array([int(sp.ceil(g.weight*N/sum(weights))) for g in gaussians])
        K = len(gaussians)
   	
   	# Draw >= N data points from mixture model
   	xis = [];
   	for k in range(K):
  	    s = mus[k] + sigmas[k]*sp.random.randn(Ns[k])
  	    xis.extend(s)
  		
   	# Shuffle xis and keep only N samples
   	sp.random.shuffle(xis)
   	xis = xis[:N]
    
   	# Determine x interval
   	xint_tight = max(xis) - min(xis)
   	xmin = min(xis) - 0.2*xint_tight
   	xmax = max(xis) + 0.2*xint_tight
   	xspan = xmax-xmin
   	xmid = 0.5*(xmax+xmin)
   	
   	# If xint is specified, move all data so that it falls within this interval
  	if len(xint) == 2:
  	    new_xint = xint
            new_xmid = sp.mean(new_xint)
            new_xspan = new_xint[1]-new_xint[0]
            assert(new_xspan > 0)
            
            # Shift and rescale xis and mus
            xis = (new_xspan/xspan)*(xis - xmid) + new_xmid
            mus = (new_xspan/xspan)*(mus - xmid) + new_xmid
            
            # Rescale sigmas
            sigmas = (new_xspan/xspan)*sigmas
   	
   	    # Now set rest of stuff
   	    xmin = new_xint[0]
   	    xmax = new_xint[1]
   	    xspan = new_xspan
   	    xmid = new_xmid
   	    
   	    # Make new gaussians
   	    gaussians = make_gaussian_mix(K=K, mus=mus, sigmas=sigmas, weights=weights)
   	
   	# Create a grid of points
   	dx = (xmax - xmin)/(Nx-1)
   	xgrid = sp.linspace(xmin, xmax, Nx)
   	xint = sp.array([xmin, xmax])
    
   	# Compute true distribution at gridpoints
   	R = sp.zeros(Nx);
   	for k in range(K):
  	    R = R + (weights[k]/sp.sqrt(2*sp.pi*sigmas[k]**2))*sp.exp(-(xgrid-mus[k])**2/(2*sigmas[k]**2))
  	Q = R/sum(dx*R)  	
  	
  	details = Details()
  	details.extended_Q_star = Q
  	details.xis = xis
  	details.extended_xgrid = xgrid
  	details.xint = xint
  	details.gaussians = gaussians
  	
  	# Compute cubic spline interpolation of true distribution field
  	phi_true_func = interp1d(xgrid, -sp.log(Q), kind='cubic')
  	Q_true_func = lambda x: sp.exp(-phi_true_func(x))
  	
  	# Return
  	#return [xis, xgrid, Q]
  	return [xis, xint, Q_true_func, details]
        
    
def make_gaussian_mix(K=0, mus=[], sigmas=[], weights=[]):
        # Set fixed parameters of mixture model
   	scale = 1
   	
   	
   	# Set random parameters of mixture model
        if K > 0 and len(mus) == 0 and len(sigmas) == 0 and len(weights) == 0:
            mus = scale*(20*sp.random.rand(K))
            sigmas = scale*sp.exp(2.0*sp.random.rand(K))
            weights = sp.random.rand(K)
            
        else:
            K = len(mus)
            assert(K == len(sigmas))
            assert(K == len(weights))
   	
   	# Fill in parameters of gaussians
   	gaussians = []
   	for k in range(K):
   	    g = Gaussian()
   	    g.mu = mus[k]
   	    g.sigma = sigmas[k]
   	    g.weight = weights[k]
   	    gaussians.append(g)    
        
        return gaussians
  	
def flat_dist(N, Nx):
    N = int(N)
    xis = sp.random.rand(N)
    xmin = 0.0
    xmax = 1.0
    
    # Create a grid of points
    xgrid = sp.linspace(xmin, xmax, Nx)
    dx = xgrid[1]-xgrid[0]
    
    Qtrue = 1.0*(xgrid <= 1)*(xgrid >= 0)
    Qtrue = Qtrue/sum(Qtrue*dx)
    
    # Return
    return [xis, xgrid, Qtrue]

def beta_dist(N, Nx, a, b):
    N = int(N)
    xis = beta.rvs(a, b, size=N)
    
    xmin = 0.0
    xmax = 1.0
    
    # Create a grid of points and do initial evaluate of Qtrue
    xgrid = sp.linspace(xmin, xmax, Nx)
    Qtrue = beta.pdf(xgrid, a, b)
    
    # Remove indices where Qtrue blows up
    indices = (Qtrue > -sp.inf)*(Qtrue < sp.inf)
    Qtrue = Qtrue[indices]
    xgrid = xgrid[indices]
    
    # Return
    return [xis, xgrid, Qtrue]