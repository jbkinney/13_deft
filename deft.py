'''
deft.py (v1.0)

Written by by Justin B. Kinney, Cold Spring Harbor Laboratory

Last updated on 18 December 2013 

Description:
    Density Estimation using Field Theory (DEFT) in 1D and 2D

Reference: 
    Kinney, J.B. (2013) Practical estimation of probability densities using scale-free field theories. arxiv preprint

Functions:
    deft_1d: Performs density estimation in one dimension
    deft_2d: Performs density estimation in two dimensions

Dependencies:
    scipy
    numpy
    copy
    time
'''

# Import stuff
import scipy as sp
from scipy.integrate import odeint
from scipy.fftpack import fft, ifft, fft2, ifft2
from scipy.linalg import eig, det
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from numpy.random import choice, randn
from copy import deepcopy
import time

# Empty container class for storing results
class Results: pass;

################################################################################
# 1D DEFT
################################################################################
def deft_1d(xis_raw, bbox, G=100, alpha=2, num_samples=0, num_ts=100, ti_shift=-5, tf_shift=0, verbose=False, details=False):
    '''
    Performs DEFT density estimation in 1D
    
    Args:
        xis_raw: The data, comprising a list (or scipy array) of real numbers. 
            Data falling outside bbox is discarded.
            
        bbox: This specifies the domain (bounding box) on which density 
            estimation is performed. This should be of the form [xmin, xmax]. 
            Density estimation will be performed on G grid points evenly spaced 
            within this interval. Max and min grid points are placed half a grid
            spacing in from the boundaries xmin and xmax. 
            
        G: Number of grid points on which to compute the estimated density
        
        alpha: The smoothness parameter, specifying which derivative of the 
            filed is constrained by the prior. May be any integer >= 1.
            
        num_samples: The number of density estimates to sample from the Bayesian
            posterior. If zero is passed, no sample are drawn.
            
        num_ts: The number of lengthscales at which to compute Q_\ell. 
        
        ti_shift: Initial integration t is set to 
            t_i = log[ (2*pi)**(2*alpha) * L**(1-2*alpha)] +  ti_shift
            (see Eq. 10)
            
        tf_shift: Final integration t is set to
            t_f = log[ (2*pi*G)**(2*alpha) * L**(1-2*alpha)] +  tf_shift
            (see Eq. 10)
        
        verbose: If True, execution time is reportred to the user.
        
        details: If True, calculation details are returned along with the 
            density estimate Q_star_func 
            
    Returns:
        Q_star_func: A function, defined within bbox, providing a cubic spline
            interpolation of the maximum a posteriori density estimate.
            
        results: Returned only if `details' is set to be True. Contains detailed
            information about the density estimation calculation. 
    '''
    
    # Time execution
    start_time = time.clock()
    
    # Check data types
    assert G==sp.floor(G)
    assert len(bbox) == 2
    assert len(xis_raw) > 1
    assert num_samples >= 0 and num_samples==sp.floor(num_samples)
    assert 2*alpha-1 >= 1
    
    ###
    ### Draw grid and histogram data	
    ###
    	 	
    # Make sure xis is a numpy array
    xis_raw = sp.array(xis_raw)
           
    # Get upper and lower bounds on x
    xlb = bbox[0]
    xub = bbox[1]
    V = xub-xlb
    L = V
    assert(V > 0)
    indices = (xis_raw >= xlb) & (xis_raw <= xub)
    
    # Throw away data outside interval
    xis = xis_raw[indices]
        
    # Determine number of data points
    N = len(xis)
        
    # Compute edges for histogramming 
    xedges = sp.linspace(xlb,xub,G+1)
    dx = xedges[1] - xedges[0]
    
    # Compute grid for binning
    xgrid = xedges[:-1]+dx/2
    
    # Converte to z = normalized x
    zis = (xis-xlb)/dx
    zedges = (xedges-xlb)/dx
    
    # Histogram data
    [R, xxx] = sp.histogram(zis, zedges, normed=1)
    
    ###
    ### Use weak-field approximation to get phi_0	
    ###
    
    # Compute fourier transform of data
    R_ks = fft(R)
    
    # Set constant component of data FT to 0
    R_ks[0] = 0
    
    # Set mode numbers seperately for even L and for odd L
    L = G
    if L%2 == 0: # If L is even
        # ks = [0, 1, ... , L/2, -L/2+1, ..., -1] 
        ks = sp.concatenate((sp.arange(0, L/2 + 1), sp.arange(1 - L/2, 0)))
    else: # If L is odd
        # ks = [0, 1, ... , (L-1)/2, -(L-1)/2, ..., -1] 
        ks = sp.concatenate((sp.arange(0, (L-1)/2 + 1), sp.arange(-(L-1)/2, 0)))
    
    # Compute eigenvalues of Delta
    lambda_ks = (2.0*sp.pi*ks/L)**(2.0*alpha)
    
    # Just to keep the logrithm from complaining.
    # This number is actually irrelevant since R_ns[0] = 0
    lambda_ks[0] = 1  
    
    # Compute corresponding taus
    tau_ks = sp.log(L*lambda_ks)  
            
    # Set t_i to the \ell_i vale in Eq. 10 plus shift
    t_i = sp.log(((2*sp.pi)**(2*alpha))*G**(1.0-2.0*alpha)) + ti_shift
    
    # Set t_f to the \ell_f value in Eq. 10 plus shift 
    t_f = sp.log(((2*sp.pi*G)**(2*alpha))*G**(1.0-2.0*alpha)) + tf_shift

    # Compute Fourier components of phi    
    phi_ks = -R_ks*V/(1.0 + sp.exp(tau_ks - t_i))

    # Invert the Fourier transform to get phi weak field caculation of phi0
    phi0 = sp.real(ifft(phi_ks)) 
    
    # Let user know integration interval
    #if verbose:    
    #    print 'Integrating from t_i == %f to t_f == %f'%(t_i, t_f)
    
    # Set integrationt times
    ts = sp.linspace(t_i,t_f,num_ts)
    
    # Compute ells in terms of ts: exp(ts) = N/ell^(2 alpha - 1)
    # Note: this is in dx = 1 units!
    ells = (N/sp.exp(ts))**(1.0/(2.0*alpha-1.0))
    
    ###
    ### Integrate ODE
    ###
    
    # Initialize phis
    phis = sp.zeros([L,num_ts])
    
    # Create periodic laplacian matrix (not sparse)
    delsq = sp.eye(L,L,-1) + sp.eye(L,L,+1) - 2*sp.eye(L)        
    delsq[0,L-1] = 1.0
    delsq[L-1,0] = 1.0
    
    # Make Delta matrix, which is sparse
    Delta = csr_matrix(-delsq)**alpha
    
    # This is the key function: computes deriviate of phi for ODE integration
    def this_flow(phi, t):
        
        # Compute distribution Q corresponding to phi
        Q = sp.exp(-phi)/L
        
        # This matrix in invertible for all t > 0
        A = Delta*sp.exp(-t) + diags(Q, 0)
        
        # Solve A*phidt = Q-R
        dphidt = spsolve(A, Q - R) 
                    
        # Return the time derivative of phi
        return dphidt
        
    # Integrate phi over specified ts
    phis = odeint(this_flow, phi0, ts)
    
    ###
    ### Identify optimal lengthscale
    ###
    
    # Initialize containers
    log_evidence = sp.nan*sp.zeros(num_ts)
    S = sp.zeros(num_ts)
    log_det_Lambda = sp.zeros(num_ts)
    Qs = sp.zeros([num_ts, L])
    R_col = sp.mat(R).T
    coeff = 1.0
    
    # Compute log evidence for each t in ts
    for i in range(num_ts):
        
        # Get t corresonding to time ts[i]
        t = ts[i]
        
        # Get lengthscale correspondin to time ts[i]
        ell = ells[i]    
                
        # Get field at time ts[i]
        phi = phis[i,:]
        
        # Renormalize phi just in case
        Q = sp.exp(-phi)/sum(sp.exp(-phi))
        phi = -sp.log(L*Q)
        phi_col = sp.mat(phi).T
        Qs[i] = Q
        beta = ell**(2.0*alpha-1.0)
        
        # Compute the value of the action of classical path at ts[i] 
        if all(sp.isfinite(phi_col)):
            S[i] = 0.5*(phi_col.T*Delta*phi_col) + sp.exp(t)*(R_col.T*phi_col)[0,0] + sp.exp(t)
        else:
            S[i] = sp.inf
        
        # Compute exact fluctuating determinant 
        # Note: the Lambda matrix often needs to be rescaled to prevent
        # numerical prblems. This loop does that automatically
        Lambda = Delta + sp.exp(t)*sp.diag(Q)
        ok_value = False
        while not ok_value:
            f = sp.log(det(coeff*Lambda))
            if f == -sp.inf:
                coeff *= 1.5
                print '%d:+'%i
            elif f == sp.inf:
                coeff /= 1.5
                print '%d:-'%i
            else: 
                ok_value = True
        log_det_Lambda[i] = f - L*sp.log(coeff) 
        
        # If i = 0, i.e. largest value of \ell, compute the deteriminant ratio
        # in weak field approximation. Use to compute log_det_nc_Delta
        if i == 0:
            fluct_wf = t + sum(sp.log(1.0 + sp.exp(t - tau_ks)))
            log_det_nc_Delta = log_det_Lambda[i] - fluct_wf
            
        # Compute evidence for each ell with correct proportionality constant
        log_evidence[i] = (N + 0.5*sp.sqrt(N) - N*sp.log(G)) - beta*S[i] - 0.5*sp.log(beta) - 0.5*(log_det_Lambda[i] - log_det_nc_Delta)
    
    # Noramlize weights for different ells
    # Note: ells are logarithmically distributed
    # So Jeffery's prior is flat
    ell_weights = sp.exp(log_evidence - max(log_evidence))
    ell_weights[ell_weights < -100] = 0.0 
    assert(not sum(ell_weights) == 0)
    assert(all(sp.isfinite(ell_weights)))
    ell_weights /= sum(ell_weights)    
                
    # Find the best lengthscale
    i_star = sp.argmax(log_evidence)
    phi_star = phis[i_star,:]
    Q_star = sp.exp(-phi_star)/sum(sp.exp(-phi_star))
    ell_star = ells[i_star]
    t_star = ts[i_star]
    M_star = sp.exp(t_star)
    
    ###
    ### Sample from posterior (only if requirested)
    ###
    
    # If user requests samples
    if num_samples > 0:
    
        Lambda_star = (ell_star**(2.0*alpha-1.0))*(Delta + M_star*sp.diag(Q_star))    
        eigvals, eigvecs = eig(Lambda_star)
        
        # Lambda_star is Hermetian; shouldn't need to do this
        eigvals = sp.real(eigvals) 
        eigvecs = sp.real(eigvecs)                      
        
        # Initialize container variables      
        Qs_sampled = sp.zeros([num_samples, L])
        phis_sampled = sp.zeros([num_samples, L])
        is_sampled = sp.zeros([num_samples])
    
        for j in range(num_samples):
            
            # First choose a classical path based on 
            i = choice(num_ts, p=ell_weights)
            phi_cl = phis[i,:]
            is_sampled[j] = int(i)
        
            # Draw random amplitudes for all modes and compute dphi
            etas = randn(L)
            dphi = sp.ravel(sp.real(sp.mat(eigvecs)*sp.mat(etas/sp.sqrt(eigvals)).T))
    
            # Record final sampled phi 
            phi = phi_cl + dphi
            Qs_sampled[j,:] = sp.exp(-phi)/sum(sp.exp(-phi))
            
    ###
    ### Return results (with everything in correct lenght units!)
    ###
        
    results = Results()
    results.G = G
    results.ts = ts
    results.N = N
    results.alpha = alpha
    results.num_samples = num_samples
    results.xgrid = xgrid
    results.bbox = bbox
    
    # Store grid results
    results.Delta_grid = Delta
    results.Qs_grid = Qs
    results.log_det_Lambda_grid = log_det_Lambda
    results.S_grid = S
    
    # Everything with units of length gets multiplied by dx!!!
    results.ells = ells*dx
    results.L = L*dx
    results.V = V*dx**2
    results.h = dx
    results.Qs = Qs/dx
    results.phis = phis
    results.log_evidence = log_evidence - N*sp.log(dx) 
    
    # Comptue star results
    results.phi_star = deepcopy(phi_star)
    results.Q_star = deepcopy(Q_star)/dx
    results.i_star = i_star
    results.ell_star = ells[i_star]*dx
        
    # Create interpolated Q_star. Need to extend grid to boundaries first
    extended_xgrid = sp.zeros(L+2)
    extended_xgrid[1:-1] = xgrid
    extended_xgrid[0] = xlb
    extended_xgrid[-1] = xub
    results.extended_xgrid = extended_xgrid
    
    extended_Q_star = sp.zeros(L+2)
    extended_Q_star[1:-1] = Q_star
    end_Q_star = 0.5*(Q_star[0]+Q_star[-1])
    extended_Q_star[0] = end_Q_star
    extended_Q_star[-1] = end_Q_star
    results.extended_Q_star = extended_Q_star/dx
    
    Q_star_func = interp1d(extended_xgrid, extended_Q_star/dx, kind='cubic')
    
    # If samples are requested, store those too
    if num_samples > 0:
        results.phis_sampled = phis_sampled
        results.Qs_sampled = Qs_sampled/dx
        results.is_sampled = is_sampled
    
    # Stop time
    time_elapsed = time.clock() - start_time
    if verbose:
        print 'deft_1d: %1.2f sec for alpha = %d, G = %d, N = %d'%(time_elapsed, alpha, G, N)
    
    if details:
        return Q_star_func, results
    else:
        return Q_star_func

################################################################################
# 2D DEFT
################################################################################

def deft_2d(xis_raw, yis_raw, bbox, G=20, alpha=2, num_samples=0, num_ts=100, ti_shift=-5, tf_shift=0, verbose=False, details=False):
    '''
    Performs DEFT density estimation in 2D
    
    Args:
        xis_raw: The x-data, comprising a list (or scipy array) of real numbers. 
            Data falling outside bbox is discarded.
            
        yis_raw: The y-data, comprising a list (or scipy array) of real numbers. 
            Data falling outside bbox is discarded.  
              
        bbox: The domain (bounding box) on which density estimation is 
            performed. This should be of the form [xmin, xmax, ymin, ymax]. 
            Density estimation will be performed on G grid points evenly spaced 
            within this interval. Max and min grid points are placed half a grid
            spacing in from the boundaries.
            
        G: Number of grid points to use in each dimension. Total number of 
            gridpoints used in the calculation is G**2.
            
        alpha: The smoothness parameter, specifying which derivative of the 
            filed is constrained by the prior. May be any integer >= 2.
            
        num_samples: The number of density estimates to sample from the Bayesian
            posterior. If zero is passed, no sample are drawn.
            
        num_ts: The number of lengthscales at which to compute Q_\ell. 
        
        ti_shift: Initial integration t is set to 
            t_i = log[ (2*pi)**(2*alpha) * L**(2-2*alpha)] +  ti_shift
            (see Eq. 10)
            
        tf_shift: Final integration t is set to
            t_f = log[ (2*pi*G)**(2*alpha) * L**(2-2*alpha)] +  tf_shift
            (see Eq. 10)
        
        verbose: If True, user feedback is provided
        
        details: If True, calculation details are returned along with the 
            density estimate Q_star_func 
            
    Returns:
        Q_star_func: A function, defined within bbox, providing a cubic spline
            interpolation of the maximum a posteriori density estimate.
            
        results: Returned only if `details' is set to be True. Contains detailed
            information about the density estimation calculation. 
    '''
    # Time execution
    start_time = time.clock()
    
    # Check data types
    L = G
    V = G**2
    assert G==sp.floor(G)
    assert len(xis_raw) > 1
    assert num_samples >= 0 and num_samples==sp.floor(num_samples)
    assert 2*alpha-2 >= 1
    assert len(bbox) == 4
    
    ###
    ### Draw grid and histogram data	
    ###
    	 	
    # Make sure xis is a numpy array
    xis_raw = sp.array(xis_raw)
        
    # If xdomain are specified, check them, and keep only data that falls within

    xlb = bbox[0]
    xub = bbox[1]
    assert(xub-xlb > 0)
    
    ylb = bbox[2]
    yub = bbox[3]
    assert(yub-ylb > 0)
    
    # Throw away data not within bbox
    indices = (xis_raw >= xlb) & (xis_raw <= xub) & (yis_raw >= ylb) & (yis_raw <= yub)
    xis = xis_raw[indices]
    yis = yis_raw[indices]
        
    # Determine number of data points within bbox
    N = len(xis)
        
    # Compute edges for histogramming 
    xedges = sp.linspace(xlb,xub,L+1)
    dx = xedges[1] - xedges[0]
    
    yedges = sp.linspace(ylb,yub,L+1)
    dy = yedges[1] - yedges[0]
    
    # Compute grid for binning
    xgrid = xedges[:-1]+dx/2
    ygrid = yedges[:-1]+dy/2
    
    # Convert to z = normalized x
    xzis = (xis-xlb)/dx
    yzis = (yis-ylb)/dx
    xzedges = (xedges-xlb)/dx
    yzedges = (yedges-ylb)/dy
    
    # Histogram data
    [H, xxx, yyy] = sp.histogram2d(xzis, yzis, [xzedges, yzedges])
    R = H/N
    R_flat = R.flatten()
    
    ###
    ### Use weak-field approximation to get phi_0	
    ###
    
    # Compute fourier transform of data
    R_ks = fft2(R)
    
    # Set constant component of data FT to 0
    R_ks[0] = 0
    
    # Set mode numbers seperately for even G and for odd G
    if G%2 == 0: # If G is even
        # ks = [0, 1, ... , G/2, -G/2+1, ..., -1] 
        ks = sp.concatenate((sp.arange(0, G/2 + 1), sp.arange(1 - G/2, 0)))
    else: # If G is odd
        # ks = [0, 1, ... , (G-1)/2, -(G-1)/2, ..., -1] 
        ks = sp.concatenate((sp.arange(0, (G-1)/2 + 1), sp.arange(-(G-1)/2, 0)))
    
    # Mode numbers corresponding to fourier transform
    A = sp.mat(sp.tile((2.0*sp.pi*ks/G)**2.0,[G,1]))
    B = A.T
    tau_ks = sp.log(V*sp.array(A+B)**alpha)
    tau_ks[0,0] = 1 # This number is actually irrelevant since R_hat[0,0] = 0 
    
    # Set t_i to the \ell_i vale in Eq. 10 plus shift
    t_i = sp.log(((2*sp.pi)**(2*alpha))*G**(2.0-2.0*alpha)) + ti_shift
    
    # Set t_f to the \ell_f value in Eq. 10 plus shift 
    t_f = sp.log(((2*sp.pi*G)**(2*alpha))*G**(2.0-2.0*alpha)) + tf_shift

    # Compute Fourier components of phi    
    phi_ks = -(G**2)*sp.array(R_ks)/sp.array((1.0 + sp.exp(tau_ks - t_i)))

    # Invert the Fourier transform to get phi weak field approx
    phi0 = sp.ravel(sp.real(ifft2(phi_ks)))
    
    # Let user know integration interval
    # if verbose:    
    #    print 'Integrating from t_i == %f to t_f == %f'%(t_i, t_f)
    
    # Set integrationt times
    ts = sp.linspace(t_i,t_f,num_ts)
    
    # Compute ells in terms of ts: exp(ts) = N/ell^(2 alpha - 2)
    # Note: this is in dx = 1 units!
    ells = (N/sp.exp(ts))**(1.0/(2.0*alpha-2.0))
    
    ###
    ### Integrate ODE
    ###
    
    # Initialize phis
    phis = sp.zeros([V,num_ts])
    
    # Build 2D Laplacian matrix
    delsq_2d = (-4.0*sp.eye(V) + 
        sp.eye(V,V,-1) + sp.eye(V,V,+1) + 
        sp.eye(V,V,-G) + sp.eye(V,V,+G) +
        sp.eye(V,V,-V+1) + sp.eye(V,V,V-1) +
        sp.eye(V,V,-V+G) + sp.eye(V,V,V-G))
    
    # Make Delta, and make it sparse
    Delta = csr_matrix(-delsq_2d)**alpha
    
    # This is the key function: computes deriviate of phi for ODE integration
    def this_flow(phi, t):
        
        # Compute distribution Q corresponding to phi
        Q = sp.exp(-phi)/V
        
        # This matrix in invertible for all t > 0
        A = Delta*sp.exp(-t) + diags(Q, 0)
        
        # Solve A*phidt = Q-R
        dphidt = spsolve(A, Q - R_flat) 
                    
        # Return the time derivative of phi
        return dphidt
        
    # Integrate phi over specified ts    
    phis = odeint(this_flow, phi0, ts)
    
    ###
    ### Identify optimal lengthscale
    ###
    
    # Initialize containers
    log_evidence = sp.nan*sp.zeros(num_ts)
    S = sp.zeros(num_ts)
    log_det_Lambda = sp.zeros(num_ts)
    Qs = sp.zeros([num_ts, G, G])
    R_col = sp.mat(sp.ravel(R)).T
    coeff = 1.0
    
    # Compute log evidence for each t in ts
    for i in range(num_ts):
        
        # Get M corresonding to time ts[i]
        t = ts[i]
        
        # Get lengthscale correspondin to time ts[i]
        ell = ells[i]    
                
        # Get field at time ts[i]
        phi = phis[i,:]
        
        # Renormalize phi just in case
        Q = sp.exp(-phi)/sum(sp.exp(-phi))
        phi = -sp.log(V*Q); 
        phi_col = sp.mat(phi).T; 
        Qs[i,:,:] = sp.reshape(Q,[L,L])
        beta = ell**(2.0*alpha-2.0)
        
        # Compute the value of the action of classical path at ts[i] 
        if all(sp.isfinite(phi_col)):
            S[i] = 0.5*(phi_col.T*Delta*phi_col) + sp.exp(t)*(R_col.T*phi_col)[0,0] + sp.exp(t)
        else:
            S[i] = sp.inf
            
        # Compute exact fluctuating determinant 
        # This requires dynamically fiddling with the overall scale of lambda
        # to avoid infinities. Should find a more sensible way to do this
        Lambda = Delta + sp.exp(t)*sp.diag(Q)
        ok_value = False
        while not ok_value:
            f = sp.log(det(coeff*Lambda))
            if f == -sp.inf:
                coeff *= 1.5
                #print '%d:+'%i
            elif f == sp.inf:
                coeff /= 1.5
                #print '%d:-'%i
            else: 
                ok_value = True
        log_det_Lambda[i] = f  - V*sp.log(coeff)
        
        # If i = 0, i.e. largest value of \ell, compute the deteriminant ratio
        # in weak field approximation. Use to compute log_det_nc_Delta
        if i == 0:
            fluct_wf = t + sum(sum(sp.log(1.0 + sp.exp(t - tau_ks))))
            log_det_nc_Delta = log_det_Lambda[i] - fluct_wf
        
        # Compute evidence for each ell with correct proportionality constant
        log_evidence[i] = (N + 0.5*sp.sqrt(N) - N*sp.log(V)) - beta*S[i] - 0.5*sp.log(beta) - 0.5*(log_det_Lambda[i] - log_det_nc_Delta)
    
    # Noramlize weights for different ells and save
    ell_weights = sp.exp(log_evidence) - max(log_evidence)
    ell_weights[ell_weights < -100] = 0.0 
    assert(not sum(ell_weights) == 0)
    assert(all(sp.isfinite(ell_weights)))
    ell_weights /= sum(ell_weights)    
                
    # Find the best lengthscale
    i_star = sp.argmax(log_evidence)
    phi_star = phis[i_star,:]
    Q_star = sp.reshape(sp.exp(-phi_star)/sum(sp.exp(-phi_star)), [G, G])
    ell_star = ells[i_star]
    t_star = ts[i_star]
    M_star = sp.exp(t_star)
    
    ###
    ### Sample from posterior (only if requirested)
    ###
    
    # If user requests samples
    if num_samples > 0:
    
        Lambda_star = (ell_star**(2.0*alpha-120))*(Delta + M_star*sp.diag(Q_star))    
        eigvals, eigvecs = eig(Lambda_star)
        
        # Lambda_star is Hermetian; shouldn't need to do this
        eigvals = sp.real(eigvals) 
        eigvecs = sp.real(eigvecs)                      
        
        # Initialize container variables      
        Qs_sampled = sp.zeros([num_samples, G, G])
        phis_sampled = sp.zeros([num_samples, V])
        is_sampled = sp.zeros([num_samples])
    
        for j in range(num_samples):
            
            # First choose a classical path based on 
            i = choice(num_ts, p=ell_weights)
            phi_cl = phis[i,:]
            is_sampled[j] = i
        
            # Draw random amplitudes for all modes and compute dphi
            etas = randn(L)
            dphi = sp.ravel(sp.real(sp.mat(eigvecs)*sp.mat(etas/sp.sqrt(eigvals)).T))
    
            # Record final sampled phi 
            phi = phi_cl + dphi
            Qs_sampled[j,:,:] = sp.reshape(sp.exp(-phi)/sum(sp.exp(-phi)), [G,G])
            
    ###
    ### Return results (with everything in correct lenght units!)
    ###
        
    results = Results()
    results.G = G
    results.ts = ts
    results.N = N
    results.alpha = alpha
    results.num_samples = num_samples
    results.xgrid = xgrid
    results.bbox = bbox
    results.dx = dx
    results.dy = dy
    
    # Everything with units of length gets multiplied by dx!!!
    results.ells = [ells*dx, ells*dy]
    results.L = [G*dx, G*dy]
    results.V = V*dx*dy
    results.h = [dx, dy]
    results.Qs = Qs/(dx*dy)
    results.phis = phis
    results.log_evidence = log_evidence
    
    # Comptue star results
    results.phi_star = deepcopy(phi_star)
    results.Q_star = deepcopy(Q_star)/(dx*dy)
    results.i_star = i_star
    results.ell_star = [ells[i_star]*dx, ells[i_star]*dy]
    bbox = [xlb, xub, ylb, yub]
    
    # Create interpolated Q_star. Man this is a pain in the ass!
    extended_xgrid = sp.zeros(G+2)
    extended_xgrid[1:-1] = xgrid
    extended_xgrid[0] = xlb
    extended_xgrid[-1] = xub
    
    extended_ygrid = sp.zeros(L+2)
    extended_ygrid[1:-1] = ygrid
    extended_ygrid[0] = ylb
    extended_ygrid[-1] = yub
    
    extended_Q_star = sp.zeros([G+2, G+2])
    extended_Q_star[1:-1,1:-1] = Q_star
    
    # Get rows
    row = 0.5*(Q_star[0,:] + Q_star[-1,:])
    extended_Q_star[0,1:-1] = row
    extended_Q_star[-1,1:-1] = row
    
    # Get cols
    col = 0.5*(Q_star[:,0] + Q_star[:,-1])
    extended_Q_star[1:-1,0] = col
    extended_Q_star[1:-1,-1] = col
    
    # Get remaining corners, which share the same value
    corner= 0.25*(row[0]+row[-1]+col[0]+col[-1])
    extended_Q_star[0,0] = corner
    extended_Q_star[0,-1] = corner
    extended_Q_star[-1,0] = corner
    extended_Q_star[-1,-1] = corner

    # Finally, compute interpolated function
    Q_star_func = RectBivariateSpline(extended_xgrid, extended_ygrid, extended_Q_star/(dx*dy), bbox=bbox)    
                   
    # If samples are requested, return those too 
    if num_samples > 0:
        results.phis_sampled = phis_sampled
        results.Qs_sampled = Qs_sampled/(dx*dy)
        results.is_sampled = is_sampled
    
    # Stop time
    time_elapsed = time.clock() - start_time
    if verbose:
        print 'deft_2d: %1.2f sec for alpha = %d, G = %d, N = %d'%(time_elapsed, alpha, G, N)
    
    if details:
        return Q_star_func, results
    else:
        return Q_star_func