'''
deft.py (v1.1)

Written by by Justin B. Kinney, Cold Spring Harbor Laboratory

Last updated on 6 August 2014 

Description:
    Density Estimation using Field Theory (DEFT) in 1D and 2D

Reference: 
    Kinney JB (2014) Estimation of probability densities using scale-free field theories. 
        Phys Rev E 90:011301(R). arXiv:1312.6661 [physics.data-an].

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
from scipy.integrate import ode
from scipy.fftpack import fft, ifft, fft2, ifft2
from scipy.linalg import eig, det
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from numpy.random import choice, randn
from copy import deepcopy
import time
import warnings

# Empty container class for storing results
class Results: pass;

# Used to compute the log determinant of Lambda
def get_log_det_Lambda(Lambda, coeff):
    G = Lambda.shape[0]
    ok_value = False
    while not ok_value:
        f = sp.log(det(coeff*Lambda))
        if f == -sp.inf:
            coeff *= 1.5
            #print '+'
        elif f == sp.inf:
            coeff /= 1.5
            #print '-'
        else: 
            ok_value = True
    log_det = f - G*sp.log(coeff) 
    return log_det, coeff

################################################################################
# 1D DEFT
################################################################################
def deft_1d(xis_raw, bbox, G=100, alpha=2, num_samples=0, ti_shift=-5, 
    tf_shift=0, tol=1E-3, verbose=False, details=False, spline_type='cubic'):
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
            
        spline_type: Type of spline with which to interpolate phi_star and thus
            compute Q_star_func. Can be 'linear', 'nearest', 'zero', 'slinear', 
            'quadratic', or 'cubic'; is passed to scipy.interpolate.interp1d.
            
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
    R_col = sp.mat(R).T
    
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
    #ts = sp.linspace(t_i,t_f,num_ts)
    
    # Compute ells in terms of ts: exp(ts) = N/ell^(2 alpha - 1)
    # Note: this is in dx = 1 units!
    #ells = (N/sp.exp(ts))**(1.0/(2.0*alpha-1.0))
    
    ###
    ### Integrate ODE
    ###
    
    # Create periodic laplacian matrix (not sparse)
    delsq = sp.eye(L,L,-1) + sp.eye(L,L,+1) - 2*sp.eye(L)        
    delsq[0,L-1] = 1.0
    delsq[L-1,0] = 1.0
    
    # Make Delta matrix, which is sparse
    Delta = csr_matrix(-delsq)**alpha
    
    # This is the key function: computes deriviate of phi for ODE integration
    def this_flow(t, phi):
        
        # Compute distribution Q corresponding to phi     
        Q = sp.exp(-phi)/G
        A = Delta + sp.exp(t)*diags(Q,0)
        dphidt = sp.real(spsolve(A, sp.exp(t)*(Q-R)))     
                         
        # Return the time derivative of phi
        return dphidt
   
    backend = 'vode'
    solver = ode(this_flow).set_integrator(backend, nsteps=1, atol=tol, rtol=tol)
    solver.set_initial_value(phi0, t_i)
    # suppress Fortran-printed warning
    solver._integrator.iwork[2] = -1
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Make containers        
    phis = []
    ts = []
    log_evidence = []
    Qs = []
    ells = []
    log_dets = []  
    
    # Will keep initial phi to check that integration is working well
    phi0_col = sp.mat(phi0).T
    kinetic0 = (phi0_col.T*Delta*phi0_col)[0,0]
          
    # Integrate phi over specified ts
    #phis = odeint(this_flow, phi0, ts)
    
    # Integrate phi over specified t range. 
    integration_start_time = time.clock()
    keep_going = True
    max_log_evidence = -sp.Inf
    coeff = 1.0
    while solver.t < t_f and keep_going:
        
        # Step integrator 
        solver.integrate(t_f, step=True)
        
        # Compute deteriminant.
        phi = solver.y
        t = solver.t
        beta = N*sp.exp(-t)
        ell = (N/sp.exp(t))**(1.0/(2.0*alpha-1.0))
        
        # Compute new distribution
        Q = sp.exp(-phi)/sum(sp.exp(-phi)) 
        phi_col = sp.mat(phi).T
        
        # Check that S[phi] < S[phi0]. If not, phi might just be fucked up, due to the 
        # integration having to solve a degenerate system of equations. 
        # In this case, set phi = phi0 and restart integration from there. 
        S = 0.5*(phi_col.T*Delta*phi_col)[0,0] + sp.exp(t)*(R_col.T*phi_col)[0,0] + sp.exp(t)*sum(sp.exp(-phi)/G)
        S0 = 0.5*(phi0_col.T*Delta*phi0_col)[0,0] + sp.exp(t)*(R_col.T*phi0_col)[0,0] + sp.exp(t)*sum(sp.exp(-phi0)/G)
        if S0 < S:
            t_i = t_i + 0.5
            solver = ode(this_flow).set_integrator(backend, nsteps=1, atol=tol, rtol=tol)
            solver.set_initial_value(phi0, t_i)
            
            # Reset containers
            phis = []
            ts = []
            log_evidence = []
            Qs = []
            ells = []
            log_dets = []
            
            keep_going = True
            max_log_evidence = -sp.Inf
            #print 'Restarting integration at t_i = %f'%t_i
            
        else:
            # Compute betaS directly again to minimize multiplying very large
            # numbers by very small numbers. Also, subtract initial kinetic term
            betaS = beta*0.5*(phi_col.T*Delta*phi_col-kinetic0)[0,0] + N*(R_col.T*phi_col)[0,0] + N              
                                            
            # Compute the log determinant of Lambda
            Lambda = Delta + sp.exp(t)*sp.diag(Q)
            log_det_Lambda, coeff = get_log_det_Lambda(Lambda, coeff)
            log_evidence_value = - betaS - 0.5*log_det_Lambda + 0.5*alpha*t #sp.log(beta) 
            if log_evidence_value > max_log_evidence:
                max_log_evidence = log_evidence_value
            if (log_evidence_value < max_log_evidence - 300) and (ell < G):
                keep_going = False
            
            # Record shit
            phis.append(phi)
            ts.append(t)
            ells.append(ell)
            Qs.append(Q)
            log_evidence.append(log_evidence_value)
            log_dets.append(log_det_Lambda)
        
    warnings.resetwarnings()
    if verbose:
        print 'Integration took %0.2f seconds.'%(time.clock()-integration_start_time)
    
    # Set ts and ells
    ts = sp.array(ts)
    phis = sp.array(phis)
    Qs = sp.array(Qs)
    ells = sp.array(ells)
    log_evidence = sp.array(log_evidence)
    #num_ts = len(ts)
    
    ###
    ### Identify optimal lengthscale
    ###

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
    
    # Get ell range
    log_ells_raw = sp.log(ells)[::-1]
    log_ell_i = max(log_ells_raw)
    log_ell_f = min(log_ells_raw)
    
    # Interploate Qs at 300 different ells
    K = 1000
    phis_raw = phis[::-1,:]
    log_ells_grid = sp.linspace(log_ell_f, log_ell_i, K)
    
    # Create function to get interpolated phis
    phis_interp_func = interp1d(log_ells_raw, phis_raw, axis=0, kind=spline_type)
    
    # Compute weights for each ell on the fine grid
    log_weights_func = interp1d(log_ells_raw, sp.log(ell_weights[::-1]), kind=spline_type)
    log_weights_grid = log_weights_func(log_ells_grid)
    weights_grid = sp.exp(log_weights_grid)
    weights_grid /= sum(weights_grid)
    
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
        #is_sampled = sp.zeros([num_samples])
        log_ells_sampled = sp.zeros([num_samples])
        
        for j in range(num_samples):
            
            # First choose a classical path based on 
            i = choice(K, p=weights_grid)
            phi_cl = phis_interp_func(log_ells_grid[i])
            #is_sampled[j] = int(i)
            log_ells_sampled[j] = log_ells_grid[i]
        
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
        
    # Create interpolated phi_star. Need to extend grid to boundaries first
    extended_xgrid = sp.zeros(L+2)
    extended_xgrid[1:-1] = xgrid
    extended_xgrid[0] = xlb
    extended_xgrid[-1] = xub
    results.extended_xgrid = extended_xgrid
    
    extended_phi_star = sp.zeros(L+2)
    extended_phi_star[1:-1] = phi_star
    end_phi_star = 0.5*(phi_star[0]+phi_star[-1])
    extended_phi_star[0] = end_phi_star
    extended_phi_star[-1] = end_phi_star
    results.extended_phi_star = extended_phi_star
    phi_star_func = interp1d(extended_xgrid, extended_phi_star, kind='cubic')
    Z = sp.sum(dx*sp.exp(-phi_star))
    Q_star_func = lambda(x): sp.exp(-phi_star_func(x))/Z
    
    # If samples are requested, store those too
    if num_samples > 0:
        results.phis_sampled = phis_sampled
        results.Qs_sampled = Qs_sampled/dx
        results.ells_sampled = sp.exp(log_ells_sampled)*dx
    
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

def deft_2d(xis_raw, yis_raw, bbox, G=20, alpha=2, num_samples=0, tol=1E-3, ti_shift=-1, tf_shift=0, verbose=False, details=False):
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
    R_col = sp.mat(R_flat).T
    
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
    
    ###
    ### Integrate ODE
    ###
    
    # Build 2D Laplacian matrix
    delsq_2d = (-4.0*sp.eye(V) + 
        sp.eye(V,V,-1) + sp.eye(V,V,+1) + 
        sp.eye(V,V,-G) + sp.eye(V,V,+G) +
        sp.eye(V,V,-V+1) + sp.eye(V,V,V-1) +
        sp.eye(V,V,-V+G) + sp.eye(V,V,V-G))
    
    # Make Delta, and make it sparse
    Delta = csr_matrix(-delsq_2d)**alpha
  
    # This is the key function: computes deriviate of phi for ODE integration
    def this_flow(t, phi):
        
        # Compute distribution Q corresponding to phi     
        Q = sp.exp(-phi)/V
        A = Delta + sp.exp(t)*diags(Q,0)
        dphidt = sp.real(spsolve(A, sp.exp(t)*(Q-R_flat)))     
                         
        # Return the time derivative of phi
        return dphidt
   
    backend = 'vode'
    solver = ode(this_flow).set_integrator(backend, nsteps=1, atol=tol, rtol=tol)
    solver.set_initial_value(phi0, t_i)
    # suppress Fortran-printed warning
    solver._integrator.iwork[2] = -1
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Make containers        
    phis = []
    ts = []
    log_evidence = []
    Qs = []
    ells = []
    log_dets = []  
    
    # Will keep initial phi to check that integration is working well
    phi0_col = sp.mat(phi0).T
    kinetic0 = (phi0_col.T*Delta*phi0_col)[0,0]
    
    # Integrate phi over specified t range. 
    integration_start_time = time.clock()
    keep_going = True
    max_log_evidence = -sp.Inf
    coeff = 1.0
    while solver.t < t_f and keep_going:
        
        # Step integrator 
        solver.integrate(t_f, step=True)
        
        # Compute deteriminant.
        phi = solver.y
        t = solver.t
        beta = N*sp.exp(-t)
        ell = (N/sp.exp(t))**(1.0/(2.0*alpha-2.0))
        
        # Compute new distribution
        Q = sp.exp(-phi)/sum(sp.exp(-phi)) 
        phi_col = sp.mat(phi).T
        
        # Check that S[phi] < S[phi0]. If not, phi might just be fucked up, due to the 
        # integration having to solve a degenerate system of equations. 
        # In this case, set phi = phi0 and restart integration from there. 
        S = 0.5*(phi_col.T*Delta*phi_col)[0,0] + sp.exp(t)*(R_col.T*phi_col)[0,0] + sp.exp(t)*sum(sp.exp(-phi)/G)
        S0 = 0.5*(phi0_col.T*Delta*phi0_col)[0,0] + sp.exp(t)*(R_col.T*phi0_col)[0,0] + sp.exp(t)*sum(sp.exp(-phi0)/G)
        if S0 < S:
            t_i = t_i + 0.5
            solver = ode(this_flow).set_integrator(backend, nsteps=1, atol=tol, rtol=tol)
            solver.set_initial_value(phi0, t_i)
            
            # Reset containers
            phis = []
            ts = []
            log_evidence = []
            Qs = []
            ells = []
            log_dets = []
            
            keep_going = True
            max_log_evidence = -sp.Inf
            #print 'Restarting integration at t_i = %f'%t_i
            
        else:
            # Compute betaS directly again to minimize multiplying very large
            # numbers by very small numbers. Also, subtract initial kinetic term
            betaS = beta*0.5*(phi_col.T*Delta*phi_col-kinetic0)[0,0] + N*(R_col.T*phi_col)[0,0] + N              
                                            
            # Compute the log determinant of Lambda
            Lambda = Delta + sp.exp(t)*sp.diag(Q)
            log_det_Lambda, coeff = get_log_det_Lambda(Lambda, coeff)
            log_evidence_value = - betaS - 0.5*log_det_Lambda + 0.5*alpha*t #sp.log(beta) 
            if log_evidence_value > max_log_evidence:
                max_log_evidence = log_evidence_value
            if (log_evidence_value < max_log_evidence - 300) and (ell < G):
                keep_going = False
            
            # Record shit
            phis.append(phi)
            ts.append(t)
            ells.append(ell)
            Qs.append(Q)
            log_evidence.append(log_evidence_value)
            log_dets.append(log_det_Lambda)
        
    warnings.resetwarnings()
    if verbose:
        print 'Integration took %0.2f seconds.'%(time.clock()-integration_start_time)  
    
    # Set ts and ells
    ts = sp.array(ts)
    phis = sp.array(phis)
    Qs = sp.array(Qs)
    ells = sp.array(ells)
    log_evidence = sp.array(log_evidence)       
                                
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
  
    # Get ell range
    log_ells_raw = sp.log(ells)[::-1]
    log_ell_i = max(log_ells_raw)
    log_ell_f = min(log_ells_raw)
    
    # Interploate Qs at 300 different ells
    K = 1000
    phis_raw = phis[::-1,:]
    log_ells_grid = sp.linspace(log_ell_f, log_ell_i, K)
    
    # Create function to get interpolated phis
    phis_interp_func = interp1d(log_ells_raw, phis_raw, axis=0, kind='cubic')
    
    # Compute weights for each ell on the fine grid
    log_weights_func = interp1d(log_ells_raw, sp.log(ell_weights[::-1]), kind='cubic')
    log_weights_grid = log_weights_func(log_ells_grid)
    weights_grid = sp.exp(log_weights_grid)
    weights_grid /= sum(weights_grid) 
        
    # If user requests samples
    if num_samples > 0:
    
        Lambda_star = (ell_star**(2.0*alpha-1.0))*(Delta + M_star*sp.diag(Q_star))    
        eigvals, eigvecs = eig(Lambda_star)
        
        # Lambda_star is Hermetian; shouldn't need to do this
        eigvals = sp.real(eigvals) 
        eigvecs = sp.real(eigvecs)                      
        
        # Initialize container variables      
        Qs_sampled = sp.zeros([num_samples, G, G])
        phis_sampled = sp.zeros([num_samples, V])
        #is_sampled = sp.zeros([num_samples])
        log_ells_sampled = sp.zeros([num_samples])
        
        for j in range(num_samples):
            
            # First choose a classical path based on 
            i = choice(K, p=weights_grid)
            phi_cl = phis_interp_func(log_ells_grid[i])
            #is_sampled[j] = int(i)
            log_ells_sampled[j] = log_ells_grid[i]
        
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
    phi_star = sp.reshape(phi_star,[G,G])
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

    # Extend grid for phi_star for interpolating function
    extended_phi_star = sp.zeros([G+2, G+2])
    extended_phi_star[1:-1,1:-1] = phi_star
    
    # Get rows
    row = 0.5*(phi_star[0,:] + phi_star[-1,:])
    extended_phi_star[0,1:-1] = row
    extended_phi_star[-1,1:-1] = row
    
    # Get cols
    col = 0.5*(phi_star[:,0] + phi_star[:,-1])
    extended_phi_star[1:-1,0] = col
    extended_phi_star[1:-1,-1] = col
    
    # Get remaining corners, which share the same value
    corner= 0.25*(row[0]+row[-1]+col[0]+col[-1])
    extended_phi_star[0,0] = corner
    extended_phi_star[0,-1] = corner
    extended_phi_star[-1,0] = corner
    extended_phi_star[-1,-1] = corner

    # Finally, compute interpolated function
    phi_star_func = RectBivariateSpline(extended_xgrid, extended_ygrid, extended_phi_star, bbox=bbox)       
    Z = sp.sum((dx*dy)*sp.exp(-phi_star))
    def Q_star_func(x,y): 
        return sp.exp(-phi_star_func(x,y))/Z      
                   
    # If samples are requested, return those too 
    if num_samples > 0:
        results.phis_sampled = phis_sampled
        results.Qs_sampled = Qs_sampled/(dx*dy)
        results.ells_sampled = sp.exp(log_ells_sampled)*dx*dy
    
    # Stop time
    time_elapsed = time.clock() - start_time
    if verbose:
        print 'deft_2d: %1.2f sec for alpha = %d, G = %d, N = %d'%(time_elapsed, alpha, G, N)
    
    if details:
        return Q_star_func, results
    else:
        return Q_star_func