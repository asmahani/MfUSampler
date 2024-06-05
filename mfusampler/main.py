import numpy as np

def _MfU_fEval(xk, k, x, fMulti, **kwargs):
    x[k] = xk
    return fMulti(x, **kwargs)

def _scalar_or_length_n(x, n):
    if np.isscalar(x):
        return np.repeat(x, n)
    elif len(x) == n:
        return x
    else:
        raise ValueError("Input must be of length 1 or 'n'")

def MfU_control(
    n
    , slice_w = 1.0, slice_m = np.inf
    , slice_lower = -np.inf, slice_upper = +np.inf
):
    """
    Generate control parameters for univariate samplers (currently, slice sampler only).
    Any parameter that is provided as a scalar will be replicated for all dimensions.
    
    Parameters
    ----------
    n : int
        Dimensionality of random variable vector whose density we wish to draw MCMC samples from.
    slice_w : float or array-like, optional
        The size of the steps for creating intervals in the slice sampler (default is 1.0).
    slice_m : float or array-like, optional
        The limit on the number of steps for expanding intervals in the slice sampler (default is infinity).
    slice_lower : float or array-like, optional
        The lower bounds for the random-variable vector, to be used in the slice sampler (default is -infinity).
    slice_upper : float or array-like, optional
        The upper bounds for the random variable vector, to be used in the slice sampler (default is +infinity).
    
    Returns
    -------
    dict
        A dictionary containing the control parameters for the univariate samplers (currently, slice sampler only).
    """
    slice_w = _scalar_or_length_n(slice_w, n)
    slice_m = _scalar_or_length_n(slice_m, n)
    slice_lower = _scalar_or_length_n(slice_lower, n)
    slice_upper = _scalar_or_length_n(slice_upper, n)
    ret = {'slice': {'w': slice_w, 'm': slice_m, 'lower': slice_lower, 'upper': slice_upper}}
    return ret

def MfU_sample(x, f, uni_sampler = 'slice', uni_sampler_control = None, **kwargs):
    """
    Perform sampling of a (log-)density using a specified univariate sampler.

    Parameters
    ----------
    x : array-like
        Initial points for sampling.
    f : callable
        Function returning the log of the probability density (plus constant) for the target distribution.
    uni_sampler : str, optional
        The type of univariate sampler to use (default is 'slice').
    uni_sampler_control : dict, optional
        Control parameters for the univariate sampler. If None, default control parameters will be used.
    **kwargs : additional keyword arguments
        Additional arguments to pass to the log density function f.

    Returns
    -------
    array-like
        A single MCMC sample (of same length as x) for the target distribution.
    
    Raises
    ------
    ValueError
        If an invalid univariate sampler is specified.
    """
    N = len(x)
    if uni_sampler_control is None:
        uni_sampler_control = MfU_control(N)
    control_slice = uni_sampler_control['slice']
    for n in range(N):
        if uni_sampler == 'slice':
            x[n] = _uni_slice(
                x[n], f = _MfU_fEval, k = n, x = x, fMulti = f
                , w = control_slice['w'][n]
                , m = control_slice['m'][n]
                , lower = control_slice['lower'][n]
                , upper = control_slice['upper'][n]
                , **kwargs
            )
        else:
            raise ValueError('Invalid univariate sampler')
    return x

def MfU_sample_run(x, f, uni_sampler = 'slice', uni_sampler_control = None, nsmp = 10, **kwargs):
    """
    Generating multiple MCMC samples for a (log-)density using a specified univariate sampler.

    Parameters
    ----------
    x : array-like
        Initial points for sampling.
    f : callable
        Function returning the log of the probability density (plus constant) for the target distribution.
    uni_sampler : str, optional
        The type of univariate sampler to use (default is 'slice').
    uni_sampler_control : dict, optional
        Control parameters for the univariate sampler. If None, default control parameters will be used.
    nsmp : int, optional
        The number of samples to draw (default is 10).
    **kwargs : additional keyword arguments
        Additional arguments to pass to the log density function f.

    Returns
    -------
    np.ndarray
        Array of shape (nsmp, len(x)) containing the sampled points.
    """
    xall = np.empty([nsmp, len(x)])
    for n in range(nsmp):
        x = MfU_sample(x, f, uni_sampler, uni_sampler_control, **kwargs)
        xall[n, :] = x
    return xall

def _uni_slice(x0, f, w=1, m=np.inf, lower=-np.inf, upper=np.inf, gx0=None, **kwargs):
    #print([w, m, lower, upper])
    """
    Perform univariate slice sampling.
    
    Parameters
    ----------
    x0 : float
        Initial point.
    f : callable
        Function returning the log of the probability density (plus constant).
    w : float, optional
        Size of the steps for creating interval (default is 1).
    m : int, optional
        Limit on steps (default is infinity).
    lower : float, optional
        Lower bound on support of the distribution (default is -infinity).
    upper : float, optional
        Upper bound on support of the distribution (default is +infinity).
    gx0 : float, optional
        Value of f(x0), if known (default is None).
    *kwargs : additional (keyword) arguments
        Additional (keyword) arguments to pass to the log density function f.
    
    Returns
    -------
    x1 : float
        New point sampled.
    """
    
    # Define a wrapper for the log density function
    def g(x):
        return f(x, **kwargs)
    
    # Check the validity of the arguments
    if not (isinstance(x0, (int, float)) and isinstance(w, (int, float)) and w > 0 and
            (isinstance(m, (int, float)) and (m == np.inf or (m > 0 and m <= 1e9 and float(m).is_integer()))) and
            isinstance(lower, (int, float)) and isinstance(upper, (int, float)) and upper > lower and
            (gx0 is None or isinstance(gx0, (int, float)))):
        raise ValueError("Invalid slice sampling argument")
    
    # Find the log density at the initial point, if not already known
    if gx0 is None:
        gx0 = g(x0)
    
    # Determine the slice level, in log terms
    logy = gx0 - np.random.exponential()
    
    # Find the initial interval to sample from
    u = np.random.uniform(0, w)
    L = x0 - u
    R = x0 + (w - u)  # should guarantee that x0 is in [L, R], even with roundoff
    
    # Expand the interval until its ends are outside the slice, or until the limit on steps is reached
    if np.isinf(m):  # no limit on number of steps
        while L > lower and g(L) > logy:
            L -= w
        while R < upper and g(R) > logy:
            R += w
    elif m > 1:  # limit on steps, bigger than one
        J = int(np.floor(np.random.uniform(0, m)))
        K = int(m - 1 - J)
        
        while J > 0 and L > lower and g(L) > logy:
            L -= w
            J -= 1
        while K > 0 and R < upper and g(R) > logy:
            R += w
            K -= 1
    
    # Shrink interval to lower and upper bounds
    L = max(L, lower)
    R = min(R, upper)
    
    # Sample from the interval, shrinking it on each rejection
    while True:
        x1 = np.random.uniform(L, R)
        gx1 = g(x1)
        
        if gx1 >= logy:
            break
        
        if x1 > x0:
            R = x1
        else:
            L = x1
    
    # Return the point sampled
    return x1
