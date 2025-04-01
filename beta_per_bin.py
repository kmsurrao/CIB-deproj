import numpy as np
import scipy.interpolate as interp
import scipy.optimize as opt
from scipy.optimize import least_squares


def compute_1sigma_beta(inp, chi2_arr,  a):
    """
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    chi2_arr: (Nbetas,) array containing chi^2 value using deprojection
        of the CIB with each value beta, for bin 'a'.
    a: int, bin number

    RETURNS
    -------
    x_min : float, beta with minimum chi^2
    x_hi  : float, beta at high end of 1-sigma range
    x_lo  : float, beta at low end of 1-sigma range
    """
    # 1) Interpolate
    x_vals = np.array(inp.beta_arr)
    y_vals = chi2_arr
    kind = 'cubic' if len(x_vals) >= 4 else 'linear'
    spline = interp.interp1d(x_vals, y_vals, kind=kind)

    # 2) Find overall minimum
    idx_min = np.argmin(y_vals)
    x_min = x_vals[idx_min]
    y_min = y_vals[idx_min]

    # We'll search for f(x) = target_value on each side of x_min
    target_value = y_min + 1.0

    def equation(x):
        return spline(x) - target_value

    # Helper: bracket-based root finder on intervals that show a sign change
    def bracketed_roots(fun, xs):
        """
        fun: callable, the function for which we want to find fun(x) = 0
        xs : sorted array of x-values spanning some region
        Returns all roots found by scanning adjacent intervals for a sign change
        """
        roots_found = []
        fvals = [fun(x) for x in xs]
        for i in range(len(xs) - 1):
            f1, f2 = fvals[i], fvals[i+1]
            if f1 == 0.0:
                # Exactly zero at boundary
                roots_found.append(xs[i])
            elif f1 * f2 < 0.0:
                # Sign change -> bracket
                x_left, x_right = xs[i], xs[i+1]
                try:
                    r = opt.brentq(fun, x_left, x_right)
                    roots_found.append(r)
                except ValueError:
                    pass  # No solution in that bracket
        return roots_found

    # 3) Search on the left side of x_min
    #    i.e. all beta <= x_min
    left_side = x_vals[x_vals <= x_min]
    # Sort in ascending order (just in case)
    left_side = np.sort(left_side)
    roots_left = bracketed_roots(equation, left_side)
    if len(roots_left) == 0:
        # Fallback if we cannot find a crossing
        print(f"Trouble finding lower 1-sigma intersection for beta in bin {a}. "
              f"Setting x_lo = {left_side[0]:.3f}")
        x_lo = left_side[0]
    else:
        # Pick the root that is closest to x_min from below
        # i.e. the largest among roots_left but still <= x_min
        valid_left = [r for r in roots_left if r <= x_min]
        if len(valid_left) == 0:
            # If all found roots are above x_min, fallback
            x_lo = left_side[0]
        else:
            x_lo = max(valid_left)

    # 4) Search on the right side of x_min
    right_side = x_vals[x_vals >= x_min]
    right_side = np.sort(right_side)
    roots_right = bracketed_roots(equation, right_side)
    if len(roots_right) == 0:
        print(f"Trouble finding upper 1-sigma intersection for beta in bin {a}. "
              f"Setting x_hi = {right_side[-1]:.3f}")
        x_hi = right_side[-1]
    else:
        # Pick the root that is closest to x_min from above
        # i.e. the smallest among roots_right but still >= x_min
        valid_right = [r for r in roots_right if r >= x_min]
        if len(valid_right) == 0:
            # If all found roots are below x_min, fallback
            x_hi = right_side[-1]
        else:
            x_hi = min(valid_right)

    return x_min, x_hi, x_lo


def get_all_1sigma_beta(inp, chi2_true, chi2_infl):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    chi2_true: (Nbins, Nbetas) ndarray containing chi2 values for idealized procedure
    chi2_infl: (Nbins, Nbetas) ndarray containing chi2 values for realistic procedure

    RETURNS
    -------
    means_true: (Nbins, ) array of mean beta values (for idealized pipeline)
    uppers_true: (Nbins, ) array of upper errors on beta (for idealized pipeline)
    lowers_true: (Nbins, ) array of lower errors on beta (for idealized pipeline)
    means_infl: (Nbins, ) array of mean beta values (for realistic pipeline)
    uppers_infl: (Nbins, ) array of upper errors on beta (for realistic pipeline)
    lowers_infl: (Nbins, ) array of lower errors on beta (for realistic pipeline)
    '''
    means_true, uppers_true, lowers_true = [], [], []
    means_infl, uppers_infl, lowers_infl = [], [], []
    Nbins = len(chi2_true)
    for a in range(Nbins):
        x_min, x2, x1 = compute_1sigma_beta(inp, chi2_true[a], a)
        upper = x2-x_min
        lower = x_min-x1
        if upper == 0:
            upper = float('inf')
        if lower == 0:
            lower = float('inf')
        means_true.append(x_min)
        uppers_true.append(upper)
        lowers_true.append(lower)
    for a in range(Nbins):
        x_min, x2, x1 = compute_1sigma_beta(inp, chi2_infl[a], a)
        upper = x2-x_min
        lower = x_min-x1
        if upper == 0:
            upper = float('inf')
        if lower == 0:
            lower = float('inf')
        means_infl.append(x_min)
        uppers_infl.append(upper)
        lowers_infl.append(lower)
    return means_true, uppers_true, lowers_true, means_infl, uppers_infl, lowers_infl


def model(x, *coeffs):
    '''
    Model for polynomial equation 
    
    ARGUMENTS
    ---------
    x: float or array-like of values to pass into polynomial
    coeffs: coefficients of polynomial

    RETURNS
    -------
    evaluation of polynomial of degree len(coeffs)-1 given x and coeffs 
    '''
    # e.g. coeffs = [a0, a1, a2] for a quadratic
    # poly(x) = a0 + a1*x + a2*x^2 + ...
    return np.polyval(coeffs[::-1], x)


def residual_asymmetric(params, x, y, yerr_lo, yerr_hi):
    '''
    Function for handling asymmetric error bars using residual

    RETURNS
    -------
    dimensionless residual r/sigma
    '''
    y_model = model(x, *params)
    r = y - y_model
    # If r>0 => data is above model => use lower error
    # If r<0 => data is below model => use upper error
    sigma = np.where(r >= 0, yerr_lo, yerr_hi)
    return r / sigma


def find_fit(x_data, y_data, yerr_lo, yerr_hi, deg):
    '''  
    ARGUMENTS
    ---------
    x_data: (Nbins, ) array of mean ells in bins
    y_data: (Nbins, ) array of best fit beta in each bin
    yerr_lo: (Nbins, ) array of lower error bars on beta in each bin
    yerr_hi: (Nbins, ) array of upper error bars on beta in each bin
    deg: int, degree of polynomial to fit

    RETURNS
    -------
    popt: (deg+1, ) array of best fit parameters
    pcov: (deg+1, deg+1) array of covariance of best fit parameters
    '''
    
    all_popt, all_pcov, all_chi2 = [], [], []
    
    #try 10 different starting points
    np.random.seed(0)
    all_p0 = [-np.ones(deg+1), np.zeros(deg+1), np.ones(deg+1)]
    for i in range(7):
        all_p0.append(2*np.random.rand(deg+1)-1)

    for p0 in all_p0:

        # Run least_squares
        res = least_squares(
            fun=residual_asymmetric,
            x0=p0,
            args=(x_data, y_data, yerr_lo, yerr_hi)
        )

        # Best-fit parameters:
        popt = res.x
        all_popt.append(popt)

        # To estimate the covariance matrix:
        J = res.jac                  # Jacobian at solution, shape (n_data, n_params)
        residuals = res.fun         # final residuals (dimensionless)
        chi2 = np.sum(residuals**2) # sum of squares of residuals
        all_chi2.append(chi2)
        n_data = len(residuals)
        n_par = len(popt)

        # Reduced chi^2
        reduced_chi2 = chi2 / (n_data - n_par)

        # Covariance ~ (J^T J)^(-1) * (chi^2 / dof)
        # (assuming the typical linearized approximation)
        JTJ_inv = np.linalg.inv(J.T @ J)
        pcov = reduced_chi2 * JTJ_inv
        all_pcov.append(pcov)
    
    idx = all_chi2.index(min(all_chi2))
    popt, pcov = all_popt[idx], all_pcov[idx]
        
    return popt, pcov


def predict_with_uncertainty(x_data, y_data, yerr_lo, yerr_hi, deg=3):
    '''  
    Compute y_fit and its 1σ uncertainty at each x in x_array.

    ARGUMENTS
    ---------
    x_data: (Nbins, ) array of mean ells in bins
    y_data: (Nbins, ) array of best fit beta in each bin
    yerr_lo: (Nbins, ) array of lower error bars on beta in each bin
    yerr_hi: (Nbins, ) array of upper error bars on beta in each bin
    deg: int, degree of polynomial to fit

    RETURNS
    -------
    y_fit: (Nbins, ) array with predicted beta in each bin based on fit
    y_fit_err: (Nbins, ) array with error (symmetric) in predicted beta in each bin
    popt: (deg+1, ) array with optimal parameter values for the model
    '''
    popt, pcov = find_fit(x_data, y_data, yerr_lo, yerr_hi, deg)

    # Evaluate the polynomial with the best-fit parameters
    y_fit = model(x_data, *popt)

    # Build the Jacobian matrix J for each x:
    # If popt has length n => polynomial degree (n-1),
    # J[i, :] = [1, x_i, x_i^2, ..., x_i^(n-1)].
    n_params = len(popt)
    Xmat = np.zeros((len(x_data), n_params))
    for i, xval in enumerate(x_data):
        for j in range(n_params):
            Xmat[i, j] = xval**j

    # Propagate variance: var(y_i) = J[i,:] * pcov * J[i,:]^T
    # => we can do this row-by-row or with a matrix product trick:
    var_y = np.sum((Xmat @ pcov) * Xmat, axis=1)

    # Numerical safety: clip very small negative values to 0
    var_y = np.clip(var_y, 0, None)

    y_fit_err = np.sqrt(var_y)
    return y_fit, y_fit_err, popt


