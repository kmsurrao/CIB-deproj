import numpy as np
import os
import yaml
from scipy import stats


def write_beta_yamls(inp):
    '''
    Writes yaml files for fg SEDs for each beta value used for CIB deprojection

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications

    RETURNS
    -------
    None
    '''
    if inp.beta_fid not in inp.beta_arr:
        betas_here = [beta for beta in inp.beta_arr] + [inp.beta_fid]
    else:
        betas_here = inp.beta_arr
    for beta in betas_here:
        pars = {'beta_CIB': float(beta), 'Tdust_CIB': 24.0, 'nu0_CIB_ghz':353.0, 'kT_e_keV':5.0, 'nu0_radio_ghz':150.0, 'beta_radio': -0.5}
        beta_yaml = f'{inp.output_dir}/pyilc_yaml_files/beta_{beta:.3f}.yaml'
        with open(beta_yaml, 'w') as outfile:
            yaml.dump(pars, outfile, default_flow_style=None)
    return


def setup_output_dir(inp, standard_ilc=False):
    '''
    Sets up directory for output files

    ARGUMENTS
    ---------
    inp: Info() object containing input specifications
    standard_ilc: Bool, whether running standard ILC without CIB deprojection

    RETURNS
    -------
    None
    '''
    os.makedirs(inp.output_dir, exist_ok=True)
    os.makedirs(f'{inp.output_dir}/pickle_files', exist_ok=True)
    os.makedirs(f'{inp.output_dir}/plots', exist_ok=True)
    os.makedirs(f'{inp.output_dir}/maps', exist_ok=True)
    os.makedirs(f'{inp.output_dir}/correlation_plots', exist_ok=True)
    os.makedirs(f'{inp.output_dir}/pyilc_yaml_files', exist_ok=True)
    os.makedirs(f'{inp.output_dir}/pyilc_outputs', exist_ok=True)
    inflation_strs = ['uninflated', 'inflated']
    if not standard_ilc:
        write_beta_yamls(inp)
        if inp.beta_fid not in inp.beta_arr:
            betas_here = [beta for beta in inp.beta_arr] + [inp.beta_fid]
        else:
            betas_here = inp.beta_arr
        for beta in betas_here:
            for i in inflation_strs:
                os.makedirs(f'{inp.output_dir}/pyilc_outputs/beta_{beta:.3f}_{i}', exist_ok=True)
    for i in inflation_strs:
        os.makedirs(f'{inp.output_dir}/pyilc_outputs/{i}', exist_ok=True)
        if not standard_ilc:
            break
    
    # directory for final y-map
    os.makedirs(f'{inp.output_dir}/pyilc_outputs/final', exist_ok=True)

    return

def dBnudT(nu_ghz):
    '''
    ARGUMENTS
    ---------
    nu_ghz: array-like of frequencies (GHz) to evaluate

    RETURNS
    -------
    blackbody derivative in uK_CMB
    '''
    TCMB = 2.726 #Kelvin
    hplanck=6.626068e-34 #MKS
    kboltz=1.3806503e-23 #MKS
    clight=299792458.0 #MKS
    TCMB_uK = 2.726e6 #micro-Kelvin
    nu = 1.e9*np.asarray(nu_ghz)
    X = hplanck*nu/(kboltz*TCMB)
    return (2.*hplanck*nu**3.)/clight**2. * (np.exp(X))/(np.exp(X)-1.)**2. * X/TCMB_uK


def tsz_delta_sed(freq):
    '''
    ARGUMENTS
    ---------
    freq: float, frequency (GHz) for which to calculate tSZ spectral response in a delta bandpass

    RETURNS
    ---------
    sed: float, tSZ spectral response to freq (units of K_CMB)
    '''
    T_cmb = 2.726
    T_cmb_uK = 2.726e6
    h = 6.62607004*10**(-34)
    kb = 1.38064852*10**(-23)
    x = h*(freq*10**9)/(kb*T_cmb) #x is v/56.9 GHz
    sed = T_cmb*(x*1/np.tanh(x/2)-4)
    return sed


def tsz_spectral_response(freqs, delta_bandpasses=True, inp=None):
    '''
    ARGUMENTS
    ---------
    freqs: array-like, contains frequencies (GHz) for which to calculate tSZ spectral response
    delta_bandpasses: Bool, if True, returns SED at freqs with Delta bandpasses;
        if False, returns SED evaluated at actual Planck bandpasses
    inp: If delta_bandpasses=False, must provide inp (Info object containing input parameter specifications)

    RETURNS
    ---------
    1D array containing tSZ spectral response to each frequency (units of K_CMB)
    '''
    if not delta_bandpasses:
        assert inp is not None, "tsz_spectral_response requires argument 'inp' if delta_bandpasses==False"
    response = []
    for freq in freqs:
        if delta_bandpasses:
            response.append(tsz_delta_sed(freq))
        else:
            bp_path = f'{inp.pyilc_path}/data/HFI_BANDPASS_F{int(freq)}_reformat.txt'
            nu_ghz, trans = np.loadtxt(bp_path, usecols=(0,1), unpack=True)
            delta_resp = np.array([tsz_delta_sed(n) for n in nu_ghz])
            val = np.trapz(trans * dBnudT(nu_ghz) * delta_resp, nu_ghz) / np.trapz(trans * dBnudT(nu_ghz), nu_ghz)
            response.append(val) #K_CMB
    return np.array(response)


def ItoDeltaT(nu_ghz):
    '''
    conversion from specific intensity to Delta T units (i.e., 1/dBdT|T_CMB)
    i.e., from W/m^2/Hz/sr (1e-26 Jy/sr) --> uK_CMB
    i.e., you would multiply a map in 1e-26 Jy/sr by this factor to get an output map in uK_CMB
    '''
    return 1./dBnudT(nu_ghz)


def dBnudT(nu_ghz):
    '''
    function needed for Planck bandpass integration/conversion following approach in Sec. 3.2 of https://arxiv.org/pdf/1303.5070.pdf
    blackbody derivative
    units are 1e-26 Jy/sr/uK_CMB
    '''
    TCMB = 2.726 #Kelvin
    TCMB_uK = 2.726e6 #micro-Kelvin
    hplanck = 6.626068e-34 #MKS
    kboltz = 1.3806503e-23 #MKS
    clight = 299792458.0 #MKS
    nu = 1.e9*np.asarray(nu_ghz)
    X = hplanck*nu/(kboltz*TCMB)
    return (2.*hplanck*nu**3.)/clight**2. * (np.exp(X))/(np.exp(X)-1.)**2. * X/TCMB_uK


def cib_delta_sed(freq, beta=1.65, jy_sr=False):
    '''
    ARGUMENTS
    ---------
    freq: float or array-like, frequency (GHz) for which to calculate CIB spectral response in a delta bandpass
    beta: float or array-like, value of beta for CIB SED 
    jy_sr: Bool, whether to compute SED in Jy/sr

    RETURNS
    ---------
    resp: float (if freq and beta are float/int)
        (Nfreqs, ) numpy array if freq is an array and beta is not,
        (Nfreqs, Nbetas) numpy array if freq and beta are both arrays  
        CIB spectral response to freq (units of K_CMB if jy_sr is False, otherwise in Jy/sr)
    '''
    TCMB = 2.726 #Kelvin
    hplanck = 6.626068e-34 #MKS
    kboltz = 1.3806503e-23 #MKS

    Tdust_CIB = 24.0       #CIB effective dust temperature [K] (Table 9 of http://www.aanda.org/articles/aa/pdf/2014/11/aa22093-13.pdf)
    nu0_CIB_ghz = 353.0    #CIB pivot frequency [GHz]

    nu_ghz = np.atleast_1d(freq).astype(float)
    beta_arr = np.atleast_1d(beta).astype(float)

    nu = 1.e9*np.asarray(nu_ghz).astype(float)
    X_CIB = hplanck*nu/(kboltz*Tdust_CIB) # shape (Nfreqs, )
    nu0_CIB = nu0_CIB_ghz*1.e9
    X0_CIB = hplanck*nu0_CIB/(kboltz*Tdust_CIB)
    resp_part1 = ((nu / nu0_CIB)[:, np.newaxis]) ** (3.0 + beta_arr) # (Nfreqs, Nbeta)
    resp_part2 = ((np.exp(X0_CIB) - 1.0) / (np.exp(X_CIB) - 1.0)) # (Nfreqs, )
    resp =  resp_part1 * resp_part2[:, np.newaxis] # (Nfreqs, Nbeta)
    if not jy_sr:
        resp *= (ItoDeltaT(np.asarray(nu_ghz).astype(float))/ItoDeltaT(nu0_CIB_ghz))[:, np.newaxis]
    
    Nfreqs = nu_ghz.size
    Nbetas = beta_arr.size
    if (Nfreqs == 1) and (Nbetas == 1):
        return resp[0, 0]
    elif (Nfreqs > 1) and (Nbetas == 1):
        return resp[:, 0]
    elif (Nfreqs == 1) and (Nbetas > 1):
        return resp[0, :]
    else:
        return resp


def cib_spectral_response(freqs, delta_bandpasses=True, inp=None, beta=1.65):
    '''
    ARGUMENTS
    ---------
    freqs: array-like, contains frequencies (GHz) for which to calculate CIB spectral response
    delta_bandpasses: Bool, if True, returns SED at freqs with Delta bandpasses;
        if False, returns SED evaluated at actual Planck bandpasses
    inp: If delta_bandpasses=False, must provide inp (Info object containing input parameter specifications)
    beta: float or array-like, value(s) of beta to compute CIB SED

    RETURNS
    ---------
    if beta is a float: (Nfreqs, ) array containing CIB spectral response to each frequency (units of K_CMB)
    if beta is an array: (Nfreqs, Nbetas) array containing CIB spectral response to each frequency using each beta (K_CMB)
    '''

    freqs = np.atleast_1d(freqs).astype(float)
    beta  = np.atleast_1d(beta).astype(float)
    if not delta_bandpasses:
        assert inp is not None, "cib_spectral_response requires argument 'inp' if delta_bandpasses==False"
    response = []
    if delta_bandpasses:
        return cib_delta_sed(freqs, beta, False) # CMB thermodynamic temperature units
    for freq in freqs:
        bp_path = f'{inp.pyilc_path}/data/HFI_BANDPASS_F{int(freq)}_reformat.txt'
        nu_ghz, trans = np.loadtxt(bp_path, usecols=(0,1), unpack=True)
        delta_resp = cib_delta_sed(nu_ghz, beta, True) #Jy/sr
        if delta_resp.ndim == 1:
            delta_resp = delta_resp[:, None]   # shape (N_nu, 1)
        trans_2d = trans[:, None] # shape (N_nu, 1)
        numerator = np.trapz(trans_2d * delta_resp, x=nu_ghz, axis=0) 
        denom = np.trapz(trans * dBnudT(nu_ghz), x=nu_ghz)
        val = numerator / denom  # shape (N_beta,) or (1,)
        response.append(val) # CMB thermodynamic temperature units
    response = np.array(response) #  shape (Nfreqs, N_beta) or (Nfreqs, 1)
    if beta.size == 1:
        return response[:, 0]
    return response


def dbeta_delta_sed(freq, beta=1.65, jy_sr=False):
    '''
    ARGUMENTS
    ---------
    freq: float, frequency (GHz) for which to calculate tSZ spectral response in a delta bandpass
    beta: float, value of beta for CIB SED 
    jy_sr: Bool, whether to compute SED in Jy/sr

    RETURNS
    ---------
    sed: float, dbeta spectral response to freq
    '''
    hplanck=6.626068e-34   #MKS
    kboltz=1.3806503e-23   #MKS
    nu0_CIB_ghz = 353.0    #CIB pivot frequency [GHz]
    Tdust_CIB = 24.0       #CIB effective dust temperature [K] (Table 9 of http://www.aanda.org/articles/aa/pdf/2014/11/aa22093-13.pdf)
    nu_ghz = [freq]
    nu = 1.e9*np.asarray(nu_ghz).astype(float)
    X_CIB = hplanck*nu/(kboltz*(Tdust_CIB))
    nu0_CIB = nu0_CIB_ghz*1.e9
    X0_CIB = hplanck*nu0_CIB/(kboltz*(Tdust_CIB))
    resp = np.log(nu/nu0_CIB)*(nu/nu0_CIB)**(3.0+(beta)) * ((np.exp(X0_CIB) - 1.0) / (np.exp(X_CIB) - 1.0)) 
    if not jy_sr:
        resp *= (ItoDeltaT(np.asarray(nu_ghz).astype(float))/ItoDeltaT(nu0_CIB_ghz))
    resp[np.where(nu_ghz == None)] = 0. #this case is appropriate for HI or other maps that contain no CMB-relevant signals (and also no CIB); they're assumed to be denoted by None in nu_ghz
    sed = resp[0]
    return sed


def dbeta_spectral_response(freqs, delta_bandpasses=True, inp=None, beta=1.65):
    '''
    ARGUMENTS
    ---------
    freqs: array-like, contains frequencies (GHz) for which to calculate CIB spectral response
    delta_bandpasses: Bool, if True, returns SED at freqs with Delta bandpasses;
        if False, returns SED evaluated at actual Planck bandpasses
    inp: If delta_bandpasses=False, must provide inp (Info object containing input parameter specifications)
    beta: float, value of beta for CIB SED

    RETURNS
    ---------
    1D array containing CIB spectral response to each frequency (units of K_CMB)

    '''
    if not delta_bandpasses:
        assert inp is not None, "dbeta_spectral_response requires argument 'inp' if delta_bandpasses==False"
    response = []
    for freq in freqs:
        if delta_bandpasses:
            response.append(dbeta_delta_sed(freq, beta, False))
        else:
            bp_path = f'{inp.pyilc_path}/data/HFI_BANDPASS_F{int(freq)}_reformat.txt'
            nu_ghz, trans = np.loadtxt(bp_path, usecols=(0,1), unpack=True)
            delta_resp = np.array([dbeta_delta_sed(n, beta, True) for n in nu_ghz])
            vnorm = np.trapz(trans * dBnudT(nu_ghz), nu_ghz)
            val = np.trapz(trans * delta_resp , nu_ghz) / vnorm
            response.append(val) 
    return np.array(response)



def binned(inp, spectrum):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    spectrum: 1D numpy array of length ellmax+1 containing some power spectrum

    RETURNS
    -------
    binned_spectrum: 1D numpy array of length Nbins containing binned power spectrum
    '''
    ells = np.arange(inp.ellmax+1)
    Dl = ells*(ells+1)/2/np.pi*spectrum
    Nbins = int(np.round((inp.ellmax-inp.ellmin+1)/inp.ells_per_bin))
    res = stats.binned_statistic(ells[inp.ellmin:], Dl[inp.ellmin:], statistic='mean', bins=Nbins)
    mean_ells = (res[1][:-1]+res[1][1:])/2
    inp.mean_ells = mean_ells
    inp.bin_edges = res[1]
    binned_spectrum = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)
    return binned_spectrum


def multifrequency_cov(inp, S, N):
    '''
    Computes multifrequency Gaussian covariance in harmonic space

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    S: 3D numpy array of shape (Nfreqs, Nfreqs, ellmax+1) containing power spectrum of signal
    N: 3D numpy array of shape (Nfreqs, Nfreqs, ellmax+1) containing power spectrum of noise
        (only nonzero for frequency-frequency auto-spectra)

    RETURNS
    -------
    covar: 5D numpy array of shape (Nfreqs, Nfreqs, Nfreqs, Nfreqs, ellmax+1)
        containing Gaussian covariance matrix
    '''
    ells = np.arange(inp.ellmin, inp.ellmax+1)
    Nmodes = 1/(2*ells+1)
    covar = np.einsum('l,ikl,jml->ijkml', Nmodes, S, S) + np.einsum('l,iml,jkl->ijkml', Nmodes, S, S) \
          + np.einsum('l,ikl,jml->ijkml', Nmodes, S, N) + np.einsum('l,jml,ikl->ijkml', Nmodes, S, N) \
          + np.einsum('l,iml,jkl->ijkml', Nmodes, S, N) + np.einsum('l,jkl,iml->ijkml', Nmodes, S, N) \
          + np.einsum('l,ikl,jml->ijkml', Nmodes, N, N) + np.einsum('l,jkl,iml->ijkml', Nmodes, N, N)
    covar /= Nmodes
    return covar

