import numpy as np
import healpy as hp
import subprocess
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
    beta_arr = np.linspace(inp.beta_range[0], inp.beta_range[1], num=inp.num_beta_vals, dtype=np.float32)
    for beta in beta_arr:
        pars = {'beta_CIB': float(beta), 'Tdust_CIB': 24.0, 'nu0_CIB_ghz':353.0, 'kT_e_keV':5.0, 'nu0_radio_ghz':150.0, 'beta_radio': -0.5}
        beta_yaml = f'{inp.output_dir}/pyilc_yaml_files/beta_{beta:.2f}.yaml'
        with open(beta_yaml, 'w') as outfile:
            yaml.dump(pars, outfile, default_flow_style=None)
    return


def setup_output_dir(inp, env, standard_ilc=False):
    '''
    Sets up directory for output files

    ARGUMENTS
    ---------
    inp: Info() object containing input specifications
    env: environment object
    standard_ilc: Bool, whether running standard ILC without CIB deprojection

    RETURNS
    -------
    None
    '''
    if not os.path.isdir(inp.output_dir):
        subprocess.call(f'mkdir {inp.output_dir}', shell=True, env=env)
    if not os.path.isdir(f'{inp.output_dir}/maps'):
        subprocess.call(f'mkdir {inp.output_dir}/maps', shell=True, env=env)
    if not os.path.isdir(f'{inp.output_dir}/results_test'):
        subprocess.call(f'mkdir {inp.output_dir}/results_test', shell=True, env=env)
    if not os.path.isdir(f'{inp.output_dir}/correlation_plots'):
        subprocess.call(f'mkdir {inp.output_dir}/correlation_plots', shell=True, env=env)
    if not os.path.isdir(f'{inp.output_dir}/pyilc_yaml_files'):
        subprocess.call(f'mkdir {inp.output_dir}/pyilc_yaml_files', shell=True, env=env)
    if not os.path.isdir(f'{inp.output_dir}/pyilc_outputs'):
        subprocess.call(f'mkdir {inp.output_dir}/pyilc_outputs', shell=True, env=env)
    if inp.realistic:
        inflation_strs = ['uninflated', 'inflated_realistic']
    else:
        inflation_strs = ['uninflated', 'inflated']
    if not standard_ilc:
        beta_arr = np.linspace(inp.beta_range[0], inp.beta_range[1], num=inp.num_beta_vals)
        write_beta_yamls(inp)
        for beta in beta_arr:
            for i in inflation_strs:
                if not os.path.isdir(f'{inp.output_dir}/pyilc_outputs/beta_{beta:.2f}_{i}'):
                    subprocess.call(f'mkdir {inp.output_dir}/pyilc_outputs/beta_{beta:.2f}_{i}', shell=True, env=env)
    for i in inflation_strs:
        if not os.path.isdir(f'{inp.output_dir}/pyilc_outputs/{i}'):
            subprocess.call(f'mkdir {inp.output_dir}/pyilc_outputs/{i}', shell=True, env=env)
        if not standard_ilc:
            break

    return


def tsz_spectral_response(freqs):
    '''
    ARGUMENTS
    ---------
    freqs: array-like, contains frequencies (GHz) for which to calculate tSZ spectral response

    RETURNS
    ---------
    1D array containing tSZ spectral response to each frequency (units of K_CMB)
    '''
    T_cmb = 2.726
    T_cmb_uK = 2.726e6
    h = 6.62607004*10**(-34)
    kb = 1.38064852*10**(-23)
    response = []
    for freq in freqs:
        x = h*(freq*10**9)/(kb*T_cmb) #x is v/56.9 GHz
        response.append(T_cmb*(x*1/np.tanh(x/2)-4))
    return np.array(response)


def cib_spectral_response(freqs):
    '''
    ARGUMENTS
    ---------
    freqs: 1D numpy array, contains frequencies (GHz) for which to calculate tSZ spectral response

    RETURNS
    ---------
    1D array containing CIB spectral response to each frequency (units of K_CMB)
    '''
    TCMB = 2.726 #Kelvin
    TCMB_uK = 2.726e6 #micro-Kelvin
    hplanck=6.626068e-34 #MKS
    kboltz=1.3806503e-23 #MKS
    clight=299792458.0 #MKS

    # function needed for Planck bandpass integration/conversion following approach in Sec. 3.2 of https://arxiv.org/pdf/1303.5070.pdf
    # blackbody derivative
    # units are 1e-26 Jy/sr/uK_CMB
    def dBnudT(nu_ghz):
        nu = 1.e9*np.asarray(nu_ghz)
        X = hplanck*nu/(kboltz*TCMB)
        return (2.*hplanck*nu**3.)/clight**2. * (np.exp(X))/(np.exp(X)-1.)**2. * X/TCMB_uK

    # conversion from specific intensity to Delta T units (i.e., 1/dBdT|T_CMB)
    #   i.e., from W/m^2/Hz/sr (1e-26 Jy/sr) --> uK_CMB
    #   i.e., you would multiply a map in 1e-26 Jy/sr by this factor to get an output map in uK_CMB
    def ItoDeltaT(nu_ghz):
        return 1./dBnudT(nu_ghz)

    Tdust_CIB = 20.0       #CIB effective dust temperature [K] (Table 9 of http://www.aanda.org/articles/aa/pdf/2014/11/aa22093-13.pdf)
    beta_CIB = 1.668         #CIB modified blackbody spectral index (Table 9 of http://www.aanda.org/articles/aa/pdf/2014/11/aa22093-13.pdf ; Table 10 of that paper contains CIB monopoles)
    nu0_CIB_ghz = 353.0    #CIB pivot frequency [GHz]

    nu_ghz = freqs
    nu = 1.e9*np.asarray(nu_ghz).astype(float)
    X_CIB = hplanck*nu/(kboltz*Tdust_CIB)
    nu0_CIB = nu0_CIB_ghz*1.e9
    X0_CIB = hplanck*nu0_CIB/(kboltz*Tdust_CIB)
    resp = (nu/nu0_CIB)**(3.0+(beta_CIB)) * ((np.exp(X0_CIB) - 1.0) / (np.exp(X_CIB) - 1.0)) * (ItoDeltaT(np.asarray(nu_ghz).astype(float))/ItoDeltaT(nu0_CIB_ghz))
    resp[np.where(nu_ghz == None)] = 0. #this case is appropriate for HI or other maps that contain no CMB-relevant signals (and also no CIB); they're assumed to be denoted by None in nu_ghz
    return resp


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
    Nbins = inp.ellmax//inp.ells_per_bin
    res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=Nbins)
    mean_ells = (res[1][:-1]+res[1][1:])/2
    inp.mean_ells = mean_ells
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
    ells = np.arange(inp.ellmax+1)
    Nmodes = 1/(2*ells+1)
    covar = np.einsum('l,ikl,jml->ijkml', Nmodes, S, S) + np.einsum('l,iml,jkl->ijkml', Nmodes, S, S) \
          + np.einsum('l,ikl,jml->ijkml', Nmodes, S, N) + np.einsum('l,jml,ikl->ijkml', Nmodes, S, N) \
          + np.einsum('l,iml,jkl->ijkml', Nmodes, S, N) + np.einsum('l,jkl,iml->ijkml', Nmodes, S, N) \
          + np.einsum('l,ikl,jml->ijkml', Nmodes, N, N) + np.einsum('l,jkl,iml->ijkml', Nmodes, N, N)
    covar /= Nmodes
    return covar

