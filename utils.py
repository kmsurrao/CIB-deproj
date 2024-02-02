import numpy as np
import healpy as hp
import subprocess
import os
import yaml
from scipy import stats
from planck_noise import get_planck_noise
from inpaint_pixels import initial_masking

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
    if not standard_ilc:
        beta_arr = np.linspace(inp.beta_range[0], inp.beta_range[1], num=inp.num_beta_vals)
        write_beta_yamls(inp)
        for beta in beta_arr:
            for i in ['inflated', 'uninflated']:
                if not os.path.isdir(f'{inp.output_dir}/pyilc_outputs/beta_{beta:.2f}_{i}'):
                    subprocess.call(f'mkdir {inp.output_dir}/pyilc_outputs/beta_{beta:.2f}_{i}', shell=True, env=env)
    else:
        for i in ['inflated', 'uninflated']:
            if not os.path.isdir(f'{inp.output_dir}/pyilc_outputs/{i}'):
                subprocess.call(f'mkdir {inp.output_dir}/pyilc_outputs/{i}', shell=True, env=env)

    return


def tsz_spectral_response(freqs):
    '''
    ARGUMENTS
    ---------
    freqs: 1D numpy array, contains frequencies (GHz) for which to calculate tSZ spectral response

    RETURNS
    ---------
    1D array containing tSZ spectral response to each frequency
    '''
    T_cmb = 2.726
    h = 6.62607004*10**(-34)
    kb = 1.38064852*10**(-23)
    response = []
    for freq in freqs:
        x = h*(freq*10**9)/(kb*T_cmb) #x is v/56.9 GHz
        response.append(T_cmb*(x*1/np.tanh(x/2)-4))
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


def cov(inp, Cl):
    '''
    Computes Gaussian covariance in harmonic space

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Cl: 3D numpy array of shape (2, 2, Nbins) containing binned power spectra
        Cl[0,0] = Cl^{hh}
        Cl[0,1] = Cl[1,0] = Cl^{hy}
        Cl[1,1] = Cl^{yy}

    RETURNS
    -------
    covar: 2D numpy array of shape (Nbins, Nbins)
        containing Gaussian covariance matrix
    '''
    Nmodes = 1/((2*inp.mean_ells+1)*inp.ells_per_bin)
    covar = np.diag(Nmodes*(Cl[0,1]**2 + Cl[0,0]*Cl[1,1]))
    return covar


def get_freq_maps(inp):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications

    RETURNS
    -------
    None (writes frequency maps to output_dir)
    '''
    PS_noise_Planck = get_planck_noise(inp)
    planck_freqs = [30, 44, 70, 100, 143, 217, 353, 545]
    tsz_response_vec = tsz_spectral_response(inp.frequencies)
    ymap = hp.read_map(inp.tsz_map_file)
    ymap = hp.ud_grade(ymap, inp.nside) #units of uK
    cib_map_143 = hp.ud_grade(hp.read_map(f'{inp.cib_map_dir}/mdpl2_len_mag_cibmap_planck_143_uk.fits'), inp.nside)
    for i, freq in enumerate(inp.frequencies):
        idx = planck_freqs.index(freq)
        PS_noise = inp.planck_noise_fraction*PS_noise_Planck[idx]
        noise_map = hp.synfast(PS_noise, nside=inp.nside) #units of uK
        tsz_map = tsz_response_vec[i]*ymap
        cib_map = hp.read_map(f'{inp.cib_map_dir}/mdpl2_len_mag_cibmap_planck_{freq}_uk.fits')
        cib_map = hp.ud_grade(cib_map, inp.nside) #units of uK
        cib_map = initial_masking(inp, cib_map, cib_map_143)
        freq_map_uninflated = tsz_map + inp.cib_inflation[0]*cib_map + noise_map
        freq_map_inflated = tsz_map + inp.cib_inflation[1]*cib_map + noise_map
        hp.write_map(f'{inp.output_dir}/maps/uninflated_{freq}.fits', freq_map_uninflated, overwrite=True, dtype=np.float32)
        hp.write_map(f'{inp.output_dir}/maps/inflated_{freq}.fits', freq_map_inflated, overwrite=True, dtype=np.float32)
        if inp.debug:
            print(f'saved {inp.output_dir}/maps/uninflated_{freq}.fits', flush=True)
            print(f'saved {inp.output_dir}/maps/inflated_{freq}.fits', flush=True)
    return

