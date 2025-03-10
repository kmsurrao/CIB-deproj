import numpy as np
import healpy as hp
from utils import binned
from plot_correlations import plot_corr_harmonic


def harmonic_space_cov(inp, y1, y2, h):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    y1: 1D numpy array in RING format containing reconstructed y map
    y2: 1D numpy array in RING format containing true y map
        (or y map with inflated CIB)
    h: 1D numpy array in RING format containing halo map

    RETURNS
    -------
    cov_hy: (Nbins, Nbins) ndarray containing Gaussian covariance matrix of halos and (y1-y2)
    ''' 
    hh = binned(inp, hp.anafast(h, lmax=inp.ellmax))
    hy = binned(inp, hp.anafast(h, y1-y2, lmax=inp.ellmax))
    yy = binned(inp, hp.anafast(y1-y2, lmax=inp.ellmax))
    Nmodes = (2*inp.mean_ells+1)*inp.ells_per_bin
    cov_hy = np.diag(1/Nmodes*(hy**2 + hh*yy))
    return cov_hy


def cov(inp, beta, h, inflated=False):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    beta: float, value of beta for CIB deprojection
    ra_halos: ndarray containing ra of halos
    dec_halos: ndarray containing dec of halos
    h: 1D numpy array in RING format containing halo map
    inflated: Bool, if True compares y recon. with y recon. (inflated CIB)
            if False, compares y recon. with y true

    RETURNS
    -------
    cov_hy: if in real space, (nrad, nrad) ndarray containing covariance matrix of halos and y_recon
            if in harmonic space, (Nbins, Nbins) ndarray containing Gaussian covariance matrix of halos and y_recon

    '''
    y1 = hp.read_map(f"{inp.output_dir}/pyilc_outputs/beta_{beta:.3f}_uninflated/needletILCmap_component_tSZ_deproject_CIB.fits")
    if inflated:
        y2 = hp.read_map(f"{inp.output_dir}/pyilc_outputs/beta_{beta:.3f}_inflated/needletILCmap_component_tSZ_deproject_CIB.fits")
    else:
        y2 = hp.read_map(f"{inp.output_dir}/pyilc_outputs/uninflated/needletILCmap_component_tSZ.fits")
        y2 = hp.ud_grade(y2, inp.nside)
    cov_hy = harmonic_space_cov(inp, y1, y2, h)
    return cov_hy


def compute_chi2_harmonic_space(inp, y1, y2, h, cov):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    y1: 1D numpy array in RING format containing reconstructed y map
    y2: 1D numpy array in RING format containing true y map
        (or y map with inflated CIB)
    h: 1D numpy array in RING format containing halo map
    cov: (Nbins, Nbins) ndarray containing covariance matrix of halos and (y1-y2)

    RETURNS
    -------
    chi2: (Nbins, ) array containing chi^2 values of <h,y1> and <h,y2> at each ell bin
    hydiff: 1D numpy array of length Nbins containing cross-spectrum of halos with (y1-y2)
    ''' 
    hydiff = binned(inp, hp.anafast(h, y1-y2, lmax=inp.ellmax))
    Nbins = len(hydiff)
    chi2 = np.zeros(Nbins, dtype=np.float32)
    cov_inv = np.linalg.inv(cov)
    for i in range(Nbins):
        chi2[i] = np.dot(hydiff[i], np.dot(cov_inv[i,i], hydiff[i]))
    return chi2, hydiff


def compare_chi2(inp, beta, h):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    beta: float, value of beta for CIB deprojection
    ra_halos: ndarray containing ra of halos
    dec_halos: ndarray containing dec of halos
    h: 1D numpy array in RING format containing halo map

    RETURNS
    -------
    chi2_true: float, chi^2 value of <h,y_reconstructed> and <h,y_true>
    chi2_inflated: float, chi^2 value of <h,y_reconstructed> and <h,y_reconstructed_with_inflated_cib>
    '''
    y_true = hp.read_map(f"{inp.output_dir}/pyilc_outputs/uninflated/needletILCmap_component_tSZ.fits")
    y_true = hp.ud_grade(y_true, inp.nside)
    y_recon = hp.read_map(f"{inp.output_dir}/pyilc_outputs/beta_{beta:.3f}_uninflated/needletILCmap_component_tSZ_deproject_CIB.fits")
    y_recon_inflated = hp.read_map(f"{inp.output_dir}/pyilc_outputs/beta_{beta:.3f}_inflated/needletILCmap_component_tSZ_deproject_CIB.fits")
    chi2_true, hy_true = compute_chi2_harmonic_space(inp, y_recon, y_true, h, inp.cov_hytrue)
    chi2_inflated, hy_infl = compute_chi2_harmonic_space(inp, y_recon, y_recon_inflated, h, inp.cov_hyinfl)
    plot_corr_harmonic(inp, beta, hy_true, hy_infl)
    if inp.debug:
        print(f'chi2_true for beta={beta}: {chi2_true}', flush=True)
        print(f'chi2_inflated for beta={beta}: {chi2_inflated}', flush=True)
    return chi2_true, chi2_inflated


def compare_chi2_star(args):
    '''
    Useful for using multiprocessing imap
    (imap supports tqdm but starmap does not)

    ARGUMENTS
    ---------
    args: arguments to function compare_chi2

    RETURNS
    -------
    function of *args, compare_chi2(inp, beta, h)
    '''
    return compare_chi2(*args)
