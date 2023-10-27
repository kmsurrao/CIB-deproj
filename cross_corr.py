import numpy as np
import healpy as hp
from get_y_map import setup_pyilc


def compute_chi2(inp, h, y1, y2):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    h: 1D numpy array in RING format containing map of halos
    y1: 1D numpy array in RING format containing reconstructed y map
    y2: 1D numpy array in RING format containing true y map
        (or y map with inflated CIB)

    RETURNS
    -------
    chi2: float, chi^2 value of <h,y1> and <h,y2>
    '''
    ells = np.arange(inp.ellmax+1)
    cross_spectrum1 = hp.anafast(h, y1, lmax=inp.ellmax)
    cross_spectrum2 = hp.anafast(h, y2, lmax=inp.ellmax)
    var = 2/(2*ells+1)*cross_spectrum1**2
    chi2 = np.sum((cross_spectrum1-cross_spectrum2)**2/var)
    return chi2


def compare_chi2(inp, env, beta, h):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    h: 1D numpy array in RING format containing map of halos
    y1: 1D numpy array in RING format containing reconstructed y map
    y2: 1D numpy array in RING format containing true y map
        (or y map with inflated CIB)

    RETURNS
    -------
    chi2_true: float, chi^2 value of <h,y_reconstructed> and <h,y_true>
    chi2_inflated: float, chi^2 value of <h,y_reconstructed> and <h,y_reconstructed_with_inflated_cib>
    '''
    y_true = hp.read_map(inp.tsz_map_file)
    y_recon = setup_pyilc(inp, env, beta, suppress_printing=False, inflated=False)
    y_recon_inflated = setup_pyilc(inp, env, beta, suppress_printing=False, inflated=True)
    chi2_true = compute_chi2(inp, h, y_recon, y_true)
    chi2_inflated = compute_chi2(inp, h, y_recon, y_recon_inflated)
    return chi2_true, chi2_inflated
