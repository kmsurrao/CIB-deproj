import numpy as np
import healpy as hp
import treecorr
import os
from scipy import stats
from get_y_map import setup_pyilc
from utils import binned
from plot_correlations import plot_corr

def randsphere(num, ra_range=[0,360], dec_range=[-90,90]):
    '''
    ARGUMENTS
    ---------
    num: int, number of random halos to generate
    ra_range: array-like, range of ra values to use for random halo generation
    dec_range: array-like, range of dec values to use for random halo generation

    RETURNS
    -------
    ra: ndarray of size num containing ra values
    dec: ndarray of size num containing dec values 
    '''
    
    rng = np.random.RandomState()
    ra = rng.uniform(low=ra_range[0], high=ra_range[1], size=num)
    cosdec_min = np.cos(np.deg2rad(90.0 + dec_range[0]))
    cosdec_max = np.cos(np.deg2rad(90.0 + dec_range[1]))

    v = rng.uniform(low=cosdec_min, high=cosdec_max, size=num)
    v = rng.uniform(low=cosdec_min, high=cosdec_max, size=num)

    np.clip(v, -1.0, 1.0, v)

    # Now this generates on [0,pi)
    dec = np.arccos(v)

    # convert to degrees
    np.rad2deg(dec, dec)

    # now in range [-90,90.0)
    dec -= 90.0
    return ra, dec



def compute_chi2_real_space(inp, y1, y2, ra_halos, dec_halos, beta):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    y1: 1D numpy array in RING format containing reconstructed y map
    y2: 1D numpy array in RING format containing true y map
        (or y map with inflated CIB)
    ra_halos: ndarray containing ra of halos
    dec_halos: ndarray containing dec of halos
    beta: float, beta value used for CIB deprojection (just used for file naming here)

    RETURNS
    -------
    chi2: float, chi^2 value of <h,y1> and <h,y2>
    '''
    rand_ra, rand_dec = randsphere(5*len(ra_halos), ra_range=[0,360], dec_range=[-90,90])
    ra_hp, dec_hp = hp.pix2ang(inp.nside, ipix=np.arange(hp.nside2npix(inp.nside)),lonlat=True)
    
    save_dir = f'{inp.output_dir}/results_test/'
    save_filename_jk_obj = f'jk_obj_test_beta{beta:.2f}.pkl'

    # number of patches to divide the full sky on. More of these, better the covariance estimate would be. Just make sure the size of patches is larger than the maximum separation you are interested in.
    njk = 512
    nthreads = 256//(4*inp.num_beta_vals)
    # This is the accuracy setting. If the bin_slop is set to 0.0, it will take longer to run but correlation would be correct. If you want to run it faster, increase the bin_slop to 0.1 or 0.2. This will make the code run faster, but the correlation estimate will be less accurate.
    bin_slop = 0.0

    # minrad and maxrad are the minimum and maximum separation you are interested in. This is in arcmin. nrad is the number of bins you want to divide the separation range into.
    minrad = 3.0
    maxrad = 100.0
    nrad = 10

    # Load the catalogs in treecorr format
    if os.path.isfile(save_dir + save_filename_jk_obj):
        datapoint_cat = treecorr.Catalog(ra=ra_halos, dec=dec_halos, ra_units='degrees',
                                    dec_units='degrees', patch_centers=save_dir + save_filename_jk_obj)           

    else:
        datapoint_cat = treecorr.Catalog(ra=ra_halos, dec=dec_halos,  ra_units='degrees',
                                        dec_units='degrees', npatch=njk)
        datapoint_cat.write_patch_centers(save_dir + save_filename_jk_obj)
    
    # Create random catalog and catalogs for y1 and y2
    rand_cat = treecorr.Catalog(ra=rand_ra, dec=rand_dec, ra_units='degrees', dec_units='degrees', patch_centers=save_dir + save_filename_jk_obj)
    y1_cat = treecorr.Catalog(ra=ra_hp, dec=dec_hp, k=y1, ra_units='degrees', dec_units='degrees')
    y2_cat = treecorr.Catalog(ra=ra_hp, dec=dec_hp, k=y2, ra_units='degrees', dec_units='degrees')

    # create the correlation objects:
    hy1 = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin', verbose=0, num_threads=nthreads, bin_slop=bin_slop, var_method='jackknife')
    hy2 = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin', verbose=0, num_threads=nthreads, bin_slop=bin_slop, var_method='jackknife')
    ry1 = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin', verbose=0, num_threads=nthreads, bin_slop=bin_slop, var_method='jackknife')
    ry2 = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin', verbose=0, num_threads=nthreads, bin_slop=bin_slop, var_method='jackknife')

    hy1.process(datapoint_cat, y1_cat)
    hy2.process(datapoint_cat, y2_cat)
    ry1.process(rand_cat, y1_cat)
    ry2.process(rand_cat, y2_cat)

    hy1.calculateXi(rk=ry1)
    hy2.calculateXi(rk=ry2)

    xi_hy1 = hy1.xi
    xi_hy2 = hy2.xi
    cov_hy1 = hy1.cov
    cov_hy2 = hy2.cov

    cov_tot = cov_hy1 + cov_hy2
    diff_DV = xi_hy1 - xi_hy2
    chi2 = np.dot(diff_DV, np.dot(np.linalg.inv(cov_tot), diff_DV))

    return chi2


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


def compute_chi2_harmonic_space(inp, y1, y2, h):
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
    chi2: float, chi^2 value of <h,y1> and <h,y2>
    '''
    hy1 = binned(inp, hp.anafast(h, y1, lmax=inp.ellmax))
    hy2 = binned(inp, hp.anafast(h, y2, lmax=inp.ellmax))
    hh = binned(inp, hp.anafast(h, lmax=inp.ellmax))
    y1y1 = binned(inp, hp.anafast(y1, lmax=inp.ellmax))
    y2y2 = binned(inp, hp.anafast(y2, lmax=inp.ellmax))
    cov_hy1 = cov(inp, np.array([[hh, hy1], [hy1, y1y1]]))
    cov_hy2 = cov(inp, np.array([[hh, hy2], [hy2, y2y2]]))
    cov_tot = cov_hy1 + cov_hy2
    diff_DV = hy1-hy2
    chi2 = np.dot(diff_DV, np.dot(np.linalg.inv(cov_tot), diff_DV))
    return chi2


def compare_chi2(inp, env, beta, ra_halos, dec_halos, h):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    env: environment object
    beta: float, value of beta for CIB deprojection
    h: 1D numpy array in RING format containing map of halos
    y1: 1D numpy array in RING format containing reconstructed y map
    y2: 1D numpy array in RING format containing true y map
        (or y map with inflated CIB)
    ra_halos: ndarray containing ra of halos
    dec_halos: ndarray containing dec of halos
    h: 1D numpy array in RING format containing halo map

    RETURNS
    -------
    chi2_true: float, chi^2 value of <h,y_reconstructed> and <h,y_true>
    chi2_inflated: float, chi^2 value of <h,y_reconstructed> and <h,y_reconstructed_with_inflated_cib>
    '''
    y_true = hp.read_map(inp.tsz_map_file)
    y_true = hp.ud_grade(y_true, inp.nside)
    y_recon = setup_pyilc(inp, env, beta, inflated=False, suppress_printing=(not inp.debug))
    y_recon_inflated = setup_pyilc(inp, env, beta, inflated=True, suppress_printing=(not inp.debug))
    if inp.harmonic_space:
        chi2_true = compute_chi2_harmonic_space(inp, y_recon, y_true, h)
        chi2_inflated = compute_chi2_harmonic_space(inp, y_recon, y_recon_inflated, h)
    else:   
        chi2_true = compute_chi2_real_space(inp, y_recon, y_true, ra_halos, dec_halos, beta)
        chi2_inflated = compute_chi2_real_space(inp, y_recon, y_recon_inflated, ra_halos, dec_halos, beta)
    if inp.debug:
        print(f'chi2_true for beta={beta}: {chi2_true}', flush=True)
        print(f'chi2_inflated for beta={beta}: {chi2_inflated}', flush=True)
    plot_corr(inp, beta, y_true, y_recon, y_recon_inflated, h)
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
    function of *args, compare_chi2(inp, env, beta, ra_halos, dec_halos, h)
    '''
    return compare_chi2(*args)
