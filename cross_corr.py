import numpy as np
import healpy as hp
import treecorr
import os
from get_y_map import setup_pyilc
from utils import binned, cov
from plot_correlations import plot_corr_harmonic, plot_corr_real

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



def compute_chi2_real_space(inp, y1, y2, ra_halos, dec_halos, beta, include_covhy2=True):
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
    include_covhy2: Bool, whether to include covariance of h x y2 in chi2 computation

    RETURNS
    -------
    chi2: float, chi^2 value of <h,y1> and <h,y2>
    xi_hy1: 1D numpy array of length nrad containing correlation of halos with y1
    xi_hy2: 1D numpy array of length nrad containing correlation of halos with y2
    cov_hy1: 3D numpy array of shape (2,2,nrad) containing covariance 
        matrix of halos and y1
    cov_hy2: 3D numpy array of shape (2,2,nrad) containing covariance 
        matrix of halos and y2
    r_hy: 1D numpy array of length nrad containing x-axis point for plotting
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
    r_hy = np.exp(hy1.meanlogr)

    if include_covhy2:
        cov_tot = cov_hy1 + cov_hy2
    else:
        cov_tot = cov_hy1
    diff_DV = xi_hy1 - xi_hy2
    chi2 = np.dot(diff_DV, np.dot(np.linalg.inv(cov_tot), diff_DV))

    return chi2, xi_hy1, xi_hy2, cov_hy1, cov_hy2, r_hy


def compute_chi2_harmonic_space(inp, y1, y2, h, include_covhy2=True):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    y1: 1D numpy array in RING format containing reconstructed y map
    y2: 1D numpy array in RING format containing true y map
        (or y map with inflated CIB)
    h: 1D numpy array in RING format containing halo map
    include_covhy2: Bool, whether to include covariance of h x y2 in chi2 computation

    RETURNS
    -------
    chi2: float, chi^2 value of <h,y1> and <h,y2>
    hy1: 1D numpy array of length Nbins containing cross-spectrum of halos with y1
    hy2: 1D numpy array of length Nbins containing cross-spectrum of halos with y2
    y1y1: 1D numpy array of length Nbins containing auto-spectrum of y1
    y2y2: 1D numpy array of length Nbins containing auto-spectrum of y2
    hh: 1D numpy array of length Nbins containing auto-spectrum of halos
    '''
    hy1 = binned(inp, hp.anafast(h, y1, lmax=inp.ellmax))
    hy2 = binned(inp, hp.anafast(h, y2, lmax=inp.ellmax))
    hh = binned(inp, hp.anafast(h, lmax=inp.ellmax))
    y1y1 = binned(inp, hp.anafast(y1, lmax=inp.ellmax))
    y2y2 = binned(inp, hp.anafast(y2, lmax=inp.ellmax))
    cov_hy1 = cov(inp, np.array([[hh, hy1], [hy1, y1y1]]))
    cov_hy2 = cov(inp, np.array([[hh, hy2], [hy2, y2y2]]))
    if include_covhy2:
        cov_tot = cov_hy1 + cov_hy2
    else:
        cov_tot = cov_hy1
    diff_DV = hy1-hy2
    chi2 = np.dot(diff_DV, np.dot(np.linalg.inv(cov_tot), diff_DV))
    return chi2, hy1, hy2, y1y1, y2y2, hh


def compare_chi2(inp, env, beta, ra_halos, dec_halos, h):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    env: environment object
    beta: float, value of beta for CIB deprojection
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
        chi2_true, uninflxh, tszxh, uninfl_auto, tsz_auto, h_auto = compute_chi2_harmonic_space(inp, y_recon, y_true, h, include_covhy2=False)
        chi2_inflated, uninflxh, inflxh, uninfl_auto, infl_auto, h_auto = compute_chi2_harmonic_space(inp, y_recon, y_recon_inflated, h)
        plot_corr_harmonic(inp, beta, uninflxh, inflxh, tszxh, uninfl_auto, infl_auto, tsz_auto, h_auto)
    else:   
        chi2_true, uninflxh, tszxh, cov_hyuninfl, cov_hytrue, r_hy = compute_chi2_real_space(inp, y_recon, y_true, ra_halos, dec_halos, beta, include_covhy2=False)
        chi2_inflated, uninflxh, inflxh, cov_hyuninfl, cov_hyinfl, r_hy = compute_chi2_real_space(inp, y_recon, y_recon_inflated, ra_halos, dec_halos, beta)
        plot_corr_real(inp, beta, uninflxh, inflxh, tszxh, cov_hyuninfl, cov_hyinfl, cov_hytrue, r_hy)
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
    function of *args, compare_chi2(inp, env, beta, ra_halos, dec_halos, h)
    '''
    return compare_chi2(*args)
