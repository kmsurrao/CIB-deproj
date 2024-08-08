import numpy as np
import healpy as hp
import treecorr
import os
from utils import binned
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


def real_space_cov(inp, y1, y2, ra_halos, dec_halos, beta):
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
    cov_hy: (nrad, nrad) ndarray containing covariance matrix of halos and (y1-y2)
    '''
    ra_hp, dec_hp = hp.pix2ang(inp.nside, ipix=np.arange(hp.nside2npix(inp.nside)),lonlat=True)
    
    save_dir = f'{inp.output_dir}/results_test/'
    save_filename_jk_obj = f'jk_obj_test_beta{beta:.3f}.pkl'

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
    

    y_cat = treecorr.Catalog(ra=ra_hp, dec=dec_hp, k=y1-y2, ra_units='degrees', dec_units='degrees')
    hy = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin', verbose=0, num_threads=nthreads, bin_slop=bin_slop, var_method='jackknife')
    hy.process(datapoint_cat, y_cat)
    cov_hy = hy.cov
    return cov_hy


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


def cov(inp, beta, ra_halos, dec_halos, h, inflated=False):
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
        infl_str = 'inflated_realistic' if inp.realistic else 'inflated'
        y2 = hp.read_map(f"{inp.output_dir}/pyilc_outputs/beta_{beta:.3f}_{infl_str}/needletILCmap_component_tSZ_deproject_CIB.fits")
    else:
        y2 = hp.read_map(f"{inp.output_dir}/pyilc_outputs/uninflated/needletILCmap_component_tSZ.fits")
        y2 = hp.ud_grade(y2, inp.nside)
    if inp.harmonic_space:
        cov_hy = harmonic_space_cov(inp, y1, y2, h)
    else: 
        cov_hy = real_space_cov(inp, y1, y2, ra_halos, dec_halos, beta)
    return cov_hy


def compute_chi2_real_space(inp, y1, y2, ra_halos, dec_halos, beta, cov):
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
    cov: (Nrad, Nrad) ndarray containing covariance matrix of halos and (y1-y2)

    RETURNS
    -------
    chi2: float, chi^2 value of <h,y1> and <h,y2>
    xi_hy: 1D numpy array of length nrad containing correlation of halos with (y1-y2)
    r_hy: 1D numpy array of length nrad containing x-axis point for plotting
    '''
    rand_ra, rand_dec = randsphere(5*len(ra_halos), ra_range=[0,360], dec_range=[-90,90])
    ra_hp, dec_hp = hp.pix2ang(inp.nside, ipix=np.arange(hp.nside2npix(inp.nside)),lonlat=True)
    
    save_dir = f'{inp.output_dir}/results_test/'
    save_filename_jk_obj = f'jk_obj_test_beta{beta:.3f}.pkl'

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
    
    # Create random catalog and catalog for (y1-y2)
    rand_cat = treecorr.Catalog(ra=rand_ra, dec=rand_dec, ra_units='degrees', dec_units='degrees', patch_centers=save_dir + save_filename_jk_obj)
    ydiff_cat = treecorr.Catalog(ra=ra_hp, dec=dec_hp, k=(y1-y2), ra_units='degrees', dec_units='degrees')

    # create the correlation objects:
    hydiff = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin', verbose=0, num_threads=nthreads, bin_slop=bin_slop, var_method='jackknife')
    ry = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin', verbose=0, num_threads=nthreads, bin_slop=bin_slop, var_method='jackknife')

    hydiff.process(datapoint_cat, ydiff_cat)
    ry.process(rand_cat, ydiff_cat)

    hydiff.calculateXi(rk=ry)
    xi_hy = hydiff.xi
    r_hy = np.exp(hydiff.meanlogr)

    chi2 = np.dot(xi_hy, np.dot(np.linalg.inv(cov), xi_hy))

    return chi2, xi_hy, r_hy


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
    chi2: float, chi^2 value of <h,y1> and <h,y2>
    hydiff: 1D numpy array of length Nbins containing cross-spectrum of halos with (y1-y2)
    ''' 
    hydiff = binned(inp, hp.anafast(h, y1-y2, lmax=inp.ellmax))
    chi2 = np.dot(hydiff, np.dot(np.linalg.inv(cov), hydiff))
    return chi2, hydiff


def compare_chi2(inp, beta, ra_halos, dec_halos, h):
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
    infl_str = 'inflated_realistic' if inp.realistic else 'inflated'
    y_recon_inflated = hp.read_map(f"{inp.output_dir}/pyilc_outputs/beta_{beta:.3f}_{infl_str}/needletILCmap_component_tSZ_deproject_CIB.fits")
    
    if inp.harmonic_space:
        chi2_true, hy_true = compute_chi2_harmonic_space(inp, y_recon, y_true, h, inp.cov_hytrue)
        chi2_inflated, hy_infl = compute_chi2_harmonic_space(inp, y_recon, y_recon_inflated, h, inp.cov_hyinfl)
        plot_corr_harmonic(inp, beta, hy_true, hy_infl)
    else:   
        chi2_true, hy_true, r_hy = compute_chi2_real_space(inp, y_recon, y_true, ra_halos, dec_halos, beta, inp.cov_hytrue)
        chi2_inflated, hy_infl, r_hy = compute_chi2_real_space(inp, y_recon, y_recon_inflated, ra_halos, dec_halos, beta, inp.cov_hyinfl)
        plot_corr_real(inp, beta, hy_true, hy_infl, r_hy)
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
    function of *args, compare_chi2(inp, beta, ra_halos, dec_halos, h)
    '''
    return compare_chi2(*args)
