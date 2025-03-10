import argparse
import os
import tqdm
import numpy as np
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt
from input import Info
from halo2map import halodir2map, halofile2map
from cross_corr import cov, compare_chi2_star
from get_y_map import get_all_ymaps_star
from generate_maps import get_freq_maps
from harmonic_ilc import HILC_map
from beta_per_bin import get_all_1sigma_beta, predict_with_uncertainty
from utils import *
plt.rcParams.update({
     'text.usetex': True,
     'font.family': 'serif',
     'font.sans-serif': ['Computer Modern'],
     'font.size'   : 20})
plt.rc_context({'axes.autolimit_mode': 'round_numbers'})


def main():

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Optimal beta value for CIB deprojection.")
    parser.add_argument("--config", default="example.yaml")
    args = parser.parse_args()
    input_file = args.config

    # read in the input file and set up relevant info object
    inp = Info(input_file)

    # set up output directory
    env = os.environ.copy()
    setup_output_dir(inp, env)
    
    # get map of halos and maps at each frequency (both with and without inflated CIB)
    if inp.halo_catalog is not None:                                                  
        h = halofile2map(inp)
    else:
        h = halodir2map(inp)
    print('got halo map', flush=True)
    print('Getting maps at different frequencies...', flush=True)
    get_freq_maps(inp, no_cib=False)


    # Build standard ILC y-map (from maps with no CIB)
    print('Building standard ILC y-map from freq. maps without CIB...', flush=True)
    standard_ILC_file = f'{inp.output_dir}/pyilc_outputs/uninflated/needletILCmap_component_tSZ.fits'
    if not os.path.isfile(standard_ILC_file):
        get_freq_maps(inp, no_cib=True)
        delta_bandpasses = False if inp.cib_decorr else True
        signal_sed = tsz_spectral_response(inp.frequencies, delta_bandpasses=delta_bandpasses, inp=inp)
        HILC_map(inp, 1.0, signal_sed, contam_sed=None, inflated=False, no_cib=True)


    # Build ILC y-maps (deprojecting beta)
    print('Building y-maps...', flush=True)
    pool = mp.Pool(inp.num_parallel)
    inputs = [(inp, beta) for beta in inp.beta_arr]
    _ = list(tqdm.tqdm(pool.imap(get_all_ymaps_star, inputs), total=len(inp.beta_arr)))
    pool.close()
    

    # Compute covariance matrix
    mean_beta = inp.beta_arr[len(inp.beta_arr)//2]
    print(f'\nComputing covariance matrix using beta={mean_beta:0.3f}...', flush=True)
    inp.cov_hytrue = cov(inp, mean_beta, h, inflated=False) 
    inp.cov_hyinfl = cov(inp, mean_beta, h, inflated=True) 
    pickle.dump(inp.cov_hytrue, open(f'{inp.output_dir}/correlation_plots/cov_hytrue.p', 'wb'))
    pickle.dump(inp.cov_hyinfl, open(f'{inp.output_dir}/correlation_plots/cov_hyinfl.p', 'wb'))


    # Compute chi2 values
    print('Computing chi2 values...', flush=True)
    pool = mp.Pool(inp.num_parallel)
    inputs = [(inp, beta, h) for beta in inp.beta_arr]
    # results shape: (Nbetas, 2 for chi2_true chi2_infl, Nbins)
    results = list(tqdm.tqdm(pool.imap(compare_chi2_star, inputs), total=len(inp.beta_arr)))
    pool.close()
    results = np.array(results, dtype=np.float32)
    chi2_true_arr = results[:,0].T # shape (Nbins, Nbetas)
    chi2_inflated_arr = results[:,1].T # shape (Nbins, Nbetas)
    print('\ngot chi2 values for each ell bin and beta value', flush=True)
    pickle.dump(inp.beta_arr, open(f'{inp.output_dir}/beta_arr.p', 'wb'))
    pickle.dump(chi2_true_arr, open(f'{inp.output_dir}/chi2_true_arr.p', 'wb'))
    pickle.dump(chi2_inflated_arr, open(f'{inp.output_dir}/chi2_inflated_arr.p', 'wb'))


    # Fit best beta with 1sigma range for every ell bin, and save
    means_true, uppers_true, lowers_true, means_infl, uppers_infl, lowers_infl = get_all_1sigma_beta(inp, chi2_true_arr, chi2_inflated_arr)
    pickle.dump([means_true, uppers_true, lowers_true, means_infl, uppers_infl, lowers_infl], open(f'{inp.output_dir}/beta_points_per_ell.p', 'wb'))
    best_fit_true, fit_err_true = predict_with_uncertainty(inp.mean_ells, means_true, lowers_true, uppers_true, deg=3)
    best_fit_infl, fit_err_infl = predict_with_uncertainty(inp.mean_ells, means_infl, lowers_infl, uppers_infl, deg=3)
    pickle.dump([best_fit_true, fit_err_true, best_fit_infl, fit_err_infl], open(f'{inp.output_dir}/best_fits.p', 'wb'))
    

    # Plot beta with error bars and best fit with error at each ell
    plt.errorbar(inp.mean_ells, means_true, yerr=[lowers_true, uppers_true], fmt='o', label='Idealized (Points)', color='royalblue', zorder=1)
    plt.errorbar(inp.mean_ells, means_infl, yerr=[lowers_infl, uppers_infl], fmt='^', label='Realistic (Points)', color='darkorange', zorder=1)
    plt.plot(inp.mean_ells, best_fit_true, label="Idealized (Fit)", zorder=2)
    plt.fill_between(inp.mean_ells, best_fit_true - fit_err_true, best_fit_true + fit_err_true, alpha=0.3, zorder=2)
    plt.plot(inp.mean_ells, best_fit_infl, label="Realistic (Fit)", zorder=2)
    plt.fill_between(inp.mean_ells, best_fit_infl - fit_err_infl, best_fit_infl + fit_err_infl, alpha=0.3, zorder=2)
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$\beta$")
    plt.legend(fontsize=12)
    plt.xlim(inp.ellmin, inp.ellmax)
    plt.title(r'$\alpha=$'+f'{inp.alpha:.1f}', fontsize=18)
    plt.grid()
    plt.savefig(f'{inp.output_dir}/beta_per_ell.pdf', bbox_inches='tight')


    # build final y-map, deprojecting optimal beta for each ell bin separately
    contam_sed = np.array([cib_spectral_response(inp.frequencies, delta_bandpasses=delta_bandpasses, \
                                    inp=inp, beta=beta) for beta in best_fit_infl]).T # shape (Nfreqs, Nbins)
    fname = f"{inp.output_dir}/pyilc_outputs/final/needletILCmap_component_tSZ_deproject_CIB.fits"
    HILC_map(inp, None, signal_sed, contam_sed=contam_sed, inflated=False, no_cib=False, fname=fname)
    
    return

if __name__ == '__main__':
    main()