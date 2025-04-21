import argparse
import os
import tqdm
import numpy as np
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt
import healpy as hp
from input import Info
from halo2map import halodir2map, halofile2map
from cross_corr import harmonic_space_cov, cov, compare_chi2_star
from get_y_map import get_all_ymaps, get_all_ymaps_star, setup_pyilc
from generate_maps import get_freq_maps
from harmonic_ilc import HILC_map
from beta_per_bin import get_all_1sigma_beta, predict_with_uncertainty
from utils import *
plt.rcParams.update({
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
    setup_output_dir(inp)
    
    # get map of halos and maps at each frequency (both with and without inflated CIB)
    if inp.halo_map is not None:
        h = hp.ud_grade(hp.read_map(inp.halo_map), inp.nside)
    elif inp.halo_catalog is not None:                                                  
        h = halofile2map(inp)
    else:
        h = halodir2map(inp)
    print('Got halo map', flush=True)
    print('Getting maps at different frequencies...', flush=True)
    get_freq_maps(inp, no_cib=False)

    # define SED of tSZ
    signal_sed = tsz_spectral_response(inp.frequencies, delta_bandpasses=inp.delta_passbands, inp=inp)


    # Build standard ILC y-map (from maps with no CIB)
    print('Building standard ILC y-map from freq. maps without CIB...', flush=True)
    standard_ILC_file = f'{inp.output_dir}/pyilc_outputs/uninflated/needletILCmap_component_tSZ.fits'
    if not os.path.isfile(standard_ILC_file):
        get_freq_maps(inp, no_cib=True)
        yopt = HILC_map(inp, 1.0, signal_sed, contam_sed=None, inflated=False, no_cib=True)
    else:
        yopt = hp.read_map(standard_ILC_file)

    # Build ILC y-map deprojecting fiducial beta
    print('Building y-map deprojecting fiducial beta...', flush=True)
    get_all_ymaps(inp, inp.beta_fid)

    # Get best h_nu
    print('Getting best h_nu...', flush=True)
    inp.h_vec = optimize_h(inp, max_iter=10, tol=1e-4)

    # Build ILC y-maps (deprojecting beta)
    print('Building all other y-maps...', flush=True)
    pool = mp.Pool(inp.num_parallel)
    inputs = [(inp, beta) for beta in inp.beta_arr]
    _ = list(tqdm.tqdm(pool.imap(get_all_ymaps_star, inputs), total=len(inputs)))
    pool.close()
    
    # Compute covariance matrix
    print(f'\nComputing covariance matrix using beta={inp.beta_fid:0.3f}...', flush=True)
    if os.path.isfile(f'{inp.output_dir}/correlation_plots/cov_hytrue.p'):
        inp.cov_hytrue = pickle.load(open(f'{inp.output_dir}/correlation_plots/cov_hytrue.p', 'rb'))
    else:   
        inp.cov_hytrue = cov(inp, inp.beta_fid, h, inflated=False) 
        pickle.dump(inp.cov_hytrue, open(f'{inp.output_dir}/correlation_plots/cov_hytrue.p', 'wb'))
    if os.path.isfile(f'{inp.output_dir}/correlation_plots/cov_hyinfl.p'):
        inp.cov_hyinfl = pickle.load(open(f'{inp.output_dir}/correlation_plots/cov_hyinfl.p', 'rb'))
    else:
        inp.cov_hyinfl = cov(inp, inp.beta_fid, h, inflated=True) 
        pickle.dump(inp.cov_hyinfl, open(f'{inp.output_dir}/correlation_plots/cov_hyinfl.p', 'wb'))


    # Compute chi2 values
    print('Computing chi2 values...', flush=True)
    chi2_true_file = f'{inp.output_dir}/pickle_files/chi2_true_arr.p'
    chi2_infl_file = f'{inp.output_dir}/pickle_files/chi2_inflated_arr.p'
    if os.path.isfile(chi2_true_file) and os.path.isfile(chi2_infl_file):
        chi2_true_arr = pickle.load(open(chi2_true_file, 'rb'))
        chi2_inflated_arr = pickle.load(open(chi2_infl_file, 'rb'))
    else:
        pool = mp.Pool(inp.num_parallel)
        inputs = [(inp, beta, h) for beta in inp.beta_arr]
        # results shape: (Nbetas, 2 for chi2_true chi2_infl, Nbins)
        results = list(tqdm.tqdm(pool.imap(compare_chi2_star, inputs), total=len(inp.beta_arr)))
        pool.close()
        results = np.array(results, dtype=np.float32)
        chi2_true_arr = results[:,0].T # shape (Nbins, Nbetas)
        chi2_inflated_arr = results[:,1].T # shape (Nbins, Nbetas)
        print('\nGot chi2 values for each ell bin and beta value', flush=True)
        pickle.dump(inp.beta_arr, open(f'{inp.output_dir}/pickle_files/beta_arr.p', 'wb'))
        pickle.dump(chi2_true_arr, open(chi2_true_file, 'wb'))
        pickle.dump(chi2_inflated_arr, open(chi2_infl_file, 'wb'))

    # Get mean ells in each bin
    ells = np.arange(inp.ellmax+1)
    Nbins = int(np.round((inp.ellmax-inp.ellmin+1)/inp.ells_per_bin))
    res = stats.binned_statistic(ells[inp.ellmin:], ells[inp.ellmin:], statistic='mean', bins=Nbins)
    inp.mean_ells = (res[1][:-1]+res[1][1:])/2
    inp.bin_edges = res[1]
    bin_number = res[2]


    # Fit best beta with 1sigma range for every ell bin, and save
    means_true, uppers_true, lowers_true, means_infl, uppers_infl, lowers_infl = get_all_1sigma_beta(inp, chi2_true_arr, chi2_inflated_arr)
    pickle.dump([means_true, uppers_true, lowers_true, means_infl, uppers_infl, lowers_infl], open(f'{inp.output_dir}/pickle_files/beta_points_per_ell.p', 'wb'))
    best_fit_true, fit_err_true, popt_true = predict_with_uncertainty(inp.mean_ells, means_true, lowers_true, uppers_true, deg=3)
    best_fit_infl, fit_err_infl, popt_infl = predict_with_uncertainty(inp.mean_ells, means_infl, lowers_infl, uppers_infl, deg=3)
    pickle.dump([best_fit_true, fit_err_true, popt_true, best_fit_infl, fit_err_infl, popt_infl], open(f'{inp.output_dir}/pickle_files/best_fits.p', 'wb'))
    

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
    plt.savefig(f'{inp.output_dir}/plots/beta_per_ell.pdf', bbox_inches='tight')


    # build final y-maps, deprojecting optimal beta for each ell bin separately
    print('Building final y-map, deprojecting optimal beta in each bin...', flush=True)
    for i in range(2):  
        pipeline_str = 'realistic' if i==0 else 'idealized'
        # popt = popt_infl if i==0 else popt_true
        # beta_vs_ell = model(ells, *popt)
        mean_betas = np.array(means_infl) if i==0 else np.array(means_true)
        bin_inds = bin_number.astype(int) - 1
        beta_vs_ell = np.zeros_like(ells, dtype=np.float32)
        beta_vs_ell[inp.ellmin:] = mean_betas[bin_inds] # below ellmin not used
        contam_sed = cib_spectral_response(inp.frequencies, delta_bandpasses=inp.delta_passbands, \
                                           inp=inp, beta=beta_vs_ell) # shape (Nfreqs, ellmax+1)
        fname = f"{inp.output_dir}/pyilc_outputs/final/needletILCmap_component_tSZ_deproject_CIB_{pipeline_str}.fits"
        if not os.path.isfile(fname):
            HILC_map(inp, None, signal_sed, contam_sed=contam_sed, inflated=False, no_cib=False, fname=fname)
    print('Saved final y-maps, built with both the realistic and idealized pipelines.', flush=True)


    # build map deprojecting first moment w.r.t. beta, for comparison
    print(f'Building y-map, deprojecting both beta and first moment, using fiducial beta={inp.beta_fid}', flush=True)
    y_beta_dbeta_file = f'{inp.output_dir}/pyilc_outputs/final/needletILCmap_component_tSZ_deproject_CIB_CIB_dbeta.fits'
    if not os.path.isfile(y_beta_dbeta_file):
        y_beta_dbeta = setup_pyilc(inp, env, inp.beta_fid, moment_deproj=True)
    else:
        y_beta_dbeta = hp.read_map(y_beta_dbeta_file)


    # plots comparing final y-map to y-map with dbeta deprojection
    y_beta = hp.read_map(f"{inp.output_dir}/pyilc_outputs/final/needletILCmap_component_tSZ_deproject_CIB_realistic.fits")
    ytrue = hp.ud_grade(hp.read_map(inp.tsz_map_file), inp.nside)
    hxy_beta = binned(inp, hp.anafast(h, y_beta, lmax=inp.ellmax))
    hxy_beta_dbeta = binned(inp, hp.anafast(h, y_beta_dbeta, lmax=inp.ellmax))
    y_beta_dbetaxy_beta_dbeta = binned(inp, hp.anafast(y_beta_dbeta, lmax=inp.ellmax))
    y_betaxy_beta = binned(inp, hp.anafast(y_beta, lmax=inp.ellmax))
    ytruexytrue = binned(inp, hp.anafast(ytrue, lmax=inp.ellmax))
    y_betaxytrue = binned(inp, hp.anafast(y_beta, ytrue, lmax=inp.ellmax))
    y_beta_dbetaxytrue = binned(inp, hp.anafast(y_beta_dbeta, ytrue, lmax=inp.ellmax))
    yoptxyopt = binned(inp, hp.anafast(yopt, lmax=inp.ellmax))
    yoptxytrue = binned(inp, hp.anafast(yopt, ytrue, lmax=inp.ellmax))
    yoptxh = binned(inp, hp.anafast(yopt, h, lmax=inp.ellmax))
    hxytrue = binned(inp, hp.anafast(h, ytrue, lmax=inp.ellmax))
    mean_ells = inp.mean_ells
    to_dl = mean_ells*(mean_ells+1)/2/np.pi
    fig, axs = plt.subplots(2,2, figsize=(8,8))
    axs = axs.flatten()
    for n, ax in enumerate(axs):
        plt.axes(ax)
        if n==0:
            hxy_beta_cov = harmonic_space_cov(inp, y_beta, np.zeros_like(y_beta), h)
            plt.errorbar(mean_ells, to_dl*hxy_beta, yerr=to_dl*np.sqrt(np.diag(hxy_beta_cov)), label=r'$y^{\beta} \times h$')
            plt.plot(mean_ells, to_dl*hxy_beta_dbeta, label=r'$y^{\beta + d\beta} \times h$')
            plt.plot(mean_ells, to_dl*yoptxh, label=r'$y^{\rm opt} \times h$', linestyle='dashed')
            plt.plot(mean_ells, to_dl*hxytrue, label=r'$y_{\rm true} \times h$', linestyle='dashed')
            plt.legend(fontsize=12)
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$\ell(\ell+1)C_\ell^{hy}/(2\pi)$')
            plt.grid()
            plt.xlim(mean_ells[0], mean_ells[-1])
            plt.title(r'Cross-spectra of ILC $y$-map with halos', fontsize=14)
        elif n==1:
            plt.plot(mean_ells, to_dl*y_betaxy_beta, label=r'$y^{\beta} \times y^{\beta}$')
            plt.plot(mean_ells, to_dl*y_beta_dbetaxy_beta_dbeta, label=r'$y^{\beta + d\beta} \times y^{\beta + d\beta}$')
            plt.plot(mean_ells, to_dl*yoptxyopt, label=r'$y^{\rm opt} \times y^{\rm opt}$', linestyle='dashed')
            plt.plot(mean_ells, to_dl*ytruexytrue, label=r'$y_{\rm true} \times y_{\rm true}$', linestyle='dashed')
            plt.legend(fontsize=12)
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$\ell(\ell+1)C_\ell^{yy}/(2\pi)$')
            plt.grid()
            plt.xlim(mean_ells[0], mean_ells[-1])
            plt.yscale('log')
            plt.title(r'Auto-spectra of ILC $y$-maps', fontsize=14)      
        elif n==2:
            plt.plot(mean_ells, to_dl*y_betaxytrue, label=r'$y^{\beta} \times y_{\mathrm{true}}$')
            plt.plot(mean_ells, to_dl*y_beta_dbetaxytrue, label=r'$y^{\beta+d\beta} \times y_{\mathrm{true}}$')
            plt.plot(mean_ells, to_dl*yoptxytrue, label=r'$y^{\rm opt} \times y_{\mathrm{true}}$', linestyle='dashed')
            plt.legend(fontsize=12)
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$\ell(\ell+1)C_\ell^{y,y_{\mathrm{true}}}/(2\pi)$')
            plt.grid(which='both')
            plt.xlim(mean_ells[0], mean_ells[-1])
            plt.title('Cross-spectra of ILC and true $y$-maps', fontsize=14)
        elif n==3:
            plt.plot(mean_ells,y_betaxytrue/np.sqrt(ytruexytrue*y_betaxy_beta), label=r'$y^{\beta} \times y_{\mathrm{true}}$')
            plt.plot(mean_ells,y_beta_dbetaxytrue/np.sqrt(ytruexytrue*y_beta_dbetaxy_beta_dbeta), label=r'$y^{\beta+d\beta} \times y_{\mathrm{true}}$')
            plt.plot(mean_ells,yoptxytrue/np.sqrt(ytruexytrue*yoptxyopt), label=r'$y^{\rm opt} \times y^{\rm opt}$', linestyle='dashed')
            plt.legend(fontsize=12)
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$r_\ell^{y,y_{\mathrm{true}}}$')
            plt.grid()
            plt.xlim(mean_ells[0], mean_ells[-1])
            plt.title('Corr. coefficients of ILC and true $y$-maps', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{inp.output_dir}/plots/dbeta.pdf', bbox_inches='tight')

    print('Completed.', flush=True)



    return 

if __name__ == '__main__':
    main()