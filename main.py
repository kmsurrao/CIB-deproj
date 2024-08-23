import argparse
import os
import tqdm
import numpy as np
import multiprocessing as mp
import pickle
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from input import Info
from halo2map import halodir2map, halofile2map
from cross_corr import cov, compare_chi2_star
from get_y_map import get_all_ymaps_star, setup_pyilc
from generate_maps import get_freq_maps
from utils import *


def main():

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Optimal SED for CIB deprojection.")
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
        h, ra_halos, dec_halos = halofile2map(inp)
    else:
        h, ra_halos, dec_halos = halodir2map(inp)
    print('got ra and dec of halos', flush=True)
    print('Getting maps at different frequencies...', flush=True)
    get_freq_maps(inp, diff_noise=False, no_cib=False)


    # Build standard ILC y-map (from maps with no CIB)
    print('Building standard ILC y-map from freq. maps without CIB...', flush=True)
    standard_ILC_file = f'{inp.output_dir}/pyilc_outputs/uninflated/needletILCmap_component_tSZ.fits'
    if not os.path.isfile(standard_ILC_file):
        get_freq_maps(inp, diff_noise=False, no_cib=True)
        setup_pyilc(inp, env, 1.0, 1.0, suppress_printing=True, inflated=False, \
                    standard_ilc=True, no_cib=True)


    # Build y-maps with pyilc (deprojecting beta)
    print('Building y-maps...', flush=True)
    beta_arr = np.linspace(inp.beta_range[0], inp.beta_range[1], num=inp.num_beta_vals, endpoint=False)
    T_arr = np.linspace(inp.T_range[0], inp.T_range[1], num=inp.num_T_vals, endpoint=False)  
    pool = mp.Pool(inp.num_parallel)
    inputs = [(inp, env, beta, T) for beta, T in list(itertools.product(beta_arr,T_arr))]
    _ = list(tqdm.tqdm(pool.imap(get_all_ymaps_star, inputs), total=len(beta_arr)))
    pool.close()
    

    # Compute covariance matrix
    mean_beta = beta_arr[len(beta_arr)//2]
    mean_T = T_arr[len(T_arr)//2]
    print(f'\nComputing covariance matrix using beta={mean_beta:0.3f}, T={mean_T:0.3f}...', flush=True)
    inp.cov_hytrue = cov(inp, mean_beta, mean_T, ra_halos, dec_halos, h, inflated=False) 
    inp.cov_hyinfl = cov(inp, mean_beta, mean_T, ra_halos, dec_halos, h, inflated=True) 
    pickle.dump(inp.cov_hytrue, open(f'{inp.output_dir}/correlation_plots/cov_hytrue.p', 'wb'))
    pickle.dump(inp.cov_hyinfl, open(f'{inp.output_dir}/correlation_plots/cov_hyinfl.p', 'wb'))


    # Compute chi2 values
    print('Computing chi2 values...', flush=True)
    pool = mp.Pool(inp.num_parallel)
    inputs = [(inp, beta, T, ra_halos, dec_halos, h) for beta, T in list(itertools.product(beta_arr,T_arr))]
    results = list(tqdm.tqdm(pool.imap(compare_chi2_star, inputs), total=len(beta_arr)*len(T_arr)))
    pool.close()
    results = np.array(results, dtype=np.float32)
    chi2_true_arr = np.array(results[:,0]).reshape((inp.num_beta_vals, inp.num_T_vals))
    chi2_inflated_arr = np.array(results[:,1]).reshape((inp.num_beta_vals, inp.num_T_vals))
    print('\ngot chi2 values', flush=True)
    print('chi2_true_arr: ', chi2_true_arr, flush=True)
    print('chi2_inflated_arr: ', chi2_inflated_arr, flush=True)


    # save files and plot
    pickle.dump(beta_arr, open(f'{inp.output_dir}/beta_arr.p', 'wb'))
    pickle.dump(T_arr, open(f'{inp.output_dir}/T_arr.p', 'wb'))
    pickle.dump(chi2_true_arr, open(f'{inp.output_dir}/chi2_true_arr.p', 'wb'))
    pickle.dump(chi2_inflated_arr, open(f'{inp.output_dir}/chi2_inflated_arr.p', 'wb'))
    
    for a, arr in enumerate([chi2_true_arr, chi2_inflated_arr]):
        extent = inp.T_range[0], inp.T_range[1], inp.beta_range[0], inp.beta_range[1]
        plt.clf()
        plt.imshow(arr, extent=extent, aspect="auto", norm=LogNorm())
        cbar = plt.colorbar()
        cbar.set_label(r'${\chi}^2$', rotation=270)
        plt.xlabel(r'$T$')
        plt.ylabel(r'$\beta$')
        fname = 'chi2_true.png' if a==0 else 'chi2_inflated.png'
        plt.savefig(f'{inp.output_dir}/{fname}')
    
    return

if __name__ == '__main__':
    main()