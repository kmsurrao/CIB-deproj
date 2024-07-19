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
from get_y_map import get_all_ymaps_star, setup_pyilc
from generate_maps import get_freq_maps
from utils import *


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
        h, ra_halos, dec_halos = halofile2map(inp)
    else:
        h, ra_halos, dec_halos = halodir2map(inp)
    print('got ra and dec of halos', flush=True)
    print('Getting maps at different frequencies...', flush=True)
    get_freq_maps(inp, diff_noise=False, no_cib=False)


    # Build standard ILC y-map (from maps with no CIB)
    print('Building standard ILC y-map from freq. maps without CIB...', flush=True)
    get_freq_maps(inp, diff_noise=False, no_cib=True)
    setup_pyilc(inp, env, 1.0, suppress_printing=True, inflated=False, \
                standard_ilc=True, no_cib=True)


    # Build y-maps with pyilc (deprojecting beta)
    print('Building y-maps...', flush=True)
    beta_arr = np.linspace(inp.beta_range[0], inp.beta_range[1], num=inp.num_beta_vals) 
    pool = mp.Pool(inp.num_parallel)
    inputs = [(inp, env, beta) for beta in beta_arr]
    _ = list(tqdm.tqdm(pool.imap(get_all_ymaps_star, inputs), total=len(beta_arr)))
    pool.close()
    

    # Compute covariance matrix
    mean_beta = beta_arr[len(beta_arr)//2]
    print(f'\nComputing covariance matrix using beta={mean_beta:0.3f}...', flush=True)
    inp.cov_hytrue = cov(inp, mean_beta, ra_halos, dec_halos, h, inflated=False) 
    inp.cov_hyinfl = cov(inp, mean_beta, ra_halos, dec_halos, h, inflated=True) 
    pickle.dump(inp.cov_hytrue, open(f'{inp.output_dir}/correlation_plots/cov_hytrue.p', 'wb'))
    pickle.dump(inp.cov_hyinfl, open(f'{inp.output_dir}/correlation_plots/cov_hyinfl.p', 'wb'))


    # Compute chi2 values
    print('Computing chi2 values...', flush=True)
    pool = mp.Pool(inp.num_parallel)
    inputs = [(inp, beta, ra_halos, dec_halos, h) for beta in beta_arr]
    results = list(tqdm.tqdm(pool.imap(compare_chi2_star, inputs), total=len(beta_arr)))
    pool.close()
    results = np.array(results, dtype=np.float32)
    chi2_true_arr = results[:,0]
    chi2_inflated_arr = results[:,1]
    print('\ngot chi2 values', flush=True)
    print('chi2_true_arr: ', chi2_true_arr, flush=True)
    print('chi2_inflated_arr: ', chi2_inflated_arr, flush=True)


    # save files and plot
    pickle.dump(beta_arr, open(f'{inp.output_dir}/beta_arr.p', 'wb'))
    pickle.dump(chi2_true_arr, open(f'{inp.output_dir}/chi2_true_arr.p', 'wb'))
    pickle.dump(chi2_inflated_arr, open(f'{inp.output_dir}/chi2_inflated_arr.p', 'wb'))
    plt.plot(beta_arr, chi2_true_arr, label='reconstructed vs. true')
    plt.plot(beta_arr, chi2_inflated_arr, label='reconstructed vs.\nreconstructed with CIB inflated', linestyle='dashed')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'${\chi}^2$')
    plt.legend()
    plt.yscale('log')
    plt.grid(which='both')
    plt.savefig(f'{inp.output_dir}/chi2.png')
    
    return

if __name__ == '__main__':
    main()