
import argparse
import os
import numpy as np
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt
from input import Info
from halo2map import convert_halo2map
from cross_corr import compare_chi2
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
    h, ra_halos, dec_halos = convert_halo2map(inp)
    get_freq_maps(inp)

    # run main computation
    beta_arr = np.arange(inp.beta_range[0], inp.beta_range[1], num=inp.num_beta_vals)
    pool = mp.Pool(inp.num_parallel)
    results = pool.starmap(compare_chi2, [(inp, env, beta, ra_halos, dec_halos) for beta in beta_arr])
    pool.close()
    results = np.array(results, dtype=np.float32)
    chi2_true_arr = results[:,0]
    chi2_inflated_arr = results[:,1]

    # save files and plot
    pickle.dump(beta_arr, open(f'{inp.output_dir}/beta_arr.p', 'wb'))
    pickle.dump(chi2_true_arr, open(f'{inp.output_dir}/chi2_true_arr.p', 'wb'))
    pickle.dump(chi2_inflated_arr, open(f'{inp.output_dir}/chi2_inflated_arr.p', 'wb'))
    plt.plot(beta_arr, chi2_true_arr, label='reconstructed vs. true')
    plt.plot(beta_arr, chi2_inflated_arr, label='reconstructed vs.\nreconstructed with CIB inflated', linestyle='dashed')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'${\chi}^2$')
    plt.legend()
    plt.savefig(f'{inp.output_dir}/chi2.png')
    
    return

if __name__ == '__main__':
    main()