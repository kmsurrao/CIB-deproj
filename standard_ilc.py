import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from input import Info
import healpy as hp
from get_y_map import setup_pyilc
from halo2map import halodir2map, halofile2map
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
    setup_output_dir(inp, env, standard_ilc=True)
    
    # get maps at each frequency (both with and without inflated CIB)
    print('Getting maps at different frequencies...', flush=True)
    get_freq_maps(inp)

    # run pyilc
    print('Running pyilc...', flush=True)  
    y_uninflated_cib = setup_pyilc(inp, env, 1.0, 1.0, suppress_printing=(not inp.debug), inflated=False, standard_ilc=True)
    y_inflated_cib = setup_pyilc(inp, env, 1.0, 1.0, suppress_printing=(not inp.debug), inflated=True, standard_ilc=True) 
    y_true = hp.ud_grade(hp.read_map(inp.tsz_map_file), inp.nside)

    # get power spectra and plot
    y_uninfl_auto = hp.anafast(y_uninflated_cib, lmax=inp.ellmax)
    y_infl_auto = hp.anafast(y_inflated_cib, lmax=inp.ellmax)
    y_true_auto = hp.anafast(y_true, lmax=inp.ellmax)
    ells = np.arange(inp.ellmax+1)
    to_dl = ells*(ells+1)/2/np.pi
    start = 2
    plt.plot(ells[start:], (to_dl*y_uninfl_auto)[start:], label='y recon.')  
    plt.plot(ells[start:], (to_dl*y_infl_auto)[start:], label='y recon. (inflated CIB)')    
    plt.plot(ells[start:], (to_dl*y_true_auto)[start:], label='y true')
    plt.legend()
    plt.yscale('log')
    plt.grid(which='both')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(\ell+1)C_\ell/(2\pi)$')  
    plt.savefig(f'{inp.output_dir}/standard_ilc_auto_spectra.png')  

    # compute and plot correlation coefficients
    yuninfl_x_ytrue = hp.anafast(y_uninflated_cib, y_true, lmax=inp.ellmax)/np.sqrt(y_uninfl_auto*y_true_auto)   
    yinfl_x_ytrue = hp.anafast(y_inflated_cib, y_true, lmax=inp.ellmax)/np.sqrt(y_infl_auto*y_true_auto)   
    plt.clf()
    plt.plot(ells[start:], yuninfl_x_ytrue[start:], label='y recon. x y true')   
    plt.plot(ells[start:], yinfl_x_ytrue[start:], label='y recon. (inflated CIB) x y true')
    plt.legend()
    plt.grid()    
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$r_{\ell}$')  
    plt.savefig(f'{inp.output_dir}/standard_ilc_corr_coeff.png') 

    # correlate with halos
    if inp.halo_catalog is not None:                                                  
        h, ra_halos, dec_halos = halofile2map(inp)
    else:
        h, ra_halos, dec_halos = halodir2map(inp)
    print('got ra and dec of halos', flush=True)
    h_auto = hp.anafast(h, lmax=inp.ellmax)
    yuninfl_x_h = hp.anafast(y_uninflated_cib, h, lmax=inp.ellmax)/np.sqrt(y_uninfl_auto*h_auto)    
    yinfl_x_h = hp.anafast(y_inflated_cib, h, lmax=inp.ellmax)/np.sqrt(y_infl_auto*h_auto)    
    ytrue_x_h = hp.anafast(y_true, h, lmax=inp.ellmax)/np.sqrt(y_true_auto*h_auto)
    plt.clf()
    plt.plot(ells[start:], yuninfl_x_h[start:], label='y recon. x h')   
    plt.plot(ells[start:], yinfl_x_h[start:], label='y recon. (inflated CIB) x h')
    plt.plot(ells[start:], ytrue_x_h[start:], label='y true x h') 
    plt.legend()
    plt.grid()    
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$r_{\ell}$')  
    plt.savefig(f'{inp.output_dir}/standard_ilc_corr_coeff_h.png')                                                                                     
    
    return

if __name__ == '__main__':
    main()