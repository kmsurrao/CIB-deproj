import numpy as np
import matplotlib.pyplot as plt
import pickle
import healpy as hp
from scipy import stats
from utils import binned
import os


def plot_corr_harmonic(inp, beta, hy_true, hy_infl):
    '''
    Plots harmonic space correlations (power spectra) given the spectra

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    beta: float, value of beta for CIB deprojection
    hy_true: 1D numpy array of length Nbins containing cross-spectrum of halos with 
        (y_reconstructed - y_true)
    hy_infl: 1D numpy array of length Nbins containing cross-spectrum of halos with 
        (y_reconstructed - y_reconstructed_with_inflated_cib)

    RETURNS
    -------
    None (saves plots in {output_dir}/plot_correlations)
    '''
    if os.path.isfile(f'{inp.output_dir}/correlation_plots/beta_{beta:0.3f}.p'):
        return
    to_dl = inp.mean_ells*(inp.mean_ells+1)/2/np.pi
    plt.clf()
    plt.errorbar(inp.mean_ells, to_dl*hy_true, label=r'$C_\ell^{h, (\mathrm{yrecon.-ytrue})}$', yerr=to_dl*np.sqrt(np.diagonal(inp.cov_hytrue)), linestyle='solid')
    plt.errorbar(inp.mean_ells, to_dl*hy_infl, label=r'$C_\ell^{h, (\mathrm{yrecon.-yreconinflcib})}$', yerr=to_dl*np.sqrt(np.diagonal(inp.cov_hyinfl)), linestyle='dashed')
    plt.grid()
    plt.ylabel(r'$\ell(\ell+1)C_\ell /(2\pi)$ [$\mu \mathrm{K}^2$]')
    plt.xlabel(r'$\ell$')
    plt.legend()
    plt.title(r'$h \times y$, $\beta=$' + f'{beta:0.3f}')
    plt.savefig(f'{inp.output_dir}/correlation_plots/beta_{beta:0.3f}.png')

    # save mean_ells, spectra, and errors to plot later
    to_save = [inp.mean_ells, to_dl, hy_true, hy_infl]
    pickle.dump(to_save, open(f'{inp.output_dir}/correlation_plots/beta_{beta:0.3f}.p', 'wb'))

    return


def plot_corr_harmonic_from_map(inp, beta, y_true, y_recon, y_recon_inflated, h):
    '''
    Plots harmonic space correlations (power spectra) given maps

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    beta: float, value of beta for CIB deprojection
    y_true: 1D numpy array in healpix RING format containing true y map
    y_recon: 1D numpy array in healpix RING format containing reconstructed y map
    y_recon_inflated: 1D numpy array in healpix RING format containing reconstructed
        y map when the CIB is inflated
    h: 1D numpy array in healpix RING format containing halo map

    RETURNS
    -------
    None (saves plots in {output_dir}/plot_correlations)
    '''

    # compute all the spectra
    hy_true = binned(inp, hp.anafast(h, y_recon-y_true, lmax=inp.ellmax))
    hy_infl = binned(inp, hp.anafast(h, y_recon-y_recon_inflated, lmax=inp.ellmax))
    
    # plot and save
    plot_corr_harmonic(inp, beta, hy_true, hy_infl)
    
    return
