import numpy as np
import matplotlib.pyplot as plt
import pickle
import healpy as hp
from scipy import stats
from utils import binned, cov


def plot_corr_harmonic(inp, beta, hy_true, hy_infl, yy_true, yy_infl, hh):
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
    yy_true: 1D numpy array of length Nbins containing auto-spectrum of 
        (y_reconstructed - y_true)
    yy_infl: 1D numpy array of length Nbins containing auto-spectrum of 
        (y_reconstructed - y_reconstructed_with_inflated_cib)
    hh: 1D numpy array of length Nbins containing auto-spectrum of halo map

    RETURNS
    -------
    None (saves plots in {output_dir}/plot_correlations)
    '''
    # compute covariances
    cov_hytrue = cov(inp, np.array([[yy_true, hy_true], [hy_true, hh]]))
    cov_hyinfl = cov(inp, np.array([[yy_infl, hy_infl], [hy_infl, hh]]))

    # plot
    ells = np.arange(inp.ellmax+1)
    Nbins = inp.ellmax//inp.ells_per_bin
    res = stats.binned_statistic(ells[2:], ells[2:], statistic='mean', bins=Nbins)
    mean_ells = (res[1][:-1]+res[1][1:])/2
    to_dl = mean_ells*(mean_ells+1)/2/np.pi
    plt.clf()
    plt.errorbar(mean_ells, to_dl*hy_true, label=r'$C_\ell^{h, (\mathrm{yrecon.-ytrue})}$', yerr=to_dl*np.sqrt(np.diagonal(cov_hytrue)), linestyle='solid')
    plt.errorbar(mean_ells, to_dl*hy_infl, label=r'$C_\ell^{h, (\mathrm{yrecon.-yrecon_infl_cib})}$', yerr=to_dl*np.sqrt(np.diagonal(cov_hyinfl)), linestyle='dashed')
    plt.grid()
    plt.ylabel(r'$\ell(\ell+1)C_\ell /(2\pi)$ [$\mu \mathrm{K}^2$]')
    plt.xlabel(r'$\ell$')
    plt.legend()
    plt.title(r'$h \times y$, $\beta=$' + f'{beta:0.3f}')
    plt.savefig(f'{inp.output_dir}/correlation_plots/beta_{beta:0.3f}.png')

    # save mean_ells, spectra, and errors to plot later
    to_save = [mean_ells, to_dl, hy_true, hy_infl, yy_true, yy_infl, hh]
    pickle.dump(to_save, open(f'{inp.output_dir}/correlation_plots/beta_{beta:0.3f}.p', 'wb'))

    return


def plot_corr_real(inp, beta, hy_true, hy_infl, cov_hytrue, cov_hyinfl, r_hy):
    '''
    Plots real space correlations

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    beta: float, value of beta for CIB deprojection
    hy_true: 1D numpy array of length nrad containing correlation of halos with 
        (reconstructed y - true y)
    hy_infl: 1D numpy array of length nrad containing correlation of halos with 
        (reconstructed y - reconstructed y with inflated CIB)
    cov_hytrue: 2D numpy array of shape (nrad, nrad) containing covariance 
        matrix of halos and (reconstructed y - true y)
    cov_hyinfl: 2D numpy array of shape (nrad, nrad) containing covariance 
        matrix of halos and (reconstructed y - reconstructed y with inflated CIB)
    r_hy: 1D numpy array of length nrad containing x-axis point for plotting

    RETURNS
    -------
    None (saves plots in {output_dir}/plot_correlations)
    '''

    # plot
    plt.clf()
    plt.errorbar(r_hy, hy_true, yerr=np.sqrt(np.diag(cov_hytrue)), ls='', marker='v', ms=3.0, label='h x (y recon. - y true)')
    plt.errorbar(1.02*r_hy, hy_infl, yerr=np.sqrt(np.diag(cov_hyinfl)), ls='', marker='o', ms=3.0, label='h x (y recon. - y recon. (CIB inflated))')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f'{inp.output_dir}/correlation_plots/beta_{beta:0.3f}.png')

    # save spectra and errors to plot later
    to_save = [hy_true, hy_infl, cov_hytrue, cov_hyinfl, r_hy]
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
    yy_true = binned(inp, hp.anafast(y_recon-y_true, lmax=inp.ellmax))
    yy_infl = binned(inp, hp.anafast(y_recon-y_recon_inflated, lmax=inp.ellmax))
    hh = binned(inp, hp.anafast(h, lmax=inp.ellmax))
    
    # plot and save
    plot_corr_harmonic(inp, beta, hy_true, hy_infl, yy_true, yy_infl, hh)
    
    return
