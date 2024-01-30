import numpy as np
import matplotlib.pyplot as plt
import pickle
import healpy as hp
from scipy import stats
from utils import binned, cov


def plot_corr_harmonic(inp, beta, uninflxh, inflxh, tszxh, uninfl_auto, infl_auto, tsz_auto, h_auto):
    '''
    Plots harmonic space correlations (power spectra) given the spectra

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    beta: float, value of beta for CIB deprojection
    uninflxh: 1D numpy array of length Nbins containing cross-spectrum of halos with reconstructed y 
        (without CIB inflation)
    inflxh: 1D numpy array of length Nbins containing cross-spectrum of halos with reconstructed y 
        (with CIB inflation)
    tszxh: 1D numpy array of length Nbins containing cross-spectrum of halos with the true y map
    uninfl_auto: 1D numpy array of length Nbins containing auto-spectrum of reconstructed y
        (without CIB inflation)
    infl_auto: 1D numpy array of length Nbins containing auto-spectrum of reconstructed y
        (with CIB inflation)
    tsz_auto: 1D numpy array of length Nbins containing auto-spectrum of the true y map
    h_auto: 1D numpy array of length Nbins containing auto-spectrum of halo map

    RETURNS
    -------
    None (saves plots in {output_dir}/plot_correlations)
    '''
    # compute covariances
    cov_hytrue = cov(inp, np.array([[tsz_auto, tszxh], [tszxh, h_auto]]))
    cov_hyuninfl = cov(inp, np.array([[uninfl_auto, uninflxh], [uninflxh, h_auto]]))
    cov_hyinfl = cov(inp, np.array([[infl_auto, inflxh], [inflxh, h_auto]]))

    # plot
    ells = np.arange(inp.ellmax+1)
    Nbins = inp.ellmax//inp.ells_per_bin
    res = stats.binned_statistic(ells[2:], ells[2:], statistic='mean', bins=Nbins)
    mean_ells = (res[1][:-1]+res[1][1:])/2
    to_dl = mean_ells*(mean_ells+1)/2/np.pi
    plt.clf()
    plt.errorbar(mean_ells, to_dl*uninflxh, label='y recon.', yerr=to_dl*np.sqrt(np.diagonal(cov_hyuninfl)))
    plt.errorbar(mean_ells, to_dl*inflxh, label='y recon. (CIB inflated)', yerr=to_dl*np.sqrt(np.diagonal(cov_hyinfl)), linestyle='dashed')
    plt.errorbar(mean_ells, to_dl*tszxh, label='y true', yerr=to_dl*np.sqrt(np.diagonal(cov_hytrue)), linestyle='dotted')
    plt.grid()
    plt.ylabel(r'$\ell(\ell+1)/(2\pi)$ [$\mu \mathrm{K}^2$]')
    plt.xlabel(r'$\ell$')
    plt.legend()
    plt.title(r'$h \times y$, $\beta=$' + f'{beta:0.3f}')
    plt.savefig(f'{inp.output_dir}/correlation_plots/beta_{beta:0.3f}.png')

    # save mean_ells, spectra, and errors to plot later
    to_save = [mean_ells, to_dl, uninflxh, inflxh, tszxh, uninfl_auto, infl_auto, tsz_auto, h_auto]
    pickle.dump(to_save, open(f'{inp.output_dir}/correlation_plots/beta_{beta:0.3f}.p', 'wb'))

    return


def plot_corr_real(inp, beta, uninflxh, inflxh, tszxh, cov_hyuninfl, cov_hyinfl, cov_hytrue, r_hy):
    '''
    Plots real space correlations

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    beta: float, value of beta for CIB deprojection
    uninflxh: 1D numpy array of length nrad containing correlation of halos with reconstructed y 
        (without CIB inflation)
    inflxh: 1D numpy array of length nrad containing correlation of halos with reconstructed y 
        (with CIB inflation)
    tszxh: 1D numpy array of length nrad containing correlation of halos with the true y map
    cov_hyuninfl: 3D numpy array of shape (2,2,nrad) containing covariance 
        matrix of halos and the reconstructed y map (without CIB inflation)
    cov_hyinfl: 3D numpy array of shape (2,2,nrad) containing covariance 
        matrix of halos and the reconstructed y map (with CIB inflation)
    cov_hytrue: 3D numpy array of shape (2,2,nrad) containing covariance 
        matrix of halos and the true y map
    r_hy: 1D numpy array of length nrad containing x-axis point for plotting

    RETURNS
    -------
    None (saves plots in {output_dir}/plot_correlations)
    '''

    # plot
    plt.clf()
    plt.errorbar(r_hy, uninflxh, yerr=np.sqrt(np.diag(cov_hyuninfl)), ls='', marker='x', ms=3.0, label='y recon.')
    plt.errorbar(1.02*r_hy, inflxh, yerr=np.sqrt(np.diag(cov_hyinfl)), ls='', marker='o', ms=3.0, label='y recon. (CIB inflated)')
    plt.errorbar(1.04*r_hy, tszxh, yerr=np.sqrt(np.diag(cov_hytrue)), ls='', marker='v', ms=3.0, label='y true')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')

    # save mean_ells, spectra, and errors to plot later
    to_save = [uninflxh, inflxh, tszxh, cov_hyuninfl, cov_hyinfl, cov_hytrue]
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
    tsz_auto = binned(inp, hp.anafast(y_true, lmax=inp.ellmax))
    uninfl_auto = binned(inp, hp.anafast(y_recon, lmax=inp.ellmax))
    infl_auto = binned(inp, hp.anafast(y_recon_inflated, lmax=inp.ellmax))
    h_auto = binned(inp, hp.anafast(h, lmax=inp.ellmax))
    uninflxh = binned(inp, hp.anafast(y_recon, h, lmax=inp.ellmax))
    inflxh = binned(inp, hp.anafast(y_recon_inflated, h, lmax=inp.ellmax))
    tszxh = binned(inp, hp.anafast(y_true, h, lmax=inp.ellmax))
    uninfl_auto = binned(inp, hp.anafast(y_recon, lmax=inp.ellmax))
    infl_auto = binned(inp, hp.anafast(y_recon_inflated, lmax=inp.ellmax))
    tsz_auto = binned(inp, hp.anafast(y_true, lmax=inp.ellmax))
    h_auto = binned(inp, hp.anafast(h, lmax=inp.ellmax))
    
    # plot and save
    plot_corr_harmonic(inp, beta, uninflxh, inflxh, tszxh, uninfl_auto, infl_auto, tsz_auto, h_auto)
    
    return
