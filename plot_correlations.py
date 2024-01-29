import numpy as np
import matplotlib.pyplot as plt
import pickle
import healpy as hp
from scipy import stats
from utils import binned, cov

def plot_corr(inp, beta, y_true, y_recon, y_recon_inflated, h):
    '''
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
    tsz_auto = binned(hp.anafast(y_true, lmax=inp.ellmax))
    uninfl_auto = binned(hp.anafast(y_recon, lmax=inp.ellmax))
    infl_auto = binned(hp.anafast(y_recon_inflated, lmax=inp.ellmax))
    h_auto = binned(hp.anafast(h, lmax=inp.ellmax))
    uninflxh = binned(hp.anafast(y_recon, h, lmax=inp.ellmax))
    inflxh = binned(hp.anafast(y_recon_inflated, h, lmax=inp.ellmax))
    tszxh = binned(hp.anafast(y_true, h, lmax=inp.ellmax))

    # compute covariances
    cov_hytrue = cov(np.array([[tsz_auto, tszxh], [tszxh, h_auto]]))
    cov_hyuninfl = cov(np.array([[uninfl_auto, uninflxh], [uninflxh, h_auto]]))
    cov_hyinfl = cov(np.array([[infl_auto, inflxh], [inflxh, h_auto]]))

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
    plt.ylabel(r'$\ell(\ell+1)/(2\pi)$')
    plt.xlabel(r'$\ell$')
    plt.legend()
    plt.title(r'$h \times y$, $\beta=$' + f'{beta:0.3d}')
    plt.savefig(f'{inp.output_dir}/correlation_plots/beta_{beta:0.3d}.png')

    # save mean_ells, spectra, and errors to plot later
    to_save = [mean_ells, to_dl, uninflxh, inflxh, tszxh, cov_hyuninfl, cov_hyinfl, cov_hytrue]
    pickle.dump(to_save, open(f'{inp.output_dir}/correlation_plots/beta_{beta:0.3d}.p', 'wb'))

    return