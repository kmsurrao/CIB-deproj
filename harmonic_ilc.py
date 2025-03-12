import numpy as np
import healpy as hp
import os
import pickle


def get_alm(inp, freq_map_names):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    freq_map_names: list of strings of length Nfreqs containing
                     names of frequency map files
    
    RETURNS
    -------
    all_alm: list of length Nfreqs containing numpy arrays with alm of 
                all frequency maps
    '''
    all_alm = []
    for fname in freq_map_names:
        freq_alm_name = fname[:-5] + '_alm' + '.p'
        if os.path.isfile(freq_alm_name):
            all_alm.append(pickle.load(open(freq_alm_name, 'rb')))
        else:
            alm = hp.map2alm(hp.read_map(fname), lmax=inp.ellmax)
            pickle.dump(alm, open(freq_alm_name, 'wb'))
            all_alm.append(alm)
    return all_alm


def get_Rlij_inv(inp, freq_alm):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    freq_alm: (Nfreqs, N_alm) ndarray 
        containing alm of freq. maps in ascending order of frequency
    
    RETURNS
    -------
    Rlij_inv: (ellmax+1, Nfreqs, Nfreqs) 
        ndarray containing inverse Rij matrix at each ell
    '''
    ells = np.arange(inp.ellmax+1)
    prefactor = (2*ells+1)/(4*np.pi)
    Nfreqs = len(inp.frequencies)
    half_delta_ell = inp.ells_per_bin//2

    Clij = np.zeros((Nfreqs, Nfreqs, inp.ellmax+1), dtype=np.float32)
    for i in range(Nfreqs):
        for j in range(Nfreqs):
            Clij[i,j] = hp.alm2cl(freq_alm[i], freq_alm[j], lmax=inp.ellmax)

    Rlij_no_binning = np.einsum('l,ijl->ijl', prefactor, Clij)
    Rlij = np.zeros((len(inp.frequencies), len(inp.frequencies), inp.ellmax+1)) 
    for i in range(Nfreqs):
        for j in range(Nfreqs):
            Rlij[i][j] = (np.convolve(Rlij_no_binning[i][j], np.ones(2*half_delta_ell+1)))[half_delta_ell: inp.ellmax+1+half_delta_ell]

    identity_matrices = np.eye(Nfreqs)[..., np.newaxis]  #shape (Nfreqs, Nfreqs, 1)
    Rlij_inv = np.linalg.solve(np.transpose(Rlij, axes=(2,0,1)), np.transpose(identity_matrices, axes=(2,0,1)))
    return Rlij_inv #index as Rlij_inv[l,i,j]
    

def weights(inp, Rlij_inv, signal_sed, contam_sed=None):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Rlij_inv: (ellmax+1, Nfreqs, Nfreqs) 
        ndarray containing inverse Rij matrix at each ell
    signal_sed: array-like of length Nfreqs containing spectral response
        of component of interest at each frequency
    contam_sed: array-like of length Nfreqs containing spectral response
        of component to deproject at each frequency (if None, standard ILC is performed).
        If provided as a 2D array of size (Nfreqs, ellmax+1), separate SEDs are deprojected
        at each ell.
    
    RETURNS
    -------
    w: (Nfreqs, ellmax+1) ndarray of harmonic ILC weights
    '''
    if contam_sed is None: #standard ILC
        numerator = np.einsum('lij,j->il', Rlij_inv, signal_sed)
        denominator = np.einsum('lkm,k,m->l', Rlij_inv, signal_sed, signal_sed)
        w = numerator/denominator 
    else: #constrained ILC
        if len(contam_sed.shape) == 2 and contam_sed.shape[1] > 1:
            A = np.einsum('lij,i,j->l', Rlij_inv, signal_sed, signal_sed)
            B = np.einsum('lij,il,jl->l', Rlij_inv, contam_sed, contam_sed)
            D = np.einsum('lij,i,jl->l', Rlij_inv, signal_sed, contam_sed)
            numerator = np.einsum('lij,l,i->jl', Rlij_inv, B, signal_sed) \
                        - np.einsum('lij,l,il->jl', Rlij_inv, D, contam_sed)
            denominator = np.einsum('l,l->l', A, B) - np.einsum('l,l->l', D, D)
        else:
            A = np.einsum('lij,i,j->l', Rlij_inv, signal_sed, signal_sed)
            B = np.einsum('lij,i,j->l', Rlij_inv, contam_sed, contam_sed)
            D = np.einsum('lij,i,j->l', Rlij_inv, signal_sed, contam_sed)
            numerator = np.einsum('lij,l,i->jl', Rlij_inv, B, signal_sed) \
                        - np.einsum('lij,l,i->jl', Rlij_inv, D, contam_sed)
            denominator = np.einsum('l,l->l', A, B) - np.einsum('l,l->l', D, D)
        w = numerator/denominator
    return w #index as w[i][l]


def HILC_map(inp, beta, signal_sed, contam_sed=None, inflated=False, no_cib=False, fname=None):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    beta: float, beta value to use for CIB deprojection (set to None if fname provided)
    signal_sed: array-like of length Nfreqs containing spectral response
        of component of interest at each frequency
    contam_sed: array-like of length Nfreqs containing spectral response
        of component to deproject at each frequency (if None, standard ILC is performed).
        If provided as a 2D array of size (Nfreqs, Nbins), separate SEDs are deprojected
        in each ell bin.
    inflated: Bool, whether or not to use inflated CIB frequency maps
    no_cib: Bool, if True, use maps without CIB included
    fname: str, filename. Default is None if beta is provided.

    RETURNS
    -------
    hilc_map: (Npix, ) numpy array containing harmonic ILC map

    '''
    inflated_str = 'inflated' if inflated else 'uninflated'
    if no_cib:
        freq_alm = get_alm(inp, [f'{inp.output_dir}/maps/no_cib_{freq}.fits' for freq in inp.frequencies])
    elif inflated:
        freq_alm = get_alm(inp, [f'{inp.output_dir}/maps/{inflated_str}_{freq}_{beta:.3f}.fits' for freq in inp.frequencies])
    else:
        freq_alm = get_alm(inp, [f'{inp.output_dir}/maps/{inflated_str}_{freq}.fits' for freq in inp.frequencies])
   
    Rlij_inv = get_Rlij_inv(inp, freq_alm)
    w = weights(inp, Rlij_inv, signal_sed, contam_sed)

    hilc_map = np.zeros(12*inp.nside**2, dtype=np.float32)
    for i, alm_orig in enumerate(freq_alm):
        alm = hp.almxfl(alm_orig, w[i])
        hilc_map += hp.alm2map(alm, inp.nside)  

    if fname is not None:
        pass
    elif contam_sed is None:
        fname = f'{inp.output_dir}/pyilc_outputs/uninflated/needletILCmap_component_tSZ.fits'
    elif not inflated:
        fname = f"{inp.output_dir}/pyilc_outputs/beta_{beta:.3f}_uninflated/needletILCmap_component_tSZ_deproject_CIB.fits"
    else:
        fname = f"{inp.output_dir}/pyilc_outputs/beta_{beta:.3f}_inflated/needletILCmap_component_tSZ_deproject_CIB.fits"
    hp.write_map(fname, hilc_map, dtype=np.float32)
    return hilc_map