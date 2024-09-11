import numpy as np
import healpy as hp


def get_Rlij_inv(inp, freq_maps):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    freq_maps: (Nfreqs, Npix) ndarray 
        containing frequency maps in ascending order of frequency
    
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
            Clij[i,j] = hp.anafast(freq_maps[i], freq_maps[j], lmax=inp.ellmax)

    Rlij_inv = np.zeros((inp.ellmax+1, Nfreqs, Nfreqs), dtype=np.float32)
    Rlij_no_binning = np.einsum('l,ijl->ijl', prefactor, Clij)
    Rlij = np.zeros((len(inp.freqs), len(inp.freqs), inp.ellmax+1)) 
    for i in range(Nfreqs):
        for j in range(Nfreqs):
            Rlij[i][j] = (np.convolve(Rlij_no_binning[i][j], np.ones(2*half_delta_ell+1)))[half_delta_ell: inp.ellmax+1+half_delta_ell]
    Rlij_inv = np.array([np.linalg.inv(Rlij[:,:,l]) for l in range(inp.ellmax+1)]) 
    return Rlij_inv #index as Rlij_inv[l,i,j]
    

def weights(Rlij_inv, signal_sed, contam_sed=None):
    '''
    ARGUMENTS
    ---------
    Rlij_inv: (ellmax+1, Nfreqs, Nfreqs) 
        ndarray containing inverse Rij matrix at each ell
    signal_sed: array-like of length Nfreqs containing spectral response
        of component of interest at each frequency
    contam_sed: array-like of length Nfreqs containing spectral response
        of component to deproject at each frequency (if None, standard ILC is performed)
    
    RETURNS
    -------
    w: (Nfreqs, ellmax+1) ndarray of harmonic ILC weights
    '''
    if contam_sed is None: #standard ILC
        numerator = np.einsum('lij,j->il', Rlij_inv, signal_sed)
        denominator = np.einsum('lkm,k,m->l', Rlij_inv, signal_sed, signal_sed)
        w = numerator/denominator 
    else: #constrained ILC
        A = np.einsum('lij,i,j->l', Rlij_inv, signal_sed, signal_sed)
        B = np.einsum('lij,i,j->l', Rlij_inv, contam_sed, contam_sed)
        D = np.einsum('lij,i,j->l', Rlij_inv, signal_sed, contam_sed)
        numerator = np.einsum('lij,l,i->jl', Rlij_inv, B, signal_sed) \
                    - np.einsum('lij,l,i->jl', Rlij_inv, D, contam_sed)
        denominator = np.einsum('l,l->l', A, B) - np.einsum('l,l->l', D, D)
        w = numerator/denominator
    return w #index as w[i][l]


def HILC_map(inp, beta, signal_sed, contam_sed=None, inflated=False, no_cib=False):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    beta: float, beta value to use for CIB deprojection
    signal_sed: array-like of length Nfreqs containing spectral response
        of component of interest at each frequency
    contam_sed: array-like of length Nfreqs containing spectral response
        of component to deproject at each frequency (if None, standard ILC is performed)
    inflated: Bool, whether or not to use inflated CIB frequency maps
    no_cib: Bool, if True, use maps without CIB included

    RETURNS
    -------
    hilc_map: (Npix, ) numpy array containing harmonic ILC map

    '''
    inflated_str = 'inflated' if inflated else 'uninflated'
    if inp.realistic and inflated:
        inflated_str += '_realistic'
    if no_cib:
        freq_maps = [f'{inp.output_dir}/maps/no_cib_{freq}.fits' for freq in inp.frequencies]
    elif inflated and inp.realistic:
        freq_maps = [f'{inp.output_dir}/maps/{inflated_str}_{freq}_{beta:.3f}.fits' for freq in inp.frequencies]
    else:
        freq_maps = [f'{inp.output_dir}/maps/{inflated_str}_{freq}.fits' for freq in inp.frequencies]
   
    Rlij_inv = get_Rlij_inv(inp, freq_maps)
    w = weights(Rlij_inv, signal_sed, contam_sed)

    hilc_map = np.zeros(inp.nside, dtype=np.float32)
    for i, map_ in enumerate(freq_maps):
        alm = hp.map2alm(map_, lmax=inp.ellmax)
        alm = hp.almxfl(alm, w[i])
        hilc_map += hp.alm2map(alm, inp.nside)  
    
    if not inflated:
        fname = f"{inp.output_dir}/pyilc_outputs/beta_{beta:.3f}_uninflated/needletILCmap_component_tSZ_deproject_CIB.fits"
    else:
        infl_str = 'inflated_realistic' if inp.realistic else 'inflated'
        fname = f"{inp.output_dir}/pyilc_outputs/beta_{beta:.3f}_{infl_str}/needletILCmap_component_tSZ_deproject_CIB.fits"
    hp.write_map(fname, hilc_map)
    return hilc_map