import healpy as hp
import numpy as np
from scipy import interpolate
from planck_noise import get_planck_noise, get_planck_specs
from inpaint_pixels import initial_masking
from utils import tsz_spectral_response, cib_spectral_response


def filter_low_ell(map_, ellmin):
    '''
    ARGUMENTS
    ---------
    map_: 1D numpy array containing healpy map in RING order
    ellmin: int, minimum ell for which to keep power in the map

    RETURNS
    -------
    filtered_map: map_ with modes below ellmin filtered out
    '''
    if ellmin > 0:
        alm = hp.map2alm(map_)
        l_arr, m_arr = hp.Alm.getlm(3*hp.get_nside(map_)-1)
        filtered_alm = alm*(l_arr >= ellmin)
        filtered_map = hp.alm2map(filtered_alm, hp.get_nside(map_))
    else:
        filtered_map = map_
    return filtered_map


def get_noise(inp, freq, diff_noise=False):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    freq: float, frequency for which to get noise (GHz)
    diff_noise: Bool, whether to use different noise realizations in uninflated
                and inflated frequency maps (only matters if not using realistic pipeline)

    RETURNS
    -------
    noise1_map, noise2_map: 1D numpy array healpix RING ordering noise maps
        (noise1_map == noise2_map if diff_noise is False)
    '''
    if inp.noise_type == 'Planck_no_beam':
        PS_noise_Planck = get_planck_noise(inp)
        planck_freqs = [30, 44, 70, 100, 143, 217, 353, 545]
    
    elif inp.noise_type == 'Planck_with_beam':
        planck_freq, planck_noise, planck_beam = get_planck_specs()
        planck_noise_interp = interpolate.interp1d(planck_freq, planck_noise)

    if inp.noise_type == 'Planck_no_beam':
        idx = planck_freqs.index(freq)
        PS_noise = inp.noise_fraction*PS_noise_Planck[idx]
        noise1_map = 10**(-6)*hp.synfast(PS_noise, nside=inp.nside) #units of K
        if not diff_noise:
            noise2_map = noise1_map
        else:
            noise2_map = 10**(-6)*hp.synfast(PS_noise, nside=inp.nside) * np.sqrt(2) #units of K
            noise1_map *= np.sqrt(2)

    elif inp.noise_type == 'Planck_with_beam':
        npix = hp.nside2npix(inp.nside)
        pix_side_arcmin = 60. * (180. / np.pi) * np.sqrt(4. * np.pi / npix)
        noise_level = planck_noise_interp(freq)
        noise_sigma = noise_level / pix_side_arcmin
        noise1_map = inp.noise_fraction * np.random.normal(scale=noise_sigma, size=npix) #units of Kcmb
        if not diff_noise:
            noise2_map = noise1_map
        else:
            noise2_map = np.sqrt(inp.noise_fraction) * np.random.normal(scale=noise_sigma, size=npix) * np.sqrt(2) #units of Kcmb
            noise1_map *= np.sqrt(2)
    
    elif inp.noise_type == 'SO':
        so_freqs = np.array([27, 39, 93, 145, 225, 280])
        nearest = so_freqs[(np.abs(so_freqs - freq)).argmin()]
        noise_file = open(f'so_noise/noise_{nearest}GHz.txt', 'r')
        rows = noise_file.readlines()
        for i, line in enumerate(rows):
            rows[i] = line.lstrip(' ').replace('\n', '').split()
        rows = np.asarray(rows, dtype=np.float32)
        ells_noise, noise_ps = rows
        f = interpolate.interp1d(ells_noise, noise_ps, fill_value="extrapolate", kind='cubic')
        noise_ps_interp = inp.noise_fraction*f(np.arange(inp.ellmax+1))
        noise1_map = 10**(-6)*hp.synfast(noise_ps_interp, nside=inp.nside) #units of K
        if not diff_noise:
            noise2_map = noise1_map
        else:
            noise2_map = 10**(-6)*hp.synfast(noise_ps_interp, nside=inp.nside) * np.sqrt(2) #units of K
            noise1_map *= np.sqrt(2)
    return noise1_map, noise2_map



def get_freq_maps(inp, diff_noise=False, no_cib=False):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    diff_noise: Bool, whether to use different noise realizations in uninflated
                and inflated frequency maps (only matters if not using realistic pipeline)
    no_cib: Bool, if True, do not include CIB in the maps

    RETURNS
    -------
    None (writes frequency maps to output_dir)
    '''

    np.random.seed(0)

    delta_bandpasses = True if not inp.cib_decorr else False
    tsz_response_vec = tsz_spectral_response(inp.frequencies, delta_bandpasses = delta_bandpasses, inp=inp)
    ymap = hp.read_map(inp.tsz_map_file)
    ymap = hp.ud_grade(ymap, inp.nside) #unitless
    cib_map_143 = 10**(-6)*hp.ud_grade(hp.read_map(f'{inp.cib_map_dir}/mdpl2_len_mag_cibmap_planck_143_uk.fits'), inp.nside) #units of K

    for i, freq in enumerate(inp.frequencies):

        noise1_map, noise2_map = get_noise(inp, freq, diff_noise=diff_noise)
            
        tsz_map = tsz_response_vec[i]*ymap #units of K
        if inp.cib_decorr:
            cib_map = hp.read_map(f'{inp.cib_map_dir}/mdpl2_len_mag_cibmap_planck_{freq}_uk.fits')
        else:
            cib_353 = hp.read_map(f'{inp.cib_map_dir}/mdpl2_len_mag_cibmap_planck_353_uk.fits')
            cib_map = cib_353/cib_spectral_response([353])[0]*cib_spectral_response([freq])[0]
        cib_map = 10**(-6)*hp.ud_grade(cib_map, inp.nside) #units of K
        cib_map = initial_masking(inp, cib_map, cib_map_143)
        additional_maps = np.zeros(12*inp.nside**2, dtype=np.float32)
        if 'kSZ' in inp.components:
            additional_maps += 10**(-6)*hp.ud_grade(hp.read_map(inp.ksz_map_file), inp.nside)
        if 'CMB' in inp.components:
            additional_maps += 10**(-6)*hp.ud_grade(hp.read_map(inp.cmb_map_file), inp.nside)
        if no_cib:
            freq_map_uninflated = tsz_map + noise1_map + additional_maps
            map_name = f'{inp.output_dir}/maps/no_cib_{freq}.fits'
        else:
            freq_map_uninflated = tsz_map + inp.cib_inflation[0]*cib_map + noise1_map + additional_maps
            map_name = f'{inp.output_dir}/maps/uninflated_{freq}.fits'

        # filter out low ell modes
        freq_map_uninflated = filter_low_ell(freq_map_uninflated, inp.ellmin)

        hp.write_map(map_name, freq_map_uninflated, overwrite=True, dtype=np.float32)
        if inp.debug:
            print(f'saved {map_name}', flush=True)
        if not inp.realistic and not no_cib:
            freq_map_inflated = tsz_map + inp.cib_inflation[1]*cib_map + noise2_map + additional_maps
            # filter out low ell modes
            freq_map_inflated = filter_low_ell(freq_map_inflated, inp.ellmin)
            map_name = f'{inp.output_dir}/maps/inflated_{freq}.fits'
            hp.write_map(map_name, freq_map_inflated, overwrite=True, dtype=np.float32)
            if inp.debug:
                print(f'saved {map_name}', flush=True)
    return


def get_realistic_infl_maps(inp, beta):
    '''
    Run this function to get realistic inflated CIB maps using
    residual (full frequency maps - tSZ SED * y-map with some deprojected beta)

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    beta: int, beta value used to build original y-map to subtract for residual

    RETURNS
    -------
    None (writes frequency maps to output_dir)
    '''
    ymap = hp.read_map(f"{inp.output_dir}/pyilc_outputs/beta_{beta:.3f}_uninflated/needletILCmap_component_tSZ_deproject_CIB.fits")
    delta_bandpasses = True if not inp.cib_decorr else False
    sed_vec = tsz_spectral_response(inp.frequencies, delta_bandpasses = delta_bandpasses, inp=inp)
    for i, freq in enumerate(inp.frequencies):
        orig_freq_map = hp.read_map(f'{inp.output_dir}/maps/uninflated_{freq}.fits')
        residual = orig_freq_map - sed_vec[i]*ymap
        infl_map = orig_freq_map + inp.cib_inflation[1]*residual
        hp.write_map(f'{inp.output_dir}/maps/inflated_realistic_{freq}_{beta:.3f}.fits', infl_map, overwrite=True, dtype=np.float32)
        if inp.debug:
            print(f'saved {inp.output_dir}/maps/inflated_realistic_{freq}_{beta:.3f}.fits', flush=True)
    return