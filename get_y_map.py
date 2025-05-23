import subprocess
import yaml
import healpy as hp
import numpy as np
import os
from harmonic_ilc import HILC_map
from generate_maps import *
from utils import *


def setup_pyilc(inp, env, beta, suppress_printing=False, inflated=False, standard_ilc=False, no_cib=False, moment_deproj=False):
    '''
    Sets up yaml files for pyilc and runs the code

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    env: environment object
    beta: float, beta value to use for CIB deprojection
    suppress_printing: Bool, whether to suppress outputs and errors from pyilc code itself
    inflated: Bool, whether or not to use inflated CIB frequency maps
    standard_ilc: Bool, whether to use a standard ILC without CIB deprojection
    no_cib: Bool, if True, use maps without CIB included
    moment_deproj: Bool, if True, deproject both beta and first moment w.r.t. beta

    RETURNS
    -------
    ymap: 1D numpy array in RING format containing final y-map from pyilc
    '''

    #set up yaml files for pyilc
    
    pyilc_input_params = {}
    inflated_str = 'inflated' if inflated else 'uninflated'
    if moment_deproj:
        pyilc_input_params['output_dir'] = str(inp.output_dir) + f"/pyilc_outputs/final/"
    elif not standard_ilc:
        pyilc_input_params['output_dir'] = str(inp.output_dir) + f"/pyilc_outputs/beta_{beta:.3f}_{inflated_str}/"
    else:
        pyilc_input_params['output_dir'] = str(inp.output_dir) + f"/pyilc_outputs/{inflated_str}/"

    pyilc_input_params['output_prefix'] = ""
    pyilc_input_params['save_weights'] = "no"
    if not standard_ilc:
        pyilc_input_params['param_dict_file'] = f'{inp.output_dir}/pyilc_yaml_files/beta_{beta:.3f}.yaml'
    
    pyilc_input_params['ELLMAX'] = inp.ellmax
    pyilc_input_params['BinSize'] = inp.ells_per_bin
    pyilc_input_params['wavelet_type'] = 'TopHatHarmonic' 
    pyilc_input_params['taper_width'] = 0
    
    pyilc_input_params['N_freqs'] = len(inp.frequencies)
    if not inp.delta_passbands:
        pyilc_input_params['bandpass_type'] = 'ActualBandpasses'
    else:
        pyilc_input_params['bandpass_type'] = 'DeltaBandpasses' 
    pyilc_input_params['freqs_delta_ghz'] = inp.frequencies

    # the files where you have saved the bandpasses:
    pyilc_input_params['freq_bp_files'] = [f'{inp.pyilc_path}/data/HFI_BANDPASS_F{int(freq)}_reformat.txt' for freq in inp.frequencies]

    if no_cib:
        pyilc_input_params['freq_map_files'] = \
            [f'{inp.output_dir}/maps/no_cib_{freq}.fits' for freq in inp.frequencies]
    elif inflated:
        pyilc_input_params['freq_map_files'] = \
            [f'{inp.output_dir}/maps/{inflated_str}_{freq}_{beta:.3f}.fits' for freq in inp.frequencies]
    else:
        pyilc_input_params['freq_map_files'] = \
            [f'{inp.output_dir}/maps/{inflated_str}_{freq}.fits' for freq in inp.frequencies]

    pyilc_input_params['beam_type'] = 'Gaussians'
    pyilc_input_params['beam_FWHM_arcmin'] = [0.1]*len(inp.frequencies)

    pyilc_input_params['N_side'] = inp.nside
    pyilc_input_params['ILC_preserved_comp'] = 'tSZ'
    if moment_deproj:
        pyilc_input_params['N_deproj'] = 2
        pyilc_input_params['ILC_deproj_comps'] = ['CIB', 'CIB_dbeta']
    elif not standard_ilc:
        pyilc_input_params['N_deproj'] = 1
        pyilc_input_params['ILC_deproj_comps'] = ['CIB']
    else:
        pyilc_input_params['N_deproj'] = 0
    pyilc_input_params['ILC_bias_tol'] = 0.01
    pyilc_input_params['N_maps_xcorr'] = 0
    pyilc_input_params['save_as'] = 'fits'

    if standard_ilc:
        beta_str = ''
    else:
        beta_str = f'beta{beta:.3f}_'
    if inflated:
        ymap_yaml = f'{inp.output_dir}/pyilc_yaml_files/{beta_str}inflated.yml'
    else:
        ymap_yaml = f'{inp.output_dir}/pyilc_yaml_files/{beta_str}uninflated.yml'
    with open(ymap_yaml, 'w') as outfile:
        yaml.dump(pyilc_input_params, outfile, default_flow_style=None)


    #run pyilc and return y-map
    stdout = subprocess.DEVNULL
    stderr = subprocess.DEVNULL if suppress_printing else None
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {ymap_yaml}"], shell=True, env=env, stdout=stdout, stderr=stderr)
    if inp.debug:
        print(f'generated ILC maps for beta={beta:.3f}, inflated={inflated}', flush=True)
    beta_str = f'beta_{beta:.3f}_' if not standard_ilc else ''
    deproj_str = '_deproject_CIB' if not standard_ilc else ''
    if moment_deproj:
        deproj_str += '_CIB_dbeta'
        ymap = hp.read_map(f"{inp.output_dir}/pyilc_outputs/final/needletILCmap_component_tSZ{deproj_str}.fits")
    else:
        ymap = hp.read_map(f"{inp.output_dir}/pyilc_outputs/{beta_str}{inflated_str}/needletILCmap_component_tSZ{deproj_str}.fits")
    
    
    return ymap


def get_all_ymaps(inp, beta):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    beta: float, value of beta for CIB deprojection

    RETURNS
    -------
    1

    '''
    y_recon_file = f"{inp.output_dir}/pyilc_outputs/beta_{beta:.3f}_uninflated/needletILCmap_component_tSZ_deproject_CIB.fits"
    if not os.path.isfile(y_recon_file):
        tsz_sed = tsz_spectral_response(inp.frequencies, delta_bandpasses=inp.delta_passbands, inp=inp)
        cib_sed = cib_spectral_response(inp.frequencies, delta_bandpasses=inp.delta_passbands, inp=inp, beta=beta)
        HILC_map(inp, beta, tsz_sed, contam_sed=cib_sed, inflated=False)
    y_recon_infl_file = f"{inp.output_dir}/pyilc_outputs/beta_{beta:.3f}_inflated/needletILCmap_component_tSZ_deproject_CIB.fits"
    if not os.path.isfile(y_recon_infl_file):
        get_realistic_infl_maps(inp, beta)
        tsz_sed = tsz_spectral_response(inp.frequencies, delta_bandpasses=inp.delta_passbands, inp=inp)
        cib_sed = cib_spectral_response(inp.frequencies, delta_bandpasses=inp.delta_passbands, inp=inp, beta=beta)
        contam_sed = cib_sed*(1+inp.h_vec)
        HILC_map(inp, beta, tsz_sed, contam_sed = contam_sed, inflated=True)
    return 1


def get_all_ymaps_star(args):
    '''
    Useful for using multiprocessing imap
    (imap supports tqdm but starmap does not)

    ARGUMENTS
    ---------
    args: arguments to function get_all_ymaps

    RETURNS
    -------
    function of *args, get_all_ymaps(inp, beta)
    '''
    return get_all_ymaps(*args)


if __name__ == '__main__':

    # for testing
    import argparse
    from input import Info
    from halo2map import halodir2map, halofile2map


    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Optimal beta value for CIB deprojection.")
    parser.add_argument("--config", default="example_websky.yaml")
    args = parser.parse_args()
    input_file = args.config

    # read in the input file and set up relevant info object
    inp = Info(input_file)

    # set up output directory
    env = os.environ.copy()
    setup_output_dir(inp, env)
    
    # get map of halos and maps at each frequency (both with and without inflated CIB)
    if inp.halo_map is not None:
        h = hp.ud_grade(hp.read_map(inp.halo_map), inp.nside)
    elif inp.halo_catalog is not None:                                                  
        h = halofile2map(inp)
    else:
        h = halodir2map(inp)
    print('got ra and dec of halos', flush=True)
    print('Getting maps at different frequencies...', flush=True)
    get_freq_maps(inp, diff_noise=False, no_cib=False)

    # test pyilc
    beta = 1.65
    setup_pyilc(inp, env, beta, suppress_printing=False, inflated=False, standard_ilc=False, no_cib=False)