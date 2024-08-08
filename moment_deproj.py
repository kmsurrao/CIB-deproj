import argparse
import os
import subprocess
import numpy as np
import healpy as hp

from input import Info
from generate_maps import get_freq_maps
from utils import *


def setup_pyilc(inp, env, beta, suppress_printing=False):
    '''
    Sets up yaml files for pyilc and runs the code

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    env: environment object
    beta: float, beta value to use for CIB deprojection
    suppress_printing: Bool, whether to suppress outputs and errors from pyilc code itself

    RETURNS
    -------
    ymap: 1D numpy array in RING format containing final y-map from pyilc
    '''

    #set up yaml files for pyilc
    
    pyilc_input_params = {}
    beta_str = f'beta_{beta::.3f}_'
    pyilc_input_params['output_dir'] = str(inp.output_dir) + f"/moment_deproj/"

    pyilc_input_params['output_prefix'] = ""
    pyilc_input_params['save_weights'] = "no"
    pyilc_input_params['param_dict_file'] = f'{inp.output_dir}/moment_deproj/{beta_str}param_dict.yaml'
    
    pyilc_input_params['ELLMAX'] = inp.ellmax
    pyilc_input_params['BinSize'] = inp.ells_per_bin
    if inp.ILC_type == 'harmonic':
        pyilc_input_params['wavelet_type'] = 'TopHatHarmonic' 
    elif inp.ILC_type == 'needlet':
        pyilc_input_params['wavelet_type'] = "GaussianNeedlets"
        pyilc_input_params['GN_FWHM_arcmin'] = [inp.GN_FWHM_arcmin[i] for i in range(len(inp.GN_FWHM_arcmin))]
        pyilc_input_params['N_scales'] = len(inp.GN_FWHM_arcmin)+1
    pyilc_input_params['taper_width'] = 200
    
    pyilc_input_params['N_freqs'] = len(inp.frequencies)
    if inp.cib_decorr:
        pyilc_input_params['bandpass_type'] = 'ActualBandpasses'
    else:
        pyilc_input_params['bandpass_type'] = 'DeltaBandpasses' 
    pyilc_input_params['freqs_delta_ghz'] = inp.frequencies

    # the files where you have saved the bandpasses:
    pyilc_input_params['freq_bp_files'] = [f'{inp.pyilc_path}/data/HFI_BANDPASS_F{int(freq)}_reformat.txt' for freq in inp.frequencies]

    pyilc_input_params['freq_map_files'] = \
        [f'{inp.output_dir}/maps/uninflated_{freq}.fits' for freq in inp.frequencies]

    pyilc_input_params['beam_type'] = 'Gaussians'
    pyilc_input_params['beam_FWHM_arcmin'] = [0.1]*len(inp.frequencies)

    pyilc_input_params['N_side'] = inp.nside
    pyilc_input_params['ILC_preserved_comp'] = 'tSZ'
    pyilc_input_params['N_deproj'] = 2
    pyilc_input_params['ILC_deproj_comps'] = ['CIB', 'CIB_dbeta']
    pyilc_input_params['ILC_bias_tol'] = 0.01
    pyilc_input_params['N_maps_xcorr'] = 0
    pyilc_input_params['save_as'] = 'fits'

    ymap_yaml = f'{inp.output_dir}/moment_deproj/{beta_str}ymap.yml'
    with open(ymap_yaml, 'w') as outfile:
        yaml.dump(pyilc_input_params, outfile, default_flow_style=None)

    #run pyilc and return y-map
    stdout = subprocess.DEVNULL
    stderr = subprocess.DEVNULL if suppress_printing else None
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {ymap_yaml}"], shell=True, env=env, stdout=stdout, stderr=stderr)
    if inp.debug:
        print(f'generated ILC maps for beta={beta::.3f}', flush=True)
    deproj_str = '_deproject_CIB_CIB_dbeta'
    ymap = hp.read_map(f"{inp.output_dir}/moment_deproj/needletILCmap_component_tSZ{deproj_str}.fits")
    
    return ymap



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
    if not os.path.isdir(inp.output_dir):
        subprocess.call(f'mkdir {inp.output_dir}', shell=True, env=env)
    if not os.path.isdir(f'{inp.output_dir}/moment_deproj'):
        subprocess.call(f'mkdir {inp.output_dir}/moment_deproj', shell=True, env=env)

    # get map of halos and maps at each frequency (both with and without inflated CIB)
    print('Getting maps at different frequencies...', flush=True)
    get_freq_maps(inp, diff_noise=False, no_cib=False)

    # write beta yaml
    beta = 1.65
    pars = {'beta_CIB': float(beta), 'Tdust_CIB': 24.0, 'nu0_CIB_ghz':353.0, 'kT_e_keV':5.0, 'nu0_radio_ghz':150.0, 'beta_radio': -0.5}
    beta_yaml = f'{inp.output_dir}/moment_deproj/beta_{beta::.3f}_param_dict.yaml'
    with open(beta_yaml, 'w') as outfile:
        yaml.dump(pars, outfile, default_flow_style=None)

    # get y-map
    ymap = setup_pyilc(inp, env, beta, suppress_printing=False)

    return



if __name__ == '__main__':
    main()