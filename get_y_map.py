import subprocess
import yaml
import healpy as hp


def setup_pyilc(inp, env, beta, suppress_printing=False, inflated=False):
    '''
    Sets up yaml files for pyilc and runs the code

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    env: environment object
    beta: float, beta value to use for CIB deprojection
    suppress_printing: Bool, whether to suppress outputs and errors from pyilc code itself
    inflated: Bool, whether or not to use inflated CIB frequency maps

    RETURNS
    -------
    ymap: 1D numpy array in RING format containing final y-map from pyilc
    '''

    #set up yaml files for pyilc
    
    pyilc_input_params = {}
    if inflated:
        pyilc_input_params['output_dir'] = str(inp.output_dir) + f"/pyilc_outputs/beta_{beta:.2f}_inflated/"
    else:
        pyilc_input_params['output_dir'] = str(inp.output_dir) + f"/pyilc_outputs/beta_{beta:.2f}_uninflated/"
    pyilc_input_params['output_prefix'] = ""
    pyilc_input_params['save_weights'] = "no"
    pyilc_input_params['param_dict_file'] = f'{inp.output_dir}/pyilc_yaml_files/beta_{beta:.2f}.yaml'
    
    pyilc_input_params['ELLMAX'] = inp.ellmax
    pyilc_input_params['wavelet_type'] = 'TopHatHarmonic' 
    pyilc_input_params['BinSize'] = inp.ells_per_bin
    
    pyilc_input_params['N_freqs'] = len(inp.frequencies)
    pyilc_input_params['bandpass_type'] = 'ActualBandpasses' 
    pyilc_input_params['freqs_delta_ghz'] = inp.frequencies

    # the files where you have saved the bandpasses:
    pyilc_input_params['freq_bp_files'] = [f'{inp.pyilc_path}/data/HFI_BANDPASS_F{int(freq)}_reformat.txt' for freq in inp.frequencies]

    if inflated:
        pyilc_input_params['freq_map_files'] = \
            [f'{inp.output_dir}/maps/inflated_{freq}.fits' for freq in inp.frequencies]
    else:
        pyilc_input_params['freq_map_files'] = \
            [f'{inp.output_dir}/maps/uninflated_{freq}.fits' for freq in inp.frequencies]


    beam_dict = {30:32.29, 44:27.94, 70:13.08, 100:9.66, 143:7.22, 217:4.92, 353:4.90, 545:4.67}
    pyilc_input_params['beam_type'] = 'Gaussians'
    pyilc_input_params['beam_FWHM_arcmin'] = [beam_dict[freq] for freq in inp.frequencies]
    pyilc_input_params['perform_ILC_at_beam'] = 10 # the FWHM of the common beam to convolve to before performing ILC (in arcmin)

    pyilc_input_params['N_side'] = inp.nside
    pyilc_input_params['ILC_preserved_comp'] = 'tSZ'
    pyilc_input_params['N_deproj'] = 1
    pyilc_input_params['ILC_deproj_comps'] = ['CIB']
    pyilc_input_params['N_maps_xcorr'] = 0

    if inflated:
        ymap_yaml = f'{inp.output_dir}/pyilc_yaml_files/beta{beta:.2f}_inflated.yml'
    else:
        ymap_yaml = f'{inp.output_dir}/pyilc_yaml_files/beta{beta:.2f}_uninflated.yml'
    with open(ymap_yaml, 'w') as outfile:
        yaml.dump(pyilc_input_params, outfile, default_flow_style=None)


    #run pyilc and return y-map

    stdout = subprocess.DEVNULL if suppress_printing else None
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {ymap_yaml}"], shell=True, env=env, stdout=stdout, stderr=stdout)
    if inp.debug:
        print(f'generated ILC maps for beta={beta:.2f}, inflated={inflated}', flush=True)
    inflated_str = 'inflated' if inflated else 'uninflated'
    ymap = hp.read_map(f"{inp.output_dir}/pyilc_outputs/beta_{beta:.2f}_{inflated_str}/needletILCmap_component_tSZ_deproject_CIB.fits")
    
    
    return ymap