import yaml
import os

##########################
# simple function for opening the file
def read_dict_from_yaml(yaml_file):
    assert(yaml_file != None)
    with open(yaml_file) as f:
        config = yaml.safe_load(f)
    return config
##########################

##########################
"""
class that contains input info
"""
class Info(object):
    def __init__(self, input_file):
        self.input_file = input_file
        p = read_dict_from_yaml(self.input_file)

        self.realistic = p['realistic']

        self.nside = p['nside']
        assert type(self.nside) is int and (self.nside & (self.nside-1) == 0) and self.nside != 0, "nside must be integer power of 2"
        self.ellmax = p['ellmax']
        if self.ellmax:
            assert type(self.ellmax) is int and self.ellmax >= 2, "ellmax must be integer >= 2"
            assert self.ellmax <= 3*self.nside-1, "ellmax must be less than 3*nside-1"
        if 'ells_per_bin' in p:
            self.ells_per_bin = p['ells_per_bin']
            assert type(self.ells_per_bin) is int and 1 <= self.ells_per_bin <= self.ellmax-2, "ells_per_bin must be int with 1 <= ells_per_bin <= ellmax-2"
        else:
            self.ells_per_bin = 1
        if 'ellmin' in p:
            self.ellmin = p['ellmin']
        else:
            self.ellmin = 0

        self.frequencies = p['frequencies']
        assert len(self.frequencies) >= 2, "ILC requires at least two frequency channels"
        assert set(self.frequencies).issubset({30, 44, 70, 100, 143, 217, 353, 545}), \
            "frequencies must be subset of Planck channels: {30, 44, 70, 100, 143, 217, 353, 545}"
        # frequencies assumed to be in strictly increasing order
        if ( any( i >= j for i, j in zip(self.frequencies, self.frequencies[1:]))):
            raise AssertionError
        self.components = p['components']
        assert 'tSZ' in self.components and 'CIB' in self.components, "'tSZ' and 'CIB' must be in components list"
        assert set(self.components).issubset({'tSZ', 'CIB', 'CMB', 'kSZ'}), "Allowed components are 'tSZ', 'CIB', 'CMB', 'kSZ'"
        self.ILC_type = p['ILC_type']
        assert self.ILC_type in {'harmonic', 'needlet'}, "ILC_type must be either 'needlet' or 'harmonic'"
        if self.ILC_type == 'needlet':
            self.GN_FWHM_arcmin = p['GN_FWHM_arcmin']
        
        self.beta_range = p['beta_range']
        assert len(self.beta_range) == 2, "beta_range must consist of two values corresponding to ends of interval"
        self.num_beta_vals = p['num_beta_vals']
        assert type(self.num_beta_vals) is int, "num_beta_vals must be an integer"
        assert self.num_beta_vals >= 1, "num_beta_vals must be at least 1"
        if 'num_parallel' in p:
            self.num_parallel = p['num_parallel']
            assert type(self.num_parallel) is int, "num_parallel must be an integer"
            assert self.num_parallel >= 1, "num_parallel must be at least 1"
        else:
            self.num_parallel = 1
        self.cib_inflation = p['cib_inflation']
        self.noise_type = p['noise_type']
        assert self.noise_type in {'Planck_no_beam', 'Planck_with_beam', 'SO'}, "Currently the only supported noise types are 'Planck_no_beam', 'Planck_with_beam', and 'SO'"
        if 'noise_fraction' in p:
            self.noise_fraction = p['noise_fraction']
            assert 0 < self.noise_fraction, "noise_fraction must be greater than 0"
        else:
            self.noise_fraction = 1.
        if self.noise_type in {'Planck_no_beam', 'Planck_with_beam'}:
            if 'beam_fraction' in p:
                self.beam_fraction = p['beam_fraction']
                assert 0 < self.beam_fraction, "beam_fraction must be greater than 0"
            else:
                self.beam_fraction = 1.
        self.harmonic_space = p['harmonic_space']
        
        self.cib_map_dir = p['cib_map_dir']
        assert type(self.cib_map_dir) is str, "TypeError: cib_map_dir must be str"
        assert os.path.isdir(self.cib_map_dir), "cib_map_dir does not exist"
        self.tsz_map_file = p['tsz_map_file']
        assert type(self.tsz_map_file) is str, "TypeError: tsz_map_file must be str"
        assert os.path.isfile(self.tsz_map_file), f"{self.tsz_map_file} does not exist"
        if 'kSZ' in self.components:
            self.ksz_map_file = p['ksz_map_file']
            assert type(self.ksz_map_file) is str, "TypeError: ksz_map_file must be str"
            assert os.path.isfile(self.ksz_map_file), f"{self.ksz_map_file} does not exist"
        if 'CMB' in self.components:
            self.cmb_map_file = p['cmb_map_file']
            assert type(self.cmb_map_file) is str, "TypeError: cmb_map_file must be str"
            assert os.path.isfile(self.cmb_map_file), f"{self.cmb_map_file} does not exist"
        assert 'halo_catalog' in p or 'halo_files_dir' in p, "Either halo_catalog or halo_files_dir must be defined"
        if 'halo_catalog' in p: #check for halo_catalog before halo_files_dir since using a single halo catalog is faster
            self.halo_catalog = p['halo_catalog']
            assert type(self.halo_catalog) is str, "TypeError: halo_catalog must be str"
            assert os.path.isfile(self.halo_catalog), "halo_catalog does not exist"
            self.halo_files_dir = None
        elif 'halo_files_dir' in p:
            self.halo_catalog = None
            self.halo_files_dir = p['halo_files_dir']
            assert type(self.halo_files_dir) is str, "TypeError: halo_files_dir must be str"
            assert os.path.isdir(self.halo_files_dir), "halo_files_dir does not exist"
        self.pyilc_path = p['pyilc_path']
        assert type(self.pyilc_path) is str, "TypeError: pyilc_path must be str"
        assert os.path.isdir(self.pyilc_path), "pyilc_path does not exist"


        for freq in self.frequencies:
            assert os.path.isfile(self.cib_map_dir+f'/mdpl2_len_mag_cibmap_planck_{freq}_uk.fits'), \
                f"missing file {self.cib_map_dir}/mdpl2_len_mag_cibmap_planck_{freq}_uk.fits"
        assert set(self.frequencies).issubset({30,44,70,100,143,217,353,545}), \
            "self.frequencies must be subset of {30,44,70,100,143,217,353,545}"

        self.cib_decorr = p['cib_decorr']

        self.output_dir = p['output_dir']
        assert type(self.output_dir) is str, "TypeError: output_dir must be str"
        self.debug = p['debug']
