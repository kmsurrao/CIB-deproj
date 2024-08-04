import healpy as hp
from tqdm import tqdm
import numpy as np
import pickle
import h5py
import os
import argparse
from input import Info
from utils import setup_output_dir

def halodir2map(inp, save_single_catalog=True):
    '''
    Use this function if providing a directory of halo catalogs in the input

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    save_single_catalog: Bool, whether to save single catalog of all halos

    RETURNS
    -------
    density: 1D numpy array in RING ordering containing halo density map
    ra_halos: ndarray containing ra of halos
    dec_halos: ndarray containing dec of halos
    '''

    halo_ldir = inp.halo_files_dir

    halo_ra_all = []
    halo_dec_all = []
    halo_z_all = []
    halo_m_all = []
    # all the different shells hosting the halo catalogs
    rot_all = np.arange(1, 120)
    for rot in tqdm(rot_all):
        fname = halo_ldir+'/haloslc_rot_'+str(rot)+'_v050223.npz'
        if os.path.isfile(fname):
            df = np.load(fname, allow_pickle=True)
            z_all = df['totz']
            ra, dec = df['totra'], df['totdec']
            mvir = df['totmvir']
            halo_ra_all.append(ra)
            halo_dec_all.append(dec)
            halo_z_all.append(z_all)
            halo_m_all.append(mvir)
    halo_ra_all = np.concatenate(halo_ra_all, dtype=np.float32)
    halo_dec_all = np.concatenate(halo_dec_all, dtype=np.float32)
    halo_z_all = np.concatenate(halo_z_all, dtype=np.float32)
    halo_m_all = np.concatenate(halo_m_all, dtype=np.float32)

    #you can change the selection to have more or less halos to boost the SNR of correlations.
    # z_min, z_max = 0.8, 1.8
    # M_min, M_max = 1e12, 1e15
    z_min, z_max = 0.3, 0.8
    M_min, M_max = 5e12, 5e14
    indsel_z = np.where((halo_z_all > z_min) & (halo_z_all < z_max))[0]
    indsel_M = np.where((halo_m_all > M_min) & (halo_m_all < M_max))[0]
    indsel_all = np.intersect1d(indsel_z, indsel_M)

    ra_halos = halo_ra_all[indsel_all]
    dec_halos = halo_dec_all[indsel_all]
    z_halos = halo_z_all[indsel_all]
    m_halos = halo_m_all[indsel_all]

    pix = hp.ang2pix(inp.nside, ra_halos, dec_halos, nest=True, lonlat=True)
    pix.sort()
    first = np.where(pix[:-1] != pix[1:])[0] + 1
    first = np.concatenate(([0],first))
    hpix = pix[first]
    cts = np.diff(first,append=len(pix))
    cts_map = np.zeros(hp.nside2npix(inp.nside),dtype=int)
    cts_map[hpix] = cts
    cts_mean = np.mean(cts_map)
    density = cts_map/cts_mean - 1.
    density = hp.reorder(density, n2r=True)
    hp.write_map(f'{inp.output_dir}/maps/halo.fits', density, overwrite=True, dtype=np.float32)

    if save_single_catalog:
        hf = h5py.File(f'{inp.halo_files_dir}/haloslc_agora_zsel_{z_min}_to_{z_max}_Msel_{M_min:.0e}_to_{M_max:.0e}.h5', 'w')
        hf.create_dataset('ra', data=ra_halos)
        hf.create_dataset('dec', data=dec_halos)
        hf.create_dataset('z', data=z_halos)
        hf.create_dataset('M', data=m_halos)
        hf.close()

    return density, ra_halos, dec_halos


def halofile2map(inp):
    '''
    Use this function if providing a single halo catalog file in the input

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    ra_halos: ndarray containing ra of halos
    dec_halos: ndarray containing dec of halos

    RETURNS
    -------
    density: 1D numpy array in RING ordering containing halo density map
    '''
    df = h5py.File(inp.halo_catalog, 'r')
    ra = df.get('ra')[()]
    dec = df.get('dec')[()]
    z_all = df.get('z')[()]
    mvir = df.get('M')[()]
    halo_ra_all = np.array(ra)
    halo_dec_all= np.array(dec)
    halo_z_all= np.array(z_all)
    halo_m_all= np.array(mvir) 

    # # comment out section above and use section below if h5py not working
    # halo_ra_all = pickle.load(open(f'{inp.output_dir}/halo_ra_all.p', 'rb'))  
    # halo_dec_all = pickle.load(open(f'{inp.output_dir}/halo_dec_all.p', 'rb'))    
    # halo_z_all = pickle.load(open(f'{inp.output_dir}/halo_z_all.p', 'rb')) 
    # halo_m_all = pickle.load(open(f'{inp.output_dir}/halo_m_all.p', 'rb'))      

    #you can change the selection to have more or less halos to boost the SNR of correlations.
    ra_halos = halo_ra_all
    dec_halos = halo_dec_all

    pix = hp.ang2pix(inp.nside, ra_halos, dec_halos, nest=True, lonlat=True)
    pix.sort()
    first = np.where(pix[:-1] != pix[1:])[0] + 1
    first = np.concatenate(([0],first))
    hpix = pix[first]
    cts = np.diff(first,append=len(pix))
    cts_map = np.zeros(hp.nside2npix(inp.nside),dtype=int)
    cts_map[hpix] = cts
    cts_mean = np.mean(cts_map)
    density = cts_map/cts_mean - 1.
    density = hp.reorder(density, n2r=True)
    hp.write_map(f'{inp.output_dir}/maps/halo.fits', density, overwrite=True, dtype=np.float32)

    return density, ra_halos, dec_halos

if __name__ == '__main__':

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Construct halo map.")
    parser.add_argument("--config", default="example.yaml")
    args = parser.parse_args()
    input_file = args.config

    # read in the input file and set up relevant info object
    inp = Info(input_file)

    # set up output directory
    env = os.environ.copy()
    setup_output_dir(inp, env)

    # create single halo catalog and map
    halodir2map(inp, save_single_catalog=True)