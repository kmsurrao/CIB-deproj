import healpy as hp
from tqdm import tqdm
import numpy as np
import pickle
# import h5py

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
    rot_all = np.arange(1,30)
    for rot in tqdm(rot_all):
        try:
            df = np.load(halo_ldir+'haloslc_rot_'+str(rot)+'.npy', allow_pickle=True)
            z_all = df[:,2]
            ra, dec = df[:,0], df[:,1]
            mvir = df[:,5]
            halo_ra_all.append(ra)
            halo_dec_all.append(dec)
            halo_z_all.append(z_all)
            halo_m_all.append(mvir)        
        except:
                pass
    halo_ra_all = np.concatenate(halo_ra_all)
    halo_dec_all = np.concatenate(halo_dec_all)
    halo_z_all = np.concatenate(halo_z_all)
    halo_m_all = np.concatenate(halo_m_all)

    #you can change the selection to have more or less halos to boost the SNR of correlations.
    indsel_z = np.where((halo_z_all > 0.0) & (halo_z_all < 0.25))[0]
    indsel_M = np.where((halo_m_all > 1e13) & (halo_m_all < 1e15))[0]
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
        hf = h5py.File('haloslc_agora_zsel_0_to_0.3_Msel_1e13_to_1e15.h5', 'w')
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
    indsel_z = np.where((halo_z_all > 0.0) & (halo_z_all < 0.25))[0]
    indsel_M = np.where((halo_m_all > 1e13) & (halo_m_all < 1e15))[0]
    indsel_all = np.intersect1d(indsel_z, indsel_M)

    ra_halos = halo_ra_all[indsel_all]
    dec_halos = halo_dec_all[indsel_all]

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