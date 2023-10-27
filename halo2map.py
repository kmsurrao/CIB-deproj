import healpy as hp
from tqdm import tqdm
import numpy as np

def convert_halo2map(inp):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications

    RETURNS
    -------
    density: 1D numpy array in RING ordering containing halo density map
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
            df = np.load(halo_ldir+'haloslc_rot_'+str(rot)+'.npy')
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

    pix = hp.ang2pix(inp.nside, ra_halos, dec_halos, nest=False, lonlat=True)
    # ra_hp, dec_hp = hp.pix2ang(inp.nside, ipix=np.arange(hp.nside2npix(inp.nside)), nest=False, lonlat=True)
    pix.sort()
    first = np.where(pix[:-1] != pix[1:])[0] + 1
    first = np.concatenate(([0],first))
    hpix = pix[first]
    cts = np.diff(first,append=len(pix))
    cts_map = np.zeros(hp.nside2npix(inp.nside),dtype=int)
    cts_map[hpix] = cts
    cts_mean = np.mean(cts_map)
    density = cts_map/cts_mean - 1.
    hp.write_map(f'{inp.output_dir}/maps/halo.fits', density)

    return density