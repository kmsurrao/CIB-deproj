# script for stampede
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pickle
import h5py

nside = 1024

base_dir = '/scratch/09334/ksurrao/cib_deproj_013125/highzhalos/planck_1p0noise_cibdecorr_highfreqspecial_highzhalos_alpha1'
h = hp.read_map(f'{base_dir}/maps/halo.fits') 
y_beta = hp.read_map(f'{base_dir}/pyilc_outputs/beta_1.690_uninflated/needletILCmap_component_tSZ_deproject_CIB.fits')
y_beta_dbeta = hp.read_map(f'{base_dir}/moment_deproj/needletILCmap_component_tSZ_deproject_CIB_CIB_dbeta.fits')
ytrue = hp.ud_grade(hp.read_map('/scratch/09334/ksurrao/agora_planck/tsz_2048.fits'), nside)
halo_catalog = '/scratch/09334/ksurrao/halo_files/haloslc_agora_zsel_0.8_to_1.8_Msel_1e12_to_1e15_rot120.h5'
odir = '/work2/09334/ksurrao/stampede3/Github/CIB-deproj/stacks'


def stack(y_map, halo_catalog, name):

    fwhm_rad = np.radians(10/60)  # adjust as needed, was 0.5
    y_map = hp.smoothing(y_map, fwhm=fwhm_rad)
    
    # Parameters for native resolution (nside=1024 ~ 3.44 arcmin per pixel)
    reso = 3.44  # arcmin per pixel
    patch_size_deg = 2.0  # patch size in degrees (2° x 2°)
    # Convert patch size in degrees to arcmin, then to pixels:
    xsize = int((patch_size_deg * 60) / reso)  # ~35 pixels for a 2° patch
    
    df = h5py.File(halo_catalog, 'r')
    ra = df.get('ra')[()]
    dec = df.get('dec')[()]
    z_all = df.get('z')[()]
    mvir = df.get('M')[()]
    print('len(ra_halos): ', len(np.array(ra)))
    ra_halos = np.array(ra)
    dec_halos = np.array(dec)

    # Convert RA, Dec (in degrees) to HEALPix theta, phi in radians:
    # theta = colatitude = 90° - dec, and phi = ra
    theta_halos = np.radians(90.0 - dec_halos)
    phi_halos = np.radians(ra_halos)
    

    # List to store each 2D projected patch
    patches_2d = []

    # Loop over each halo to extract a patch via a gnomonic projection
    for theta, phi in zip(theta_halos, phi_halos):
        # For gnomonic projection, healpy expects the rotation in degrees.
        # Convert theta back to latitude in degrees (latitude = 90° - theta in deg)
        lat_deg = 90.0 - np.degrees(theta)
        lon_deg = np.degrees(phi)

        # Extract the 2D patch around the halo.
        # The parameter 'reso' is in arcmin/pixel and xsize is the number of pixels.
        patch = hp.gnomview(y_map, rot=(lon_deg, lat_deg), xsize=xsize, reso=reso,
                            return_projected_map=True, no_plot=True)
        patches_2d.append(patch)

    # Convert list of patches to a numpy array (assuming they all have the same shape)
    patches_array = np.array(patches_2d)
    # Stack (average) the patches
    stacked_patch_2d = np.mean(patches_array, axis=0)
    pickle.dump(stacked_patch_2d, open(f'{odir}/{name}.p', 'wb'))
    
    return stacked_patch_2d


stack(h, halo_catalog, 'stack_h')
stack(ytrue, halo_catalog, 'stack_ytrue')
stack(y_beta, halo_catalog, 'stack_y_beta')
stack(y_beta_dbeta, halo_catalog, 'stack_y_beta_dbeta')
