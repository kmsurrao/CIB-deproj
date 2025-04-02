# script for stampede
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pickle
import h5py
import concurrent.futures

nside = 1024


base_dir = '/scratch/09334/ksurrao/cib_deproj_040125/planck1p0_highz_decorr_alpha1_bin200'
h = hp.read_map(f'{base_dir}/maps/halo.fits') 
y_beta = hp.read_map(f'{base_dir}/pyilc_outputs/final/needletILCmap_component_tSZ_deproject_CIB_realistic.fits')
y_beta_dbeta = hp.read_map(f'{base_dir}/pyilc_outputs/final/needletILCmap_component_tSZ_deproject_CIB_CIB_dbeta.fits')
ytrue = hp.ud_grade(hp.read_map('/scratch/09334/ksurrao/agora_planck/tsz_2048.fits'), nside)
halo_catalog = '/scratch/09334/ksurrao/halo_files/haloslc_agora_zsel_0.8_to_1.8_Msel_1e12_to_1e15_rot120.h5'
odir = '/work2/09334/ksurrao/stampede3/Github/CIB-deproj/stacks'

# Smooth the maps (10 arcmin FWHM)
fwhm_rad = np.radians(10/60)
h = hp.smoothing(h, fwhm=fwhm_rad)
y_beta = hp.smoothing(y_beta, fwhm=fwhm_rad)
y_beta_dbeta = hp.smoothing(y_beta_dbeta, fwhm=fwhm_rad)
ytrue = hp.smoothing(ytrue, fwhm=fwhm_rad)

def process_chunk(chunk_indices, theta_halos, phi_halos, y_map, xsize, reso):
    sum_patch = np.zeros((xsize, xsize))
    for i in chunk_indices:
        theta = theta_halos[i]
        phi = phi_halos[i]
        lat_deg = 90.0 - np.degrees(theta)
        lon_deg = np.degrees(phi)
        patch = hp.gnomview(y_map, rot=(lon_deg, lat_deg), xsize=xsize, reso=reso,
                            return_projected_map=True, no_plot=True)
        sum_patch += patch
    return sum_patch, len(chunk_indices)


def stack(y_map, halo_catalog, name, num_workers=8, frac=0.001):
    """
    Stack patches from the halo map using a random fraction (frac) of the halos.
    frac=0.001 uses 0.1% of the halos.
    """
    
    reso = 3.44  # arcmin per pixel
    patch_size_deg = 2.0  # patch size in degrees
    xsize = int((patch_size_deg * 60) / reso)  # ~35 pixels for a 2° patch

    with h5py.File(halo_catalog, 'r') as df:
        ra = np.array(df.get('ra')[()])
        dec = np.array(df.get('dec')[()])

    theta_halos = np.radians(90.0 - dec)
    phi_halos = np.radians(ra)

    num_halos = len(theta_halos)
    print("Original number of halos:", num_halos)
    
    # Subsample: randomly choose a fraction of halos
    np.random.seed(0)  # for reproducibility
    all_indices = np.arange(num_halos)
    subsample_indices = np.random.choice(all_indices, size=int(frac * num_halos), replace=False)
    subsample_indices.sort()
    
    print("Using", len(subsample_indices), "halos after subsampling.")

    # Split indices into chunks for parallel processing
    chunks = np.array_split(subsample_indices, num_workers)
    sum_total = np.zeros((xsize, xsize))
    total_count = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_chunk, chunk, theta_halos, phi_halos, y_map, xsize, reso)
                   for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            chunk_sum, count = future.result()
            sum_total += chunk_sum
            total_count += count
    
    stacked_patch_2d = sum_total / total_count
    pickle.dump(stacked_patch_2d, open(f'{odir}/{name}.p', 'wb'))
    return stacked_patch_2d

# Example calls with subsampling (adjust frac as needed)
stack(h, halo_catalog, 'stack_h')
stack(ytrue, halo_catalog, 'stack_ytrue')
stack(y_beta, halo_catalog, 'stack_y_beta')
stack(y_beta_dbeta, halo_catalog, 'stack_y_beta_dbeta')