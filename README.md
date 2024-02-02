# CIB-deproj
Computation of the optimal beta value for CIB deprojection in Compton-y ILC maps  

## Requirements and Set-up
 - Requires a clone of the pyilc repository (https://github.com/jcolinhill/pyilc).  
 - Requires agora CIB maps at Planck frequency channels. All CIB maps should be stored in the cib_map_dir directory that's provided in the yaml.    
 - Requires agora tSZ map.   
 - Requires halo catalogs from rot 1 to rot 30 (https://yomori.github.io/agora/index.html) provided either as a set of files in halo_files_dir or a single catalog in halo_catalog in example.yaml.    

## Running
Modify or create a yaml file similar to [example.yaml](example.yaml).  
```python main.py --config = example.yaml```   

## Outputs
- The code prints out the $\chi^2$ arrays. The $\chi^2$ plot is also saved in output_dir, defined in the yaml file.  
- Correlation plots for individual $\beta$ values are saved in output_dir.  
- Pickle files for individual $\beta$ values are also saved in output_dir. 
    - If running a harmonic space pipeline, each pickle file contains [mean_ells, to_dl, hy_true, hy_infl, yy_true, yy_infl, hh]. mean_ells are the mean multipole values for each bin. to_dl is a $C_\ell$ to $D_\ell$ conversion factor using the mean multipoles in each bin. hy_true is the cross-spectrum $C_\ell^{h , (y_{\mathrm{recon.}}-y_{\mathrm{true}})}$. hy_infl is the cross-spectrum $C_\ell^{h , (y_{\mathrm{recon.}}-y_{\mathrm{recon. (inflated CIB)}})}$. yy_true is the auto-spectrum $C_\ell^{(y_{\mathrm{recon.}}-y_{\mathrm{true}}) , (y_{\mathrm{recon.}}-y_{\mathrm{true}})}$. yy_infl is the auto-spectrum of $C_\ell^{(y_{\mathrm{recon.}}-y_{\mathrm{recon. (inflated CIB)}}) , (y_{\mathrm{recon.}}-y_{\mathrm{recon. (inflated CIB)}})}$. hh is the auto-spectrum $C_\ell^{hh}$.
    - If running a real space pipeline, each pickle file contains [hy_true, hy_infl, cov_hytrue, cov_hyinfl, r_hy]. hy_true is the cross-correlation of $h$ with $(y_\mathrm{recon.}-y_{\mathrm{true}})$. hy_infl is the cross-correlation of $h$ with $(y_\mathrm{recon.}-y_{\mathrm{recon. (inflated CIB)}})$. cov_hytrue is the covariance matrix of hy_true. cov_hyinfl is the covariance matrix of hy_infl. r_hy is an array of bins for the separation range.  

## Dependencies
healpy  
pyyaml  
tqdm  
treecorr  
