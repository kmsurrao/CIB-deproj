# CIB-deproj
Computation of the optimal SED for CIB deprojection for enhancing signal-to-noise in tSZ cross-correlations. If you use the code in a publication, please cite https://arxiv.org/abs/2505.14644.  

## Requirements and Set-up
 - Requires a clone of the pyilc repository (https://github.com/jcolinhill/pyilc).  
 - Requires agora CIB maps at Planck frequency channels. All CIB maps should be stored in the cib_map_dir directory that's provided in the yaml.    
 - Requires agora tSZ map.   
 - Requires halo catalogs from rot 1 to rot 30 (https://yomori.github.io/agora/index.html) provided either as a set of files in halo_files_dir or a single catalog in halo_catalog in example.yaml.    

## Running
Modify or create a yaml file similar to [example.yaml](example.yaml).  
```python main.py --config = example.yaml```   

## Outputs
- The main product is the final harmonic ILC $y$-map. This is saved in OUTPUT_DIR/pyilc_outputs/final/needletILCmap_component_tSZ_deproject_CIB_realistic.fits. A harmonic ILC $y$-map built using moment deprojection is also saved in OUTPUT_DIR/pyilc_outputs/needletILCmap_component_tSZ_deproject_CIB_CIB_dbeta.fits.  
- Other intermediate outputs include the following (see [paper_plots.ipynb](paper_plots.ipynb) for an example of how to use these products):  
    - Frequency maps $T(\mathbf{\hat{n}})$ in ```OUTPUT_DIR/maps/uninflated_{freq}.fits``` and frequency maps $T \text{ } \prime (\mathbf{\hat{n}})$ constructed using each $\beta$ value in ```OUTPUT_DIR/maps/inflated_{freq}_{beta}.fits```.  
    - Maps $y^{\beta}$ in ```OUTPUT_DIR/pyilc_outputs/beta_{beta}_uninflated/needletILCmap_component_tSZ_deproject_CIB.fits``` and $y^{\beta}_{\alpha}$ in ```OUTPUT_DIR/pyilc_outputs/beta_{beta}_inflated/needletILCmap_component_tSZ_deproject_CIB.fits```.    
    - Pickle file containing array of beta values in ```OUTPUT_DIR/pickle_files/beta_arr.p```.  
    - Pickle files ```OUTPUT_DIR/pickle_files/chi2_true_arr.p``` and ```OUTPUT_DIR/pickle_files/chi2_infl_arr.p``` for the idealized and realistic pipelines, repsectively. Each file contains an array of shape (Nbins, Nbeta) giving the $\chi^2$ of each $\beta$ value in each bin.  
    - Pickle file containing best fit beta values in each multipole bin in ```OUTPUT_DIR/pickle_files/beta_points_per_ell.p```. Specifically, the file is a list containing 6 arrays of length Nbins: [means_true, uppers_true, lowers_true, means_infl, uppers_infl, lowers_infl]. Here means_true are the central values of beta per bin using the idealized pipeline, while uppers_true and lowers_true are the upper and lower error bars, respectively. means_infl, uppers_infl, lowers_infl are the same for the realistic pipeline.    
    


## Dependencies
healpy  
pyyaml  
tqdm  
