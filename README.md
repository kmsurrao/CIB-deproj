# CIB-deproj
Computation of the optimal beta value for CIB deprojection in Compton-y ILC maps  

## Requirements and Set-up
 - Requires a clone of the pyilc repository (https://github.com/jcolinhill/pyilc).  
 - Requires agora CIB maps at Planck frequency channels as well as a 150 GHz CIB map to use for determining outlier pixels to inpaint (for example, the ACT map). All CIB maps should be stored in the cib_map_dir directory that's provided in the yaml.    
 - Requires agora tSZ map.   
 - Requires halo catalogs from rot 1 to rot 30 (https://yomori.github.io/agora/index.html) provided either as a set of files in halo_files_dir or a single catalog in halo_catalog in example.yaml.    

## Running
Modify or create a yaml file similar to [example.yaml](example.yaml).  
```python main.py --config = example.yaml```   

## Dependencies
healpy  
pyyaml  
tqdm  
treecorr  
