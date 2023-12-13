# CIB-deproj
Computation of the optimal beta value for CIB deprojection in Compton-y ILC maps  

## Requirements and Set-up
 - Requires a clone of the pyilc repository (https://github.com/jcolinhill/pyilc). 
 - Requires agora CIB maps at Planck frequency channels.  
 - Requires agora tSZ map.   
 - Requires halo catalogs from rot 1 to rot 30 (https://yomori.github.io/agora/index.html).  

## Running
Modify or create a yaml file similar to [example.yaml](example.yaml).  
```python main.py --config = example.yaml```   

## Dependencies
healpy  
pyyaml  
tqdm  
treecorr  
tqdm  
