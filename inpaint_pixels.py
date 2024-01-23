import numpy as np
import healpy as hp
import diffusive_inpaint
from conversions import *

def inpaint_map(map_to_inpaint, mask):
    '''
    ARGUMENTS
    ---------
    map_to_inpaint: numpy array of map in healpix format, should be in Jy/sr
    mask: numpy array in healpix RING format containing zeros for masked pixels and ones elsewhere
    
    RETURNS
    -------
    inpainted_map: numpy array containing map in healpix format
    '''
    MASK_VAL = -1.e30
    if len(map_to_inpaint)==3:
        for i in range(3):
            map_to_inpaint[i] = hp.remove_monopole(map_to_inpaint[i])
    else:
        map_to_inpaint = hp.remove_monopole(map_to_inpaint)
    map_raw = map_to_inpaint.copy()
    map_raw[np.where(mask == 0.)] = MASK_VAL
    if len(map_to_inpaint)==3:
        inpainted_map = np.zeros_like(map_to_inpaint)
        for i in range(3):
            inpainted_map[i] = diffusive_inpaint.diff_inpaint_vectorized(map_raw[i], MASK_VAL=MASK_VAL)
    else:
        inpainted_map = diffusive_inpaint.diff_inpaint_vectorized(map_raw, MASK_VAL=MASK_VAL)
    return inpainted_map


def initial_masking(inp, map_, map_150):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    map_: numpy array of map to inpaint in healpix format
    map_150: numpy array of 150 GHz CIB healpix map. 
        This is the map from which to create the flux cut mask.
    
    RETURNS
    -------
    inpainted_map: numpy array, either [I,Q,U] healpix maps if pol, or just one intensity healpix map if not pol
        should be in Jy/sr
        Very bright values in the input map are inpainted
    '''
    mask = np.ones_like(map_)
    cut_val = JytoK(150, inp.nside)*0.1*10**6
    mask[map_150 >= cut_val] = 0 #mask pixels >= 100 mJy at 150 GHz
    neighbors = hp.pixelfunc.get_all_neighbours(inp.nside, np.where(mask==0)).flatten()
    mask[neighbors] = 0
    inpainted_map = inpaint_map(map_, mask)
    return inpainted_map