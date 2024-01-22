import numpy as np

def dBnudT(nu_ghz):
    '''
    function from pyilc

    ARGUMENTS
    ---------
    nu_ghz: float, frequency in GHz

    RETURNS
    -------
    float, blackbody derivative needed for Planck bandpass integration/conversion 
        following approach in Sec. 3.2 of https://arxiv.org/pdf/1303.5070.pdf
        units are 1e-26 Jy/sr/uK_CMB
    '''
    TCMB = 2.726 #Kelvin
    TCMB_uK = 2.726e6 #micro-Kelvin
    hplanck=6.626068e-34 #MKS
    kboltz=1.3806503e-23 #MKS
    clight=299792458.0 #MKS
    nu = 1.e9*np.asarray(nu_ghz)
    X = hplanck*nu/(kboltz*TCMB)
    return (2.*hplanck*nu**3.)/clight**2. * (np.exp(X))/(np.exp(X)-1.)**2. * X/TCMB_uK


def ItoDeltaT(nu_ghz):
    '''
    function from pyilc

    ARGUMENTS
    ---------
    nu_ghz: float, frequency in GHz

    RETURNS
    -------
    float, conversion factor from specific intensity to Delta T units (i.e., 1/dBdT|T_CMB)
      i.e., from W/m^2/Hz/sr (1e-26 Jy/sr) --> uK_CMB
      i.e., you would multiply a map in 1e-26 Jy/sr by this factor to get an output map in uK_CMB
    '''
    return 1./dBnudT(nu_ghz)


def JysrtoK(nu_ghz):
    '''
    ARGUMENTS
    ---------
    nu_ghz: float, frequency in GHz

    RETURNS
    -------
    float, conversion factor from Jy/sr to K
    '''
    ItoDeltaT_conversion = ItoDeltaT(nu_ghz)
    return ItoDeltaT_conversion*1e-26*1e-6
    
def KtoJysr(nu_ghz):
    '''
    ARGUMENTS
    ---------
    nu_ghz: float, frequency in GHz

    RETURNS
    -------
    float, conversion factor from K to Jy/sr
    '''
    return 1/JysrtoK(nu_ghz)

def KtoJy(nu_ghz, nside):
    '''
    ARGUMENTS
    ---------
    nu_ghz: float, frequency in GHz
    nside: int, resolution parameter for map under consideration

    RETURNS
    -------
    float, conversion factor from K to Jy 
        (multiplies K to Jy/sr conversion by pixel window, i.e. 4pi/Npix)
    '''
    return KtoJysr(nu_ghz)*(4*np.pi/(12*nside**2))

def JytoK(nu_ghz, nside):
    '''
    ARGUMENTS
    ---------
    nu_ghz: float, frequency in GHz
    nside: int, resolution parameter for map under consideration

    RETURNS
    -------
    float, conversion factor from Jy to K 
        (divides Jy/sr to K conversion by pixel window, i.e. 4pi/Npix)
    '''
    return JysrtoK(nu_ghz)/(4*np.pi/(12*nside**2))