import numpy as np

def get_planck_noise(inp):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications

    RETURNS
    -------
    PS_noise_Planck: (Nfreqs_Planck, ellmax+1) ndarray containing Planck noise power spectra in uK^2
        first dimension is of size 8 for frequencies 30, 44, 70, 100, 143, 217, 353, 545

    '''

    Nfreqs_Planck = 8
    freqs_Planck = []
    freqs_Planck.append('030')
    freqs_Planck.append('044')
    freqs_Planck.append('070')
    freqs_Planck.append('100')
    freqs_Planck.append('143')
    freqs_Planck.append('217')
    freqs_Planck.append('353')
    freqs_Planck.append('545')
    
    # Planck white noise
    noise_arr_Planck = np.zeros(Nfreqs_Planck)
    noise_arr_Planck[0] = 195.079975053 #uK-arcmin, from Table 7 (first column) of https://arxiv.org/pdf/1502.01585.pdf -- converted via sqrt(3224.4*(4*Pi*(180/Pi)^2*60^2/(12*1024^2)))
    noise_arr_Planck[1] = 226.090506617 # converted via sqrt(4331.0*(4*Pi*(180/Pi)^2*60^2/(12*1024^2)))
    noise_arr_Planck[2] = 199.09525581 # converted via sqrt(3358.5*(4*Pi*(180/Pi)^2*60^2/(12*1024^2))) assuming Nside=1024
    noise_arr_Planck[3] = 77.4 #uK-arcmin, from Table 6 of https://arxiv.org/pdf/1502.01587v2.pdf
    noise_arr_Planck[4] = 33.0
    noise_arr_Planck[5] = 46.8
    noise_arr_Planck[6] = 153.6
    noise_arr_Planck[7] = 0.78 * 1e-3 * 0.01723080316 * 1e6 * 60 #kJy/sr * deg --> converted to uK-arcmin

    # Planck beams
    FWHM_arr_Planck = np.zeros(Nfreqs_Planck)
    FWHM_arr_Planck[0] = 32.239
    FWHM_arr_Planck[1] = 27.005
    FWHM_arr_Planck[2] = 13.252
    FWHM_arr_Planck[3] = 9.69 #arcmin, from Table 6 of https://arxiv.org/pdf/1502.01587v2.pdf
    FWHM_arr_Planck[4] = 7.30
    FWHM_arr_Planck[5] = 5.02
    FWHM_arr_Planck[6] = 4.94
    FWHM_arr_Planck[7] = 4.83 #arcmin
    FWHM_arr_Planck *= inp.planck_beam_fraction #reduce beam FWHHM if desired

    # convert to sigma in radians
    sigma_arr_Planck = FWHM_arr_Planck / np.sqrt(8. * np.log(2)) /60. * np.pi/180.
    # Planck noise power spectra
    MAX_NOISE = 1.e9
    delta_ell = 1
    ell = np.arange(0, inp.ellmax+1, delta_ell)
    PS_noise_Planck = np.zeros((Nfreqs_Planck, int(inp.ellmax)+1))
    for i in range(Nfreqs_Planck):
        PS_noise_Planck[i] = (noise_arr_Planck[i] * (1.0/60.0) * (np.pi/180.0))**2.0 * np.exp( ell*(ell+1)* sigma_arr_Planck[i]**2. ) #square to get the white-noise level -- see e.g. Eq. 2.32 (pg 15) of https://arxiv.org/pdf/1509.06770v4.pdf
        # handle overflow due to high noise at high ell
        PS_noise_Planck[i][(np.where(PS_noise_Planck[i] > MAX_NOISE))[0]] = MAX_NOISE

    return PS_noise_Planck


def planck_MJystr_to_Kcmb(nu):
    # see Table 6 of https://arxiv.org/pdf/1303.5070.pdf
    if (nu == 545.0):
        return 1./58.04
    elif (nu == 857.0):
        return 1./2.27
    else:
        print("Planck map should already be in Kcmb")
        

def get_planck_specs():
    # returns frequencies in GHz, noise in K-arcminute, beam in arcmin
    frequencies = np.array([100, 143, 217, 353, 545, 857])
    beam = np.array([9.68, 7.30, 5.02, 4.94, 4.83, 4.64])
    # see https://arxiv.org/pdf/1502.01587.pdf, Table 6
    noise100 = 60. * 1.29e-6
    noise143 = 60. * 0.55e-6
    noise217 = 60. * 0.78e-6
    noise353 = 60. * 2.56e-6
    noise545 = 60. * (0.78 / 1000.) * planck_MJystr_to_Kcmb(545)
    noise857 = 60. * (0.72 / 1000.) * planck_MJystr_to_Kcmb(857)
    noise = np.array([noise100, noise143, noise217, noise353, noise545, noise857])
    return frequencies, noise, beam