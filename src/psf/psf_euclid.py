import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.nddata.bitmask import BitFlagNameMap, bitfield_to_boolean_mask
from tqdm import tqdm

class EUCLID_FLAG(BitFlagNameMap):
    INVALID = 2**0
    OBMASK = 2**1           ###### Ignored
    DISCONNECTED = 2**2
    ZEROQE = 2**3
    BADBASE = 2**4
    LOWQE = 2**5            ######
    SUPERQE = 2**6
    HOT = 2**7              ######
    RTN = 2**8
    SNOWBALL = 2**9
    SATUR = 2**10
    NLINEAR = 2**11         ######
    NLMODFAIL = 2**12
    PERSIST = 2**13
    PERMODFAIL = 2**14      ######
    DARKNODET = 2**15       ######
    COSMIC = 2**16
    FLATLH = 2**17          ######
    GHOST = 2**18
    SCATTER = 2**19         ######
    MOVING = 2**20          ######
    TRANS = 2**21           ######
    CROSSTALK = 2**22       ######
    FLOWER = 2**23
    VIGNET = 2**24          ######
    


def get_psf_indiv(fname_psf_grid):    
    hdul = fits.open(fname_psf_grid)
    psfdat = Table(hdul[2].data)

    if 'x_center' in psfdat.colnames:
        xcol = 'x_center'
        ycol = 'y_center'
    else:
        xcol = 'x'
        ycol = 'y'


    xcents = np.sort(  np.unique(psfdat[xcol].data) )
    ycents = np.sort(  np.unique(psfdat[ycol].data) )

    dx = int( np.diff(xcents)[0] )
    dy = int( np.diff(ycents)[0] )




    psf_all = np.zeros((len(psfdat), dx, dy))
    N0, N1 = hdul[1].data.shape

    bad_ind = []
    bad_x = []
    bad_y = []
    for i in range(len(psfdat)):
        xc = psfdat[xcol][i].astype(int)
        yc = psfdat[ycol][i].astype(int)
        if xcol == 'x':
            xc -= 1
            yc -= 1
        
        x1 = int(xc-dx/2)
        x2 = int(xc+dx/2)
        y1 = int(yc-dy/2)
        y2 = int(yc+dy/2) 
        psf_all[i] = hdul[1].data[x1:x2,y1:y2]
        
    return psf_all, psfdat['FWHM'].data


def get_psf_all(fnames, output_fname):
    
    psf_all = []
    fwhm_all = []
    for fname in fnames:
        psf, fwhm = get_psf_indiv(fname)
        psf_all.append(psf)
        fwhm_all.append(fwhm)
        
    print('\t Combining PSF grids')
    psf_all = np.concatenate(psf_all, axis=0)
    fwhm_all = np.concatenate(fwhm_all)    
    
    print('\t Getting median PSF')
    p16, med, p84 = np.percentile(psf_all, [16, 50, 84], axis=0) 
    psf_tot = med.copy()
    psf_err = ( (p84-med) + (med-p16) )/2

    print('\t Getting median FWHM')
    p16, med, p84 = np.percentile(fwhm_all, [16, 50, 84])
    fwhm_tot = med
    fwhm_err = ( (p84-med) + (med-p16) )/2
    
    print('\t Saving')
    hdu0 = fits.PrimaryHDU()
    hdu0.header['FWHM'] = fwhm_tot
    hdu0.header['FWHM_ERR'] = fwhm_err
    hdu0.header['FWHM_UNIT'] = 'px'
    hdu0.header['FWHM_ERR_UNIT'] = 'px'
    
    hdu1 = fits.ImageHDU(psf_tot, name='PSF')
    hdu1.header['COMMENT'] = 'Median PSF from PSF grid'
    
    hdu2 = fits.ImageHDU(psf_err, name='PSF_ERR')
    hdu2.header['COMMENT'] = '1-sigma error on median PSF'
    
    hdul_out = fits.HDUList([hdu0, hdu1, hdu2])
    hdul_out.writeto(output_fname, overwrite=True)    
    
    return
    
    



def get_mask_image(fname_flag):
    im_dat = fits.open(fname_flag)[0].data
    
    ignore_flags = 'OBMASK,LOWQE,HOT,NLINEAR,PERMODFAIL,DARKNODET,FLATLH,SCATTER,MOVING,TRANS,CROSSTALK,VIGNET'
    bad_flag_mask = bitfield_to_boolean_mask(im_dat, ignore_flags=ignore_flags, 
                                           flag_name_map=EUCLID_FLAG, dtype=bool)
    
    return bad_flag_mask
