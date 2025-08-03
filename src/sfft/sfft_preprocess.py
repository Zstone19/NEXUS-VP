import os
import subprocess
from astropy.table import Table
from astropy.io import fits

from astropy.coordinates import SkyCoord
from astropy import units as u

import numpy as np
from astropy.convolution import convolve_fft

from skysub import SEx_SkySubtract

##########################################################################################################################################
##########################################################################################################################################

# def align(fname_obj, fname_ref, fname_out):
    
#     #OBJ, REF images should be sky-subtracted
#     #The OBJ image will be resampled onto the REF frame
    
#     PY_SWarp.PS(FITS_obj=fname_obj, FITS_ref=fname_ref, FITS_resamp=fname_out,
#             GAIN_KEY='GAIN', SATUR_KEY='ESATUR', OVERSAMPLING=1, RESAMPLING_TYPE='LANCZOS3',
#             SUBTRACT_BACK='N', FILL_VALUE=np.nan, VERBOSE_TYPE='NORMAL', VERBOSE_LEVEL=1)
    
#     return

##########################################################################################################################################
##########################################################################################################################################

def cross_conv(maindir, ref_name, sci_name, conv_ref, conv_sci, skysub, logger):
    indir = maindir + 'input/'
    psfdir = maindir + 'psf/'
    outdir = maindir + 'output/'
    
    if skysub:
        fname_lref = indir + '{}.skysub.fits'.format(ref_name)
        fname_lsci = indir + '{}.skysub.fits'.format(sci_name)                
    else:
        fname_lref = indir + '{}.fits'.format(ref_name)
        fname_lsci = indir + '{}.fits'.format(sci_name)
    
    
    fname_psf_lref = psfdir + '{}.psf.fits'.format(ref_name)
    fname_psf_lsci = psfdir + '{}.psf.fits'.format(sci_name)
    
    fname_mask_lref = indir + '{}.maskin.fits'.format(ref_name)
    fname_mask_lsci = indir + '{}.maskin.fits'.format(sci_name)
    
    ##############################################################
    #Load data
    
    logger.info('Loading data')
    im_lref = fits.getdata(fname_lref, ext=0).T
    im_lsci = fits.getdata(fname_lsci, ext=0).T
    
    logger.info('Loading PSFs')
    im_psf_lref = fits.getdata(fname_psf_lref, ext=0).T
    im_psf_lsci = fits.getdata(fname_psf_lsci, ext=0).T
    
    logger.info('Loading masks')
    im_mask_lref = fits.getdata(fname_mask_lref, ext=0).T.astype(bool)
    im_mask_lsci = fits.getdata(fname_mask_lsci, ext=0).T.astype(bool)
    im_mask = im_mask_lref | im_mask_lsci
    
    #Make sure the PSFs are normalized
    im_psf_lref /= np.sum(im_psf_lref)
    im_psf_lsci /= np.sum(im_psf_lsci)

    ##############################################################
    # Do ref x sci_psf
    
    if conv_ref:    
        fname_lref_convd = outdir + '{}.crossconvd.fits'.format(ref_name)
        
        if not os.path.exists(fname_lref_convd):    
            logger.info('Convolving REF x SCI PSF')
            im_lref_convd = convolve_fft(im_lref, im_psf_lsci, boundary='fill', nan_treatment='fill', fill_value=0., normalize_kernel=True, allow_huge=True)
            im_lref_convd[im_mask] = 0.
            
            with fits.open(fname_lref) as hdul:
                mssg = 'Convolved {} with {} PSF'.format(ref_name, sci_name)
                logger.info('MelOn CheckPoint: {}'.format(mssg))
                
                hdul[0].data[:,:] = im_lref_convd.T
                hdul.writeto(fname_lref_convd, overwrite=True)
                
        else:
            logger.info('Found REF x SCI PSF convolution, skipping')
        
        
    ##############################################################
    # Do sci x ref_psf
    
    if conv_sci:
        fname_lsci_convd = outdir + '{}.crossconvd.fits'.format(sci_name)
        
        if not os.path.exists(fname_lsci_convd):    
            logger.info('Convolving SCI x REF PSF')
            im_lsci_convd = convolve_fft(im_lsci, im_psf_lref, boundary='fill', nan_treatment='fill', fill_value=0., normalize_kernel=True, allow_huge=True)
            im_lsci_convd[im_mask] = 0.
            
            with fits.open(fname_lsci) as hdul:
                mssg = 'Convolved {} with {} PSF'.format(sci_name, ref_name)
                logger.info('MelOn CheckPoint: {}'.format(mssg))
                
                hdul[0].data[:,:] = im_lsci_convd.T
                hdul.writeto(fname_lsci_convd, overwrite=True)
                
        else:
            logger.info('Found SCI x REF PSF convolution, skipping')
        
    return


##########################################################################################################################################
##########################################################################################################################################

def subtract_sky(maindir, ref_name, sci_name, logger, mask_type='sextractor'):    
    
    indir = maindir + '/input/'
    outdir = maindir + '/input/'

    fname_ref = indir + '{}.fits'.format(ref_name)
    fname_sci = indir + '{}.fits'.format(sci_name)
    
    fname_out_ref = outdir + '{}.skysub.fits'.format(ref_name)
    fname_out_sci = outdir + '{}.skysub.fits'.format(sci_name)
    
    if os.path.exists(fname_out_ref) and os.path.exists(fname_out_sci):
        logger.info('Found sky-subtracted REF/SCI images, skipping')
        return
    else:
        
        logger.info('Subtracting sky from input images')

    #REF
    SEx_SkySubtract.SSS(FITS_obj=fname_ref, FITS_skysub=fname_out_ref, mask_type=mask_type, FITS_sky=None, FITS_skyrms=None,
                        SATUR_KEY='SATURATE', ESATUR_KEY='ESATUR', BACK_SIZE=64, BACK_FILTERSIZE=3,
                        DETECT_THRESH=1.5, DETECT_MINAREA=5, DETECT_MAXAREA=0, VERBOSE_LEVEL=2)
    logger.info('\t REF image complete')
    
    #SCI
    SEx_SkySubtract.SSS(FITS_obj=fname_sci, FITS_skysub=fname_out_sci, mask_type=mask_type, FITS_sky=None, FITS_skyrms=None,
                        SATUR_KEY='SATURATE', ESATUR_KEY='ESATUR', BACK_SIZE=64, BACK_FILTERSIZE=3,
                        DETECT_THRESH=1.5, DETECT_MINAREA=5, DETECT_MAXAREA=0, VERBOSE_LEVEL=2)
    logger.info('\t SCI image complete')
    
    
    
    #Mask again
    fname_mref = indir + '{}.maskin.fits'.format(ref_name)
    fname_msci = indir + '{}.maskin.fits'.format(sci_name)
    with fits.open(fname_mref) as hdul:
        im_mref = hdul[0].data.astype(bool)
    with fits.open(fname_msci) as hdul:
        im_msci = hdul[0].data.astype(bool)
        
    im_mask = im_mref | im_msci
    
    with fits.open(fname_out_ref) as hdul:
        im_ref = hdul[0].data.copy()
        im_ref[im_mask] = 0.
        hdul[0].data[:,:] = im_ref
        hdul.writeto(fname_out_ref, overwrite=True)
    with fits.open(fname_out_sci) as hdul:
        im_sci = hdul[0].data.copy()
        im_sci[im_mask] = 0.
        hdul[0].data[:,:] = im_sci
        hdul.writeto(fname_out_sci, overwrite=True)
    
    logger.info('Sky subtraction complete')
    return

##########################################################################################################################################
##########################################################################################################################################
