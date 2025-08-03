import os
import shutil

from astropy.io import fits
from astropy.convolution import convolve_fft
import numpy as np

from PyZOGY.subtract import run_subtraction




def run_zogy(maindir, ref_name, sci_name, logger, 
             conv_ref=False, conv_sci=False, skysub=False, 
             use_var=True, match_gain=False, n_stamps=1, max_iter=5):
    fname_psf_r_in = maindir + 'psf/{}.psf.fits'.format(ref_name)
    fname_psf_s_in = maindir + 'psf/{}.psf.fits'.format(sci_name)
    
    fname_noise_r = maindir + 'noise/{}.noise.fits'.format(ref_name)
    fname_noise_s = maindir + 'noise/{}.noise.fits'.format(sci_name)
    
    if conv_ref:
        fname_ref = maindir + 'output/{}.crossconvd.fits'.format(ref_name)
    else:
        if skysub:
            fname_ref = maindir + 'input/{}.skysub.fits'.format(ref_name)
        else:
            fname_ref = maindir + 'input/{}.fits'.format(ref_name)


    if conv_sci:
        fname_sci = maindir + 'output/{}.crossconvd.fits'.format(sci_name)
    else:
        if skysub:
            fname_sci = maindir + 'input/{}.skysub.fits'.format(sci_name)
        else:
            fname_sci = maindir + 'input/{}.fits'.format(sci_name)    
            
    logger.info('Found REF image: {}'.format(fname_ref))
    logger.info('Found SCI image: {}'.format(fname_sci))    
    
    if conv_ref:
        logger.info('Getting PSF of REFxSCI image')
        
        with fits.open(fname_psf_r_in) as hdul:
            im_psf_r = hdul[0].data
            hdr_psf_r = hdul[0].header
        with fits.open(fname_psf_s_in) as hdul:
            im_psf_s = hdul[0].data
        
        im_psf_r_new = convolve_fft(im_psf_r, im_psf_s, boundary='fill', nan_treatment='fill', fill_value=0., normalize_kernel=True, allow_huge=True)

        fname_psf_r = maindir + 'psf/{}.psf.crossconvd.fits'.format(ref_name)
        fits.writeto(fname_psf_r, im_psf_r_new, hdr_psf_r, overwrite=True)
        
        logger.info('Wrote PSF image: {}'.format(fname_psf_r))
    else:
        fname_psf_r = fname_psf_r_in
        
    if conv_sci:
        logger.info('Getting PSF of SCIxREF image')
        
        with fits.open(fname_psf_r) as hdul:
            im_psf_r = hdul[0].data
            hdr_psf_r = hdul[0].header
        with fits.open(fname_psf_s_in) as hdul:
            im_psf_s = hdul[0].data
        
        im_psf_s_new = convolve_fft(im_psf_s, im_psf_r, boundary='fill', nan_treatment='fill', fill_value=0., normalize_kernel=True, allow_huge=True)

        fname_psf_s = maindir + 'psf/{}.psf.crossconvd.fits'.format(sci_name)
        fits.writeto(fname_psf_s, im_psf_s_new, hdr_psf_r, overwrite=True)
        
        logger.info('Wrote PSF image: {}'.format(fname_psf_s))
    else:
        fname_psf_s = fname_psf_s_in
        
    
    if use_var:
        #Get variance images
        fname_var_r = maindir + 'noise/{}.var.fits'.format(ref_name)
        with fits.open(fname_noise_r) as hdul:
            im_noise_r = hdul[0].data
            hdr_noise_r = hdul[0].header
            
        im_var_r = im_noise_r**2
        fits.writeto(fname_var_r, im_var_r, hdr_noise_r, overwrite=True)
        logger.info('Wrote REF variance image: {}'.format(fname_var_r))
        
        
        
        fname_var_s = maindir + 'noise/{}.var.fits'.format(sci_name)
        with fits.open(fname_noise_s) as hdul:
            im_noise_s = hdul[0].data
            hdr_noise_s = hdul[0].header
            
        im_var_s = im_noise_s**2
        fits.writeto(fname_var_s, im_var_s, hdr_noise_s, overwrite=True)
        logger.info('Wrote SCI variance image: {}'.format(fname_var_s))
    else:
        fname_var_s = None
        fname_var_r = None
        

    # #Euclid Q1
    # if sci_name == 'euclid_J':
    #     sci_sat = 181.53356517957647
    # if ref_name == 'euclid_J':
    #     ref_sat = 181.53356517957647
    # if sci_name == 'euclid_H':
    #     sci_sat = 199.04759608754375
    # if ref_name == 'euclid_H':
    #     ref_sat = 199.04759608754375
    # if sci_name == 'euclid_Y':
    #     sci_sat = 218.25134910470683
    # if ref_name == 'euclid_Y':
    #     ref_sat = 218.25134910470683

    # #NEXUS Epoch 01
    # if 'F115W' in ref_name:
    #     ref_sat = 11752.201074528331
    # if 'F115W' in sci_name:
    #     sci_sat = 11752.201074528331
    # if 'F150W' in ref_name:
    #     ref_sat = 11752.201074528331
    # if 'F150W' in sci_name:
    #     sci_sat = 11752.201074528331
    # if 'F090W' in ref_name:
    #     ref_sat = 11752.201074528331
    # if 'F090W' in sci_name:
    #     sci_sat = 11752.201074528331
    
    # ref_sat = 11752.201074528331
    # sci_sat = 11752.201074528331

    ref_sat = np.inf
    sci_sat = np.inf

    logger.info('Using REF saturation: {}'.format(ref_sat))
    logger.info('Using SCI saturation: {}'.format(sci_sat))
    
    if match_gain:
        gain_ratio = np.inf
    else:
        gain_ratio = 1.
        logger.info('Assuming gain ratio = 1. (equal zeropoints)')

    #ASSUMING GAIN_RATIO=ZP_RATIO=1
    logger.info('Starting ZOGY subtraction')
    output_fname = maindir + 'output/{}.zogydiff.fits'.format(sci_name)
    
    try:
        im_diff, im_psf_diff = run_subtraction(fname_sci, fname_ref, fname_psf_s, fname_psf_r, output=output_fname,
                                            science_variance=fname_var_s, reference_variance=fname_var_r,
                                            science_saturation=sci_sat, reference_saturation=ref_sat,
                                            gain_ratio=gain_ratio, n_stamps=n_stamps,
                                            #    matched_filter=True, corrected=True,
                                            max_iterations=max_iter)
        logger.info('Wrote ZOGY output image: {}'.format(output_fname))

    except Exception as e:
        logger.error('ZOGY subtraction failed: {}'.format(e))
        logger.error('Saving empty file')

        with fits.open(fname_sci) as hdul:
            hdr = hdul[0].header
            im_diff = np.zeros(hdul[0].data.shape, dtype=np.float32)
            im_psf_diff = np.zeros(hdul[0].data.shape, dtype=np.float32)
            
        fits.writeto(output_fname, im_diff, hdr, overwrite=True)
        
        output_psf_fname = maindir + 'output/{}.zogydiff.psf.fits'.format(sci_name)
        fits.writeto(output_psf_fname, im_psf_diff, overwrite=True)
        


    return

