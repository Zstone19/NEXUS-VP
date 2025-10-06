import os

from astropy.io import fits
import numpy as np


names = ['wide', 'deep']
epochs = ['01', '03']


for jwst_band in ['F200W', 'F444W']:
    print(jwst_band)
    
    #Get directories, make output dirs
    print('\t Make directories')
    
    maindir = '/data6/stone28/nexus/NEXUS/cropped_{}{}_{}{}/'.format(names[0], epochs[0], names[1], epochs[1]) + jwst_band.lower() + '/'
    maindir_og = '/data3/web/nexus_collab/nircam/Deep_epoch/'
    prefix = ''

    if jwst_band == 'F444W':
        prefix = '_60mas'

    outdir = '/data6/stone28/nexus/zogy_nexus_{}{}_{}{}_{}/'.format(names[0], epochs[0], names[1], epochs[1], jwst_band)
    os.makedirs(outdir, exist_ok=True)
    for subdir in ['input/', 'output/', 'noise/', 'psf/']:
        os.makedirs(outdir + subdir, exist_ok=True)
        
    ############################################################################################
    #Write data/noise/mask files for ZOGY
    print('\t Write files')
        
    #Copy data
    fname_mask = maindir + 'nexus_{}_ep{}_{}_mask.shift.fits'.format(names[0], epochs[0], jwst_band.lower())
    with fits.open(fname_mask) as hdul:
        im_mask1 = hdul[0].data.astype(bool)
        
    fname_mask = maindir + 'nexus_{}_ep{}_{}_mask.shift.fits'.format(names[1], epochs[1], jwst_band.lower())
    with fits.open(fname_mask) as hdul:
        im_mask2 = hdul[0].data.astype(bool)

    im_mask = im_mask1 | im_mask2



    for suffix in ['data', 'error', 'mask']:   
        if suffix in ['data', 'error']:
            dtype_out = np.float32
        else:
            dtype_out = np.int16
        
        fname_og = maindir_og + 'nexus_central_{}_ep{}_{}{}_i2d_{}.fits'.format(names[0], epochs[0], jwst_band.lower(), prefix, suffix)
        
        if suffix == 'error':
            suffix2 = 'err'
        else:
            suffix2 = suffix
        fname = maindir + 'nexus_{}_ep{}_{}_{}.shift.fits'.format(names[0], epochs[0], jwst_band.lower(), suffix2)
        
        if suffix == 'data':
            with fits.open(fname_og) as hdul:
                hdr_og = hdul[0].header
        
        with fits.open(fname) as hdul:
                    
            hdul[0].header['MAGZERO'] = 23.90
            hdul[0].header['BUNIT'] = 'uJy'
                
            if suffix == 'data':
                fname_out = outdir + 'input/nexus_{}{}_{}.fits'.format(names[0], epochs[0], jwst_band)
            if suffix == 'error':
                fname_out = outdir + 'noise/nexus_{}{}_{}.noise.fits'.format(names[0], epochs[0], jwst_band)
            if suffix == 'mask':
                fname_out = outdir + 'input/nexus_{}{}_{}.maskin.fits'.format(names[0], epochs[0], jwst_band)

            
            zp_new = hdr_og['MAG_ZP']
            factor = 10**( (8.9 - zp_new) / 2.5 ) / 1e-6
            if suffix in ['data', 'error']: 
                hdul[0].data[im_mask] = 0.
                hdul[0].data[ np.isnan(hdul[0].data) ] = 0.
                hdul[0].data = hdul[0].data * factor

            hdul[0].data = hdul[0].data.astype(dtype_out)
            hdul.writeto(fname_out, overwrite=True)
            
            
            
        if names[1] + epochs[1] == 'deep03':
            prefix_i = '_006_60mas'
        else:
            prefix_i = prefix
            
        fname_og = maindir_og + 'nexus_central_{}_ep{}_{}{}_i2d_{}.fits'.format(names[1], epochs[1], jwst_band.lower(), prefix_i, suffix)
        fname = maindir + 'nexus_{}_ep{}_{}_{}.shift.fits'.format(names[1], epochs[1], jwst_band.lower(), suffix2)

        if suffix == 'data':
            with fits.open(fname_og) as hdul:
                hdr_og = hdul[0].header
        
        with fits.open(fname) as hdul:
                    
            hdul[0].header['MAGZERO'] = 23.90
            hdul[0].header['BUNIT'] = 'uJy'
                
            if suffix == 'data':
                fname_out = outdir + 'input/nexus_{}{}_{}.fits'.format(names[1], epochs[1], jwst_band)
            if suffix == 'error':
                fname_out = outdir + 'noise/nexus_{}{}_{}.noise.fits'.format(names[1], epochs[1], jwst_band)
            if suffix == 'mask':
                fname_out = outdir + 'input/nexus_{}{}_{}.maskin.fits'.format(names[1], epochs[1], jwst_band)

            
            zp_new = hdr_og['MAG_ZP']
            factor = 10**( (8.9 - zp_new) / 2.5 ) / 1e-6
            if suffix in ['data', 'error']: 
                hdul[0].data[im_mask] = 0.
                hdul[0].data[ np.isnan(hdul[0].data) ] = 0.
                hdul[0].data = hdul[0].data * factor

            hdul[0].data = hdul[0].data.astype(dtype_out)
            hdul.writeto(fname_out, overwrite=True)



    #print('JWST: 1 DN = ', hdr_og['PHOTMJSR'] * (factor)/hdr_og['EXPTIME'] / 1e-5, 'e-05 uJy')

    ############################################################################################
    # Copy PSF (WebbPSF)
    print('\t Write PSFs')

    fname = '/data6/stone28/nexus/NEXUS/webbpsf/final/nexus_{}{}_{}.psf.fits'.format(names[0], epochs[0], jwst_band.lower())
    with fits.open(fname) as hdul:
        hdr = hdul[0].header
        im = hdul[0].data

    # im_out = im[400:-400, 400:-400].copy()
    fits.writeto(outdir + 'psf/nexus_{}{}_{}.psf.fits'.format(names[0], epochs[0], jwst_band), im, hdr, overwrite=True)


    fname = '/data6/stone28/nexus/NEXUS/webbpsf/final/nexus_{}{}_{}.psf.fits'.format(names[1], epochs[1], jwst_band.lower())
    with fits.open(fname) as hdul:
        hdr = hdul[0].header
        im = hdul[0].data
        
    # im_out = im[400:-400, 400:-400].copy()
    fits.writeto(outdir + 'psf/nexus_{}{}_{}.psf.fits'.format(names[1], epochs[1], jwst_band), im, hdr, overwrite=True)
    
    ############################################################################################
    # Write subdiff
    print('\t Write subtracted image')
    
    fname_mr = outdir + 'input/nexus_{}{}_{}.maskin.fits'.format(names[0], epochs[0], jwst_band)
    fname_ms = outdir + 'input/nexus_{}{}_{}.maskin.fits'.format(names[1], epochs[1], jwst_band)
    fname_r = outdir + 'input/nexus_{}{}_{}.fits'.format(names[0], epochs[0], jwst_band)
    fname_s = outdir + 'input/nexus_{}{}_{}.fits'.format(names[1], epochs[1], jwst_band)
    fname_nr = outdir + 'noise/nexus_{}{}_{}.noise.fits'.format(names[0], epochs[0], jwst_band)
    fname_ns = outdir + 'noise/nexus_{}{}_{}.noise.fits'.format(names[1], epochs[1], jwst_band)

    with fits.open(fname_mr) as hdul:
        hdr_mr = hdul[0].header
        im_mask_r = hdul[0].data.astype(bool)
    with fits.open(fname_ms) as hdul:
        hdr_ms = hdul[0].header
        im_mask_s = hdul[0].data.astype(bool)
    with fits.open(fname_r) as hdul:
        hdr_r = hdul[0].header
        im_r = hdul[0].data.copy()
    with fits.open(fname_s) as hdul:
        hdr_s = hdul[0].header
        im_s = hdul[0].data.copy()
    with fits.open(fname_nr) as hdul:
        hdr_nr = hdul[0].header
        im_nr = hdul[0].data.copy()
    with fits.open(fname_ns) as hdul:
        hdr_ns = hdul[0].header
        im_ns = hdul[0].data.copy()
        
        
    im_mask = im_mask_r | im_mask_s | np.isnan(im_r) | np.isnan(im_s) | (im_r == 0.) | (im_s == 0.)
    im_noise = np.sqrt(im_nr**2 + im_ns**2)
    im_subdiff = im_s - im_r

    im_subdiff[im_mask] = 0.
    im_noise[im_mask] = 0.


    fname_diff = outdir + 'output/nexus_{}{}_{}.subdiff.fits'.format(names[1], epochs[1], jwst_band)
    fname_mdiff = outdir + 'output/nexus_{}{}_{}.mask.fits'.format(names[1], epochs[1], jwst_band)
    fname_ndiff = outdir + 'output/nexus_{}{}_{}.noise.fits'.format(names[1], epochs[1], jwst_band)

    fits.writeto(fname_diff, im_subdiff.astype(np.float32), hdr_s, overwrite=True)
    fits.writeto(fname_mdiff, im_mask.astype(np.int16), hdr_s, overwrite=True)
    fits.writeto(fname_ndiff, im_noise.astype(np.float32), hdr_s, overwrite=True)