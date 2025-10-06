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

    outdir = '/data6/stone28/nexus/sfft_nexus_{}{}_{}{}_{}/'.format(names[0], epochs[0], names[1], epochs[1], jwst_band)
    os.makedirs(outdir, exist_ok=True)
    for subdir in ['input/', 'output/', 'noise/', 'psf/', 'mask/']:
        os.makedirs(outdir + subdir, exist_ok=True)
        
        
    ############################################################################################
    #Write data/noise/mask files for SFFT
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
            
            
        if (names[1]+epochs[1] == 'deep03') and (jwst_band == 'F444W'):
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
        
    im_out = im.copy()
    # im_out = im[498:-498, 498:-498].copy()
    fits.writeto(outdir + 'psf/nexus_{}{}_{}.psf.fits'.format(names[0], epochs[0], jwst_band), im_out, hdr, overwrite=True)


    fname = '/data6/stone28/nexus/NEXUS/webbpsf/final/nexus_{}{}_{}.psf.fits'.format(names[1], epochs[1], jwst_band.lower())
    with fits.open(fname) as hdul:
        hdr = hdul[0].header
        im = hdul[0].data

    im_out = im.copy()
    # im_out = im[498:-498, 498:-498].copy()
    fits.writeto(outdir + 'psf/nexus_{}{}_{}.psf.fits'.format(names[1], epochs[1], jwst_band), im_out, hdr, overwrite=True)