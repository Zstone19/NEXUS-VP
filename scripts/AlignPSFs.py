import numpy as np
from astropy.io import fits
import os

import sys
sys.path.append('..src/')
from align import align_psf



size = (1201,1201)
names = ['wide', 'deep', 'deep']
epochs = ['01', '01', '02']
bands = ['F200W', 'F444W']


for name, epoch in zip(names, epochs):
    for band in bands:

        fname_dat = '/data3/web/nexus_collab/nircam/Deep_epoch/'
        if band == 'F200W':
            suffix = ''
        elif band == 'F444W':
            suffix = '_60mas'
            
            
        fname_dat += 'nexus_central_{}_ep{}_{}{}_i2d_data.fits'.format(name, epoch, band.lower(), suffix)
        fname_psf = '/data6/stone28/nexus/NEXUS/webbpsf/nexus_{}{}_{}.webbpsf.fits'.format(name, epoch, band.lower())

        outdir = '/data6/stone28/nexus/NEXUS/webbpsf/align_{}/'.format(band.lower())
        os.makedirs(outdir, exist_ok=True)
        
        prefix = 'nexus_{}{}_{}'.format(name, epoch, band.lower())
        fname_default = '/home/stone28/projects/WebbDiffImg/NEXUS-VP/param_files/default.swarp'

        #Run swarp
        align_psf(fname_dat, fname_psf, outdir, prefix, fname_default, ref_shape=size)

        #Normalize
        outdir2 = '/data6/stone28/nexus/NEXUS/webbpsf/final/'
        os.makedirs(outdir2, exist_ok=True)
        
        fname1 = outdir + prefix + '_psf.align.fits'
        fname2 = outdir2 + prefix + '.psf.fits'

        with fits.open(fname_psf) as hdul:
            fwhm = hdul[0].header['FWHM']
        with fits.open(fname1) as hdul:
            im = hdul[0].data
            hdr = hdul[0].header
            
        im /= np.nansum(im)
        hdr['FWHM'] = fwhm
        fits.writeto(fname2, im, hdr, overwrite=True)