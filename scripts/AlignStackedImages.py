from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.nddata import Cutout2D

import numpy as np
import os
import glob
import shutil
import subprocess

# import sys
# sys.path.append('../pipeline/src/')
# sys.path.append('../pipeline/src/psf')
# import psf_euclid as psfe
# import align as al




def get_ref_wcsfile(fname_ref, fname_wcs):
    hdr = fits.open(fname_ref)[0].header
    
    if (hdr['CTYPE1'] == 'RA---TAN') and ('PV1_0' in hdr):
        _hdr = hdr.copy()
        _hdr['CTYPE1'] = 'RA---TPV'
        _hdr['CTYPE2'] = 'DEC--TPV'
    else: 
        _hdr = hdr

    w_ref = WCS(_hdr)

    wcshdr_ref = w_ref.to_header(relax=True)   # SIP distorsion requires relax=True

    wcshdr_ref['BITPIX'] = hdr['BITPIX']
    wcshdr_ref['NAXIS'] = hdr['NAXIS']

    wcshdr_ref['NAXIS1'] = hdr['NAXIS1']
    wcshdr_ref['NAXIS2'] = hdr['NAXIS2']

    wcshdr_ref.tofile(fname_wcs, overwrite=True)
    return


def make_swarp_config(fname_swarp_default, 
                     fname_out_swarp, fname_out, fname_weight_out, ncpu=1):

    with open(fname_swarp_default, 'r') as f:
        content = f.readlines()

    content[4]   = "IMAGEOUT_NAME          {}              # Output filename\n".format(fname_out)
    content[5]   = "WEIGHTOUT_NAME         {}              # Output weight-map filename\n".format(fname_weight_out)

    content[12]  = "WEIGHT_TYPE            MAP_WEIGHT      # BACKGROUND,MAP_RMS,MAP_VARIANCE\n"    
    content[14]  = "RESCALE_WEIGHTS        N               # Rescale input weights/variances (Y/N)?\n"
    content[15]  = "WEIGHT_SUFFIX          _weight.fits      # Suffix to use for weight-maps\n"

    content[22]  = "COMBINE                Y               # Combine resampled images (Y/N)?\n"
    content[23]  = "COMBINE_TYPE           MEDIAN          # MEDIAN,AVERAGE,MIN,MAX,WEIGHTED,CLIPPED\n"

    content[76]  = "SUBTRACT_BACK          N               # Subtraction sky background (Y/N)?\n"

    content[93]  = "VMEM_MAX               200000           # Maximum amount of virtual memory (MB)\n"
    content[94]  = "MEM_MAX                200000           # Maximum amount of physical memory (MB)\n"
    content[95]  = "COMBINE_BUFSIZE        200000           # RAM dedicated to co-addition(MB)\n"
    content[113] = "NTHREADS               {}              # Number of simultaneous threads for\n".format(ncpu)

    f = open(fname_out_swarp, 'w+')
    f.writelines(content)
    f.close()

    return




bands = ['f200w', 'f444w']
names = ['wide', 'deep']
epochs = ['01', '02']

output_dir = '/data6/stone28/nexus/NEXUS/stacked_cropped_{}{}_{}{}/'.format(names[0], epochs[0], names[1], epochs[1])
indir_stack = '/data4/jwst/nexus/reduced_data/SEXtractor/detection_maps/'
fname_in = 'nexus_full_wide_ep01+deep_ep01+ep02_detection.fits'

os.makedirs(output_dir, exist_ok=True)


#Run SWarp
fname_default = '/home/stone28/projects/WebbDiffImg/NEXUS-VP/param_files/default.swarp'

for b in bands:
    print(b.upper())
    
    swarp_dir = output_dir + b + '/'
    os.makedirs(swarp_dir, exist_ok=True)
    
    #Input REF file
    indir = '/data6/stone28/nexus/zogy_nexus_{}{}_{}{}_{}/input/'.format(names[0], epochs[0], names[1], epochs[1], b.upper())
    fname_ref = indir + 'nexus_{}{}_{}.fits'.format(names[1], epochs[1], b.upper())
    
    fname_mask1 = indir + 'nexus_{}{}_{}.maskin.fits'.format(names[0], epochs[0], b.upper())
    fname_mask2 = indir + 'nexus_{}{}_{}.maskin.fits'.format(names[1], epochs[1], b.upper())

    with fits.open(fname_ref) as hdul:
        im_r = hdul[0].data
        wcs_r = WCS(hdul[0].header)
        ps_r = hdul[0].header['CDELT1'] * 3600 #arcsec/px
        
    with fits.open(fname_mask1) as hdul:
        mask1 = hdul[0].data.astype(bool)
    with fits.open(fname_mask2) as hdul:
        mask2 = hdul[0].data.astype(bool)

    mask_tot = mask1 | mask2

        
    N0 = im_r.shape[0]
    N1 = im_r.shape[1]
    
    #Orientation can be different
    Nnew = int( np.max([N0, N1]) * (np.sqrt(2)+1)/2) + 10
    shape_new = (Nnew*ps_r*u.arcsec, Nnew*ps_r*u.arcsec)

    xc = N1/2
    yc = N0/2
    rac, decc = wcs_r.wcs_pix2world(xc, yc, 0) # Central RA/DEC of REF image
    coord = SkyCoord(ra=rac, dec=decc, unit=(u.deg, u.deg))
    
    
    if not os.path.exists(output_dir +fname_in):
        with fits.open(indir_stack + fname_in) as hdul:
            im = hdul[0].data
            wcs = WCS(hdul[0].header)

        cutout = Cutout2D(im, coord, shape_new, wcs=wcs)
        hdr = cutout.wcs.to_header()
        fits.writeto(output_dir + fname_in, cutout.data, hdr, overwrite=True)

        with fits.open(indir_stack + fname_in.replace('.fits', '_weight.fits')) as hdul:
            im = hdul[0].data
            wcs = WCS(hdul[0].header)

        cutout = Cutout2D(im, coord, shape_new, wcs=wcs)
        hdr = cutout.wcs.to_header()
        fits.writeto(output_dir + fname_in.replace('.fits', '_weight.fits'), cutout.data, hdr, overwrite=True)    
    

    # if b == 'f200w':        
    #     with fits.open(indir_stack + fname_in) as hdul:
    #         im = hdul[0].data
    #         wcs = WCS(hdul[0].header)

    #     cutout = Cutout2D(im, coord, (N0*u.pixel, N1*u.pixel), wcs=wcs)
    #     hdr = cutout.wcs.to_header()
    #     im_out = cutout.data
    #     im_out[mask_tot] = np.nan
    #     fits.writeto(output_dir + '{}/nexus_wide01_deep01_stacked_{}.fits'.format(b, b.upper()), im_out, hdr, overwrite=True)
        
    #     with fits.open(indir_stack + fname_in.replace('.fits', '_weight.fits')) as hdul:
    #         im = hdul[0].data
    #         wcs = WCS(hdul[0].header)
            
    #     cutout = Cutout2D(im, coord, (N0*u.pixel, N1*u.pixel), wcs=wcs)
    #     hdr = cutout.wcs.to_header()
    #     im_out = cutout.data
    #     im_out[mask_tot] = np.nan
    #     fits.writeto(output_dir + '{}/nexus_wide01_deep01_stacked_{}.weight.fits'.format(b, b.upper()), im_out, hdr, overwrite=True)


    # else:
    prefix = 'nexus_{}{}_{}{}_stacked_{}'.format(names[0], epochs[0], names[1], epochs[1], b.upper())
    
    #Get WCS file
    fname_wcs = swarp_dir + prefix + '.head'
    get_ref_wcsfile(fname_ref, fname_wcs)
    
    #Make SWarp config file
    fname_out = swarp_dir + prefix + '.fits'
    fname_weight_out = swarp_dir + prefix + '.weight.fits'
    fname_swarp_out = swarp_dir + prefix + '.swarp'
    make_swarp_config(fname_default, fname_swarp_out, fname_out, fname_weight_out, ncpu=30)
    

    #Run SWarp
    f = output_dir + fname_in
    command = ['swarp', f, '-c', fname_swarp_out]
    
    stdout = subprocess.PIPE
    out = subprocess.run(command, check=True, stdout=stdout, stderr=stdout).stdout
    
