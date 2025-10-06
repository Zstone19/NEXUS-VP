
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.nddata import Cutout2D

import numpy as np
import os
import glob
import shutil

# import sys
# sys.path.append('../NEXUS-VP/src/')
# sys.path.append('../NEXUS-VP/src/psf')
# import psf_euclid as psfe
# import align as al


bands = ['f200w', 'f444w']
names = ['wide', 'deep']
epochs = ['01', '03']


# maindir = '/data3/web/nexus_collab/NIRCam/Deep_epoch/epoch1/60mas/'
output_dir = '/data6/stone28/nexus/NEXUS/cropped_{}{}_{}{}/'.format(names[0], epochs[0], names[1], epochs[1])
os.makedirs(output_dir, exist_ok=True)


# for b in bands:
#     fnames1 = [maindir1 + 'wide_ep1_01_' + b +'_i2d_' + x + '.fits.gz' for x in ['data', 'wht', 'err']]
#     fnames2 = [maindir2 + 'nexus_deep_epoch1_' + b + '_i2d_' + x + '.fits' for x in ['data', 'weight', 'error']] 
        
#     outdir = output_dir + 'align_' + b + '/'
#     os.makedirs(outdir, exist_ok=True)
    
#     fout1 = [outdir + 'nexus_wide_ep1_' + b + '_' + x + '.fits' for x in ['data', 'wht', 'err']]
#     fout2 = [outdir + 'nexus_deep_ep1_' + b + '_' + x + '.fits' for x in ['data', 'wht', 'err']]   



#     suffixes = ['data', 'wht', 'err']
#     for fin, fout in zip(fnames1, fout1):
#         if os.path.exists(fout):
#             continue
        
#         for s in suffixes:
#             if s == 'data':
#                 with fits.open(fin) as hdul:
#                     hdr = hdul[0].header
                    
#             if s == 'wht':
#                 with fits.open(fin) as hdul:
#                     wht = hdul[0].data

#                 mask = np.zeros_like(wht)
#                 filter_mask = (wht == 0)
#                 mask[filter_mask] = 1
#                 mask = mask.astype(int)
                
#                 fits.writeto(outdir + 'nexus_wide_ep1_' + b + '_mask.fits', mask, hdr, overwrite=True)
            
#             else:
#                 with fits.open(fin) as hdul:
#                     data = hdul[0].data
#                     hdr = hdul[0].header
                
#                 fits.writeto(fout, data, hdr, overwrite=True)                
       


#     for fin, fout in zip(fnames2, fout2):
#         if os.path.exists(fout):
#             continue
        
#         for s in suffixes:
#             if s == 'data':
#                 with fits.open(fin) as hdul:
#                     hdr = hdul[0].header
                    
#             if s == 'wht':
#                 with fits.open(fin) as hdul:
#                     wht = hdul[0].data

#                 mask = np.zeros_like(wht)
#                 filter_mask = (wht == 0)
#                 mask[filter_mask] = 1
#                 mask = mask.astype(int)
                
#                 fits.writeto(outdir + 'nexus_deep_ep1_' + b + '_mask.fits', mask, hdr, overwrite=True)
            
#             else:
#                 shutil.copy(fin, fout)


# #Run SWarp
# fname_default = '/home/stone28/projects/WebbDiffImg/pipeline/param_files/default.swarp'

# for b in bands:
#     swarp_dir = output_dir + 'align_' + b + '/'
#     ref_name = 'nexus_wide_ep1_' + b
#     sci_name = 'nexus_deep_ep1_' + b

#     al.align_frames(swarp_dir, fname_default, ref_name, sci_name, masktype='manual', ncpu=60, verbose=False)


for b in bands:
    print(b.upper())
    
    if b.upper() == 'F444W':
        maindir = '/data3/web/nexus_collab/nircam/Deep_epoch/'
        
        outdir_i = output_dir + b + '/'
        os.makedirs(outdir_i, exist_ok=True)
                
        fname1 = maindir + 'nexus_central_{}_ep{}_'.format(names[0], epochs[0]) + b + '_60mas_i2d_data.fits'    
        fname2 = maindir + 'nexus_central_{}_ep{}_'.format(names[1], epochs[1]) + b + '_006_60mas_i2d_data.fits'
        fname1e = maindir + 'nexus_central_{}_ep{}_'.format(names[0], epochs[0]) + b + '_60mas_i2d_error.fits'
        fname2e = maindir + 'nexus_central_{}_ep{}_'.format(names[1], epochs[1]) + b + '_006_60mas_i2d_error.fits'
        
    elif b.upper() == 'F200W':
        maindir = '/data3/web/nexus_collab/nircam/Deep_epoch/'
        
        outdir_i = output_dir + b + '/'
        os.makedirs(outdir_i, exist_ok=True)
                
        fname1 = maindir + 'nexus_central_{}_ep{}_'.format(names[0], epochs[0]) + b + '_i2d_data.fits'    
        fname2 = maindir + 'nexus_central_{}_ep{}_'.format(names[1], epochs[1]) + b + '_i2d_data.fits'
        fname1e = maindir + 'nexus_central_{}_ep{}_'.format(names[0], epochs[0]) + b + '_i2d_error.fits'
        fname2e = maindir + 'nexus_central_{}_ep{}_'.format(names[1], epochs[1]) + b + '_i2d_error.fits'


    
    # Read the data from the files
    with fits.open(fname1) as hdul:
        data1 = hdul[0].data
        header1 = hdul[0].header
    with fits.open(fname2) as hdul:
        data2 = hdul[0].data
        header2 = hdul[0].header
        
    wcs1 = WCS(header1)
    wcs2 = WCS(header2)
        
        
    mask1 = np.isnan(data1) | (data1 == 0.) | (~np.isfinite(data1))
    mask2 = np.isnan(data2) | (data2 == 0.) | (~np.isfinite(data2))
    mask_all = mask1 | mask2
    
    #Crop empty space
    ind = np.argwhere(~mask_all)
    x_min = ind[:,1].min()
    y_min = ind[:,0].min()
    x_max = ind[:,1].max()
    y_max = ind[:,0].max()
    
    y_min = max(0, y_min - 5)
    y_max = min(data1.shape[0], y_max + 5)
    x_min = max(0, x_min - 5)
    x_max = min(data1.shape[1], x_max + 5)
    
    xc = (x_min + x_max) / 2
    yc = (y_min + y_max) / 2
    shape_new = (y_max-y_min, x_max-x_min)
    
    rac, decc = wcs1.wcs_pix2world(xc, yc, 0)
    coord = SkyCoord(ra=rac, dec=decc, unit=(u.deg, u.deg))
    
    #############################################
    #DATA
    print('\t Data')
    
    cutout1 = Cutout2D(data1, coord, shape_new, wcs=wcs1)
    cutout2 = Cutout2D(data2, coord, shape_new, wcs=wcs2)
    hdr1_o = cutout1.wcs.to_header()
    hdr2_o = cutout2.wcs.to_header()
    
    maskA = (cutout1.data == 0) | (~np.isfinite(cutout1.data)) | np.isnan(cutout1.data)
    maskB = (cutout2.data == 0) | (~np.isfinite(cutout2.data)) | np.isnan(cutout2.data)
    mask_both = maskA | maskB
    
    cutout1.data[mask_both] = np.nan
    cutout2.data[mask_both] = np.nan
    
    fits.writeto(outdir_i + 'nexus_{}_ep{}_'.format(names[0], epochs[0]) + b + '_data.shift.fits', cutout1.data, header=hdr1_o, overwrite=True)
    fits.writeto(outdir_i + 'nexus_{}_ep{}_'.format(names[1], epochs[1]) + b + '_data.shift.fits', cutout2.data, header=hdr2_o, overwrite=True)
    
    #############################################
    #ERROR
    print('\t Error')
    
    with fits.open(fname1e) as hdul:
        err1 = hdul[0].data
        hdr1 = hdul[0].header
    with fits.open(fname2e) as hdul:
        err2 = hdul[0].data
        hdr2 = hdul[0].header
        
    cutout1 = Cutout2D(err1, coord, shape_new, wcs=wcs1)
    cutout2 = Cutout2D(err2, coord, shape_new, wcs=wcs2)
    hdr1_o = cutout1.wcs.to_header()
    hdr2_o = cutout2.wcs.to_header()
    
    maskA = (cutout1.data == 0) | (~np.isfinite(cutout1.data)) | np.isnan(cutout1.data)
    maskB = (cutout2.data == 0) | (~np.isfinite(cutout2.data)) | np.isnan(cutout2.data)
    mask_both = maskA | maskB

    cutout1.data[mask_both] = np.nan
    cutout2.data[mask_both] = np.nan
    
    fits.writeto(outdir_i + 'nexus_{}_ep{}_'.format(names[0], epochs[0]) + b + '_err.shift.fits', cutout1.data, header=hdr1_o, overwrite=True)
    fits.writeto(outdir_i + 'nexus_{}_ep{}_'.format(names[1], epochs[1]) + b + '_err.shift.fits', cutout2.data, header=hdr2_o, overwrite=True)
    
    #############################################
    #MASK      
    print('\t Mask')
      
    cutout1 = Cutout2D(mask1, coord, shape_new, wcs=wcs1)
    cutout2 = Cutout2D(mask2, coord, shape_new, wcs=wcs2)
    hdr1_o = cutout1.wcs.to_header()
    hdr2_o = cutout2.wcs.to_header()

    maskA = (cutout1.data) | (~np.isfinite(cutout1.data)) | np.isnan(cutout1.data)
    maskB = (cutout2.data) | (~np.isfinite(cutout2.data)) | np.isnan(cutout2.data)
    mask_both = maskA | maskB
    
    cutout1.data[mask_both] = True
    cutout2.data[mask_both] = True
    
    
    fits.writeto(outdir_i + 'nexus_{}_ep{}_'.format(names[0], epochs[0]) + b + '_mask.shift.fits', cutout1.data.astype(np.int16), header=hdr1_o, overwrite=True)
    fits.writeto(outdir_i + 'nexus_{}_ep{}_'.format(names[1], epochs[1]) + b + '_mask.shift.fits', cutout2.data.astype(np.int16), header=hdr2_o, overwrite=True)
