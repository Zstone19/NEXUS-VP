import os
import shutil

import numpy as np
from astropy.table import Table

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import astropy.units as u



def remove_empty_space(x1_vals, x2_vals, y1_vals, y2_vals, im_mref, im_msci):

    N0, N1 = im_mref.shape

    x1_new = []
    x2_new = []
    y1_new = []
    y2_new = []
    n_nz_new = []
    
    for i in range(len(x1_vals)):
        x1 = x1_vals[i]
        x2 = x2_vals[i]
        y1 = y1_vals[i]
        y2 = y2_vals[i]
        
        im_ij1_m = im_msci[x1:x2, y1:y2].copy()
        im_ij2_m = im_mref[x1:x2, y1:y2].copy()
        
        x1a = 0
        x2a = im_ij1_m.shape[0]
        y1a = 0
        y2a = im_ij1_m.shape[1]
    
        for im in [im_ij1_m, im_ij2_m]:
            nonzero_ind = np.argwhere(~im)
            
            if len(nonzero_ind) == 0:
                y1a_i = 0
                y2a_i = 0
                x1a_i = 0
                x2a_i = 0
            else:
                y1a_i = nonzero_ind[:,1].min()-1
                y2a_i = nonzero_ind[:,1].max()+1
                x1a_i = nonzero_ind[:,0].min()-1
                x2a_i = nonzero_ind[:,0].max()+1                    
            

            if y1a_i > y1a:
                y1a = y1a_i
            if y2a_i < y2a:
                y2a = y2a_i
            if x1a_i > x1a:
                x1a = x1a_i
            if x2a_i < x2a:
                x2a = x2a_i

        if y1a < 0:
            y1a = 0
        if y2a > im_ij1_m.shape[1]:
            y2a = im_ij1_m.shape[1]
        if x1a < 0:
            x1a = 0
        if x2a > im_ij1_m.shape[0]:
            x2a = im_ij1_m.shape[0]
        

        y1b = y1 + y1a
        y2b = y1 + y2a
        x1b = x1 + x1a
        x2b = x1 + x2a

        if x1b < 0:
            x1b = 0
        if y1b < 0:
            y1b = 0
        if x2b > N0:
            x2b = N0
        if y2b > N1:
            y2b = N1
            
    
        im_ij1_m = im_msci[x1b:x2b, y1b:y2b].copy()
        im_ij2_m = im_mref[x1b:x2b, y1b:y2b].copy()
        n_nz = np.sum( (~im_ij1_m) & (~im_ij2_m) )
        if (n_nz == 0) or np.all(im_ij1_m == 1) or np.all(im_ij2_m == 1):
            continue
        
        n_nz_new.append(n_nz)
        
        if (x2b-x1b == 0) or (y2b-y1b == 0) or (n_nz == 0):
            continue
        
        x1_new.append(x1b)
        x2_new.append(x2b)
        y1_new.append(y1b)
        y2_new.append(y2b)
        

    x1_new = np.array(x1_new, dtype=int)
    x2_new = np.array(x2_new, dtype=int)
    y1_new = np.array(y1_new, dtype=int)
    y2_new = np.array(y2_new, dtype=int)
    n_nz_new = np.array(n_nz_new, dtype=int)
    
    return x1_new, x2_new, y1_new, y2_new, n_nz_new


###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################

def make_cutout_subdir_together(cutout_name, ra, dec, maindir, ref_name, sci_name, cutout_shape, skysub=False, crossconv=True):    
    outdir = maindir + 'output_{}/'.format(cutout_name)
    
    if os.path.exists(outdir):
        return
    
    os.makedirs(outdir, exist_ok=True)
    for subdir in ['input/', 'noise/', 'psf/', 'mask/', 'output/']:
        os.makedirs(outdir + subdir, exist_ok=True)
    
    
    ##################################################
    # Get WCS

    obj_coord = SkyCoord(ra, dec, unit=(u.deg, u.deg))
    
    if skysub:
        fname_ref = maindir + 'input/{}.skysub.fits'.format(ref_name)
        fname_sci = maindir + 'input/{}.skysub.fits'.format(sci_name)    
    else:
        fname_ref = maindir + 'input/{}.fits'.format(ref_name)
        fname_sci = maindir + 'input/{}.fits'.format(sci_name)
    
    wcs_ref = WCS(fits.open(fname_ref)[0].header)
    wcs_sci = WCS(fits.open(fname_sci)[0].header)
    
    ################################################
    #Input image
    
    hdu_r = fits.open(fname_ref)[0]
    hdu_s = fits.open(fname_sci)[0]
    
    im_r = hdu_r.data
    im_s = hdu_s.data
    
    cutout_r = Cutout2D(im_r, obj_coord, cutout_shape, wcs=wcs_ref)
    cutout_s = Cutout2D(im_s, obj_coord, cutout_shape, wcs=wcs_sci)
    
    hdu_r.data = cutout_r.data
    hdu_s.data = cutout_s.data
    
    hdu_r.data[np.isnan(hdu_r.data)] = 0.
    hdu_s.data[np.isnan(hdu_s.data)] = 0.
    
    hdu_r.header.update(cutout_r.wcs.to_header())
    hdu_s.header.update(cutout_s.wcs.to_header())
    
    fname_out_r = outdir + 'input/{}.fits'.format(ref_name)
    fname_out_s = outdir + 'input/{}.fits'.format(sci_name)
    
    hdu_r.writeto(fname_out_r, overwrite=True)
    hdu_s.writeto(fname_out_s, overwrite=True)
    
    ##################################################
    #Noise image
    
    fname_ref = maindir + 'noise/{}.noise.fits'.format(ref_name)
    fname_sci = maindir + 'noise/{}.noise.fits'.format(sci_name)
    
    hdu_r = fits.open(fname_ref)[0]
    hdu_s = fits.open(fname_sci)[0]
    
    im_r = hdu_r.data
    im_s = hdu_s.data
    
    cutout_r = Cutout2D(im_r, obj_coord, cutout_shape, wcs=wcs_ref)
    cutout_s = Cutout2D(im_s, obj_coord, cutout_shape, wcs=wcs_sci)
    
    hdu_r.data = cutout_r.data
    hdu_s.data = cutout_s.data
    
    hdu_r.data[np.isnan(hdu_r.data)] = 0.
    hdu_s.data[np.isnan(hdu_s.data)] = 0.
    
    hdu_r.header.update(cutout_r.wcs.to_header())
    hdu_s.header.update(cutout_s.wcs.to_header())
    
    fname_out_r = outdir + 'noise/{}.noise.fits'.format(ref_name)
    fname_out_s = outdir + 'noise/{}.noise.fits'.format(sci_name)
    
    hdu_r.writeto(fname_out_r, overwrite=True)
    hdu_s.writeto(fname_out_s, overwrite=True)
    
    ##################################################
    #PSF image
    
    fname_ref = maindir + 'psf/{}.psf.fits'.format(ref_name)
    fname_sci = maindir + 'psf/{}.psf.fits'.format(sci_name)
    
    shutil.copy(fname_ref, outdir + 'psf/{}.psf.fits'.format(ref_name))
    shutil.copy(fname_sci, outdir + 'psf/{}.psf.fits'.format(sci_name))
    
    ##################################################
    #Mask image
    
    fname_ref = maindir + 'input/{}.maskin.fits'.format(ref_name)
    fname_sci = maindir + 'input/{}.maskin.fits'.format(sci_name)
    
    hdu_r = fits.open(fname_ref)[0]
    hdu_s = fits.open(fname_sci)[0]
    
    im_r = hdu_r.data
    im_s = hdu_s.data
    
    cutout_r = Cutout2D(im_r, obj_coord, cutout_shape, wcs=wcs_ref)
    cutout_s = Cutout2D(im_s, obj_coord, cutout_shape, wcs=wcs_sci)
    
    hdu_r.data = cutout_r.data
    hdu_s.data = cutout_s.data
    
    hdu_r.data[np.isnan(hdu_r.data)] = 0.
    hdu_s.data[np.isnan(hdu_s.data)] = 0.
    
    hdu_r.header.update(cutout_r.wcs.to_header())
    hdu_s.header.update(cutout_s.wcs.to_header())
    
    fname_out_r = outdir + 'input/{}.maskin.fits'.format(ref_name)
    fname_out_s = outdir + 'input/{}.maskin.fits'.format(sci_name)
    
    hdu_r.writeto(fname_out_r, overwrite=True)
    hdu_s.writeto(fname_out_s, overwrite=True)
    
    ##################################################
    #Cross-convolved image
    
    if crossconv:
        fname_ref = maindir + 'output/{}.crossconvd.fits'.format(ref_name)
        fname_sci = maindir + 'output/{}.crossconvd.fits'.format(sci_name)
        
        hdu_r = fits.open(fname_ref)[0]
        hdu_s = fits.open(fname_sci)[0]
        
        im_r = hdu_r.data
        im_s = hdu_s.data
        
        cutout_r = Cutout2D(im_r, obj_coord, cutout_shape, wcs=wcs_ref)
        cutout_s = Cutout2D(im_s, obj_coord, cutout_shape, wcs=wcs_sci)
        
        hdu_r.data = cutout_r.data
        hdu_s.data = cutout_s.data
        
        hdu_r.data[np.isnan(hdu_r.data)] = 0.
        hdu_s.data[np.isnan(hdu_s.data)] = 0.
        
        hdu_r.header.update(cutout_r.wcs.to_header())
        hdu_s.header.update(cutout_s.wcs.to_header())
        
        fname_out_r = outdir + 'output/{}.crossconvd.fits'.format(ref_name)
        fname_out_s = outdir + 'output/{}.crossconvd.fits'.format(sci_name)
        
        hdu_r.writeto(fname_out_r, overwrite=True)
        hdu_s.writeto(fname_out_s, overwrite=True)
    
    ##################################################
    #Masked cross-convolved image
    
    if crossconv:
        fname_ref = maindir + 'output/{}.crossconvd.masked.fits'.format(ref_name)
        fname_sci = maindir + 'output/{}.crossconvd.masked.fits'.format(sci_name)
    else:
        fname_ref = maindir + 'output/{}.fits'.format(ref_name)
        fname_sci = maindir + 'output/{}.fits'.format(sci_name)
    
    hdu_r = fits.open(fname_ref)[0]
    hdu_s = fits.open(fname_sci)[0]
    
    im_r = hdu_r.data
    im_s = hdu_s.data
    
    cutout_r = Cutout2D(im_r, obj_coord, cutout_shape, wcs=wcs_ref)
    cutout_s = Cutout2D(im_s, obj_coord, cutout_shape, wcs=wcs_sci)
    
    hdu_r.data = cutout_r.data
    hdu_s.data = cutout_s.data
    
    hdu_r.data[np.isnan(hdu_r.data)] = 0.
    hdu_s.data[np.isnan(hdu_s.data)] = 0.
    
    hdu_r.header.update(cutout_r.wcs.to_header())
    hdu_s.header.update(cutout_s.wcs.to_header())
    
    if crossconv:
        fname_out_r = outdir + 'output/{}.crossconvd.masked.fits'.format(ref_name)
        fname_out_s = outdir + 'output/{}.crossconvd.masked.fits'.format(sci_name)
    else:
        fname_out_r = outdir + 'output/{}.fits'.format(ref_name)
        fname_out_s = outdir + 'output/{}.fits'.format(sci_name)
    
    hdu_r.writeto(fname_out_r, overwrite=True)
    hdu_s.writeto(fname_out_s, overwrite=True)
    
    ##################################################
    #SFFT mask
    
    fname_ref = maindir + 'mask/{}.mask4sfft.fits'.format(ref_name)
    fname_sci = maindir + 'mask/{}.mask4sfft.fits'.format(sci_name)
    
    hdu_r = fits.open(fname_ref)[0]
    hdu_s = fits.open(fname_sci)[0]
    
    im_r = hdu_r.data
    im_s = hdu_s.data
    
    cutout_r = Cutout2D(im_r, obj_coord, cutout_shape, wcs=wcs_ref)
    cutout_s = Cutout2D(im_s, obj_coord, cutout_shape, wcs=wcs_sci)
    
    hdu_r.data = cutout_r.data
    hdu_s.data = cutout_s.data
    
    hdu_r.data[np.isnan(hdu_r.data)] = 0.
    hdu_s.data[np.isnan(hdu_s.data)] = 0.
    
    hdu_r.header.update(cutout_r.wcs.to_header())
    hdu_s.header.update(cutout_s.wcs.to_header())
    
    fname_out_r = outdir + 'mask/{}.mask4sfft.fits'.format(ref_name)
    fname_out_s = outdir + 'mask/{}.mask4sfft.fits'.format(sci_name)
    
    hdu_r.writeto(fname_out_r, overwrite=True)
    hdu_s.writeto(fname_out_s, overwrite=True)
    
    ####################################################
    #Segmaps
    
    fname_ref = maindir + 'mask/{}_sexseg.fits'.format(ref_name)
    fname_sci = maindir + 'mask/{}_sexseg.fits'.format(sci_name)
    
    if os.path.exists(fname_ref) and os.path.exists(fname_sci):
        hdu_r = fits.open(fname_ref)[0]
        hdu_s = fits.open(fname_sci)[0]
        
        im_r = hdu_r.data
        im_s = hdu_s.data
        
        cutout_r = Cutout2D(im_r, obj_coord, cutout_shape, wcs=wcs_ref)
        cutout_s = Cutout2D(im_s, obj_coord, cutout_shape, wcs=wcs_sci)
        
        hdu_r.data = cutout_r.data
        hdu_s.data = cutout_s.data
        
        hdu_r.data[np.isnan(hdu_r.data)] = 0.
        hdu_s.data[np.isnan(hdu_s.data)] = 0.
        
        hdu_r.header.update(cutout_r.wcs.to_header())
        hdu_s.header.update(cutout_s.wcs.to_header())
        
        fname_out_r = outdir + 'mask/{}_sexseg.fits'.format(ref_name)
        fname_out_s = outdir + 'mask/{}_sexseg.fits'.format(sci_name)
        
        hdu_r.writeto(fname_out_r, overwrite=True)
        hdu_s.writeto(fname_out_s, overwrite=True)
    
    return


def make_cutout_subdir_separate(cutout_name, ra, dec, maindir, ref_name, sci_name, cutout_shape):    
    outdir = maindir + 'output_{}/'.format(cutout_name)
    
    if os.path.exists(outdir):
        return
    
    os.makedirs(outdir, exist_ok=True)
    for subdir in ['input/', 'noise/', 'psf/', 'mask/', 'output/']:
        os.makedirs(outdir + subdir, exist_ok=True)
    
    
    ##################################################
    # Get WCS

    obj_coord = SkyCoord(ra, dec, unit=(u.deg, u.deg))
    
    fname_ref = maindir + 'input/{}.fits'.format(ref_name)
    fname_sci = maindir + 'input/{}.fits'.format(sci_name)
    
    wcs_ref = WCS(fits.open(fname_ref)[0].header)
    wcs_sci = WCS(fits.open(fname_sci)[0].header)
    
    ################################################
    #Input image
    
    hdu_r = fits.open(fname_ref)[0]
    hdu_s = fits.open(fname_sci)[0]
    
    im_r = hdu_r.data
    im_s = hdu_s.data
    
    cutout_r = Cutout2D(im_r, obj_coord, cutout_shape, wcs=wcs_ref)
    cutout_s = Cutout2D(im_s, obj_coord, cutout_shape, wcs=wcs_sci)
    
    hdu_r.data = cutout_r.data
    hdu_s.data = cutout_s.data
    
    hdu_r.data[np.isnan(hdu_r.data)] = 0.
    hdu_s.data[np.isnan(hdu_s.data)] = 0.
    
    hdu_r.header.update(cutout_r.wcs.to_header())
    hdu_s.header.update(cutout_s.wcs.to_header())
    
    fname_out_r = outdir + 'input/{}.fits'.format(ref_name)
    fname_out_s = outdir + 'input/{}.fits'.format(sci_name)
    
    hdu_r.writeto(fname_out_r, overwrite=True)
    hdu_s.writeto(fname_out_s, overwrite=True)
    
    ##################################################
    #Noise image
    
    fname_ref = maindir + 'noise/{}.noise.fits'.format(ref_name)
    fname_sci = maindir + 'noise/{}.noise.fits'.format(sci_name)
    
    hdu_r = fits.open(fname_ref)[0]
    hdu_s = fits.open(fname_sci)[0]
    
    im_r = hdu_r.data
    im_s = hdu_s.data
    
    cutout_r = Cutout2D(im_r, obj_coord, cutout_shape, wcs=wcs_ref)
    cutout_s = Cutout2D(im_s, obj_coord, cutout_shape, wcs=wcs_sci)
    
    hdu_r.data = cutout_r.data
    hdu_s.data = cutout_s.data
    
    hdu_r.data[np.isnan(hdu_r.data)] = 0.
    hdu_s.data[np.isnan(hdu_s.data)] = 0.
    
    hdu_r.header.update(cutout_r.wcs.to_header())
    hdu_s.header.update(cutout_s.wcs.to_header())
    
    fname_out_r = outdir + 'noise/{}.noise.fits'.format(ref_name)
    fname_out_s = outdir + 'noise/{}.noise.fits'.format(sci_name)
    
    hdu_r.writeto(fname_out_r, overwrite=True)
    hdu_s.writeto(fname_out_s, overwrite=True)
    
    ##################################################
    #PSF image
    
    fname_ref = maindir + 'psf/{}.psf.fits'.format(ref_name)
    fname_sci = maindir + 'psf/{}.psf.fits'.format(sci_name)
    
    shutil.copy(fname_ref, outdir + 'psf/{}.psf.fits'.format(ref_name))
    shutil.copy(fname_sci, outdir + 'psf/{}.psf.fits'.format(sci_name))
    
    ##################################################
    #Mask image
    
    fname_ref = maindir + 'input/{}.maskin.fits'.format(ref_name)
    fname_sci = maindir + 'input/{}.maskin.fits'.format(sci_name)
    
    hdu_r = fits.open(fname_ref)[0]
    hdu_s = fits.open(fname_sci)[0]
    
    im_r = hdu_r.data
    im_s = hdu_s.data
    
    cutout_r = Cutout2D(im_r, obj_coord, cutout_shape, wcs=wcs_ref)
    cutout_s = Cutout2D(im_s, obj_coord, cutout_shape, wcs=wcs_sci)
    
    hdu_r.data = cutout_r.data
    hdu_s.data = cutout_s.data
    
    hdu_r.data[np.isnan(hdu_r.data)] = 0.
    hdu_s.data[np.isnan(hdu_s.data)] = 0.
    
    hdu_r.header.update(cutout_r.wcs.to_header())
    hdu_s.header.update(cutout_s.wcs.to_header())
    
    fname_out_r = outdir + 'input/{}.maskin.fits'.format(ref_name)
    fname_out_s = outdir + 'input/{}.maskin.fits'.format(sci_name)
    
    hdu_r.writeto(fname_out_r, overwrite=True)
    hdu_s.writeto(fname_out_s, overwrite=True)
        
    return


###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################


def make_split_subdir_together(npx_side, maindir, ref_name, sci_name, npx_boundary=30, 
                               skysub=False, conv_ref=False, conv_sci=False, subset=None, dither=None):    

    #Need to take into account that gkerhw pixels from the edge of the image are masked
    #So overlap the cutouts by gkerhw pixels        

    npx_boundary = 30

    ra_vals = []
    dec_vals = []
    row_inds = []
    col_inds = []
    shapes_x = []
    shapes_y = []
    numbers = []
    


    fname_ref = maindir + 'input/{}.fits'.format(ref_name)
    fname_sci = maindir + 'input/{}.fits'.format(sci_name)    
    fname_mref = maindir + 'input/{}.maskin.fits'.format(ref_name)
    fname_msci = maindir + 'input/{}.maskin.fits'.format(sci_name)
        
    with fits.open(fname_ref) as hdul:
        wcs_ref = WCS(hdul[0].header)
        im_ref = hdul[0].data

    with fits.open(fname_sci) as hdul:
        wcs_sci = WCS(hdul[0].header)    
        im_sci = hdul[0].data
        
    with fits.open(fname_mref) as hdul:
        im_mref = hdul[0].data
    with fits.open(fname_msci) as hdul:
        im_msci = hdul[0].data

    assert im_ref.shape == im_sci.shape, "Reference and science images must have the same shape."
    N0, N1 = im_sci.shape
    wcs = wcs_ref
    
    if dither == 'dec':
        xsplit1 = np.arange(0, N0+npx_side, npx_side)
        xsplit1[-1] = N0
        nsplitx = len(xsplit1)-1

        x1 = int(npx_side/2)
        xsplit2 = np.arange(x1, N0+npx_side, npx_side)
        xsplit2[-1] = N0
        
        xsplit = np.concatenate((xsplit1, xsplit2))   
                     
        ysplit = np.arange(0, N1+npx_side, npx_side)
        ysplit[-1] = N1
        

    elif dither == 'ra':
        ysplit1 = np.arange(0, N1+npx_side, npx_side)
        ysplit1[-1] = N1
        nsplity = len(ysplit1)-1

        y1 = int(npx_side/2)
        ysplit2 = np.arange(y1, N1+npx_side, npx_side)
        ysplit2[-1] = N1
        
        xsplit = np.arange(0, N0+npx_side, npx_side)
        xsplit[-1] = N0
        
        ysplit = np.concatenate((ysplit1, ysplit2))
        
        
    elif dither == 'both':
        xsplit1 = np.arange(0, N0+npx_side, npx_side)
        xsplit1[-1] = N0
        nsplitx = len(xsplit1)-1

        x1 = int(-npx_side/2)
        xsplit2 = np.arange(x1, N0+npx_side, npx_side)
        xsplit2[-1] = N0
        
        xsplit = np.concatenate((xsplit1, xsplit2))   
                     
        ysplit1 = np.arange(0, N1+npx_side, npx_side)
        ysplit1[-1] = N1
        nsplity = len(ysplit1)-1

        y1 = int(-npx_side/2)
        ysplit2 = np.arange(y1, N1+npx_side, npx_side)
        ysplit2[-1] = N1
        
        ysplit = np.concatenate((ysplit1, ysplit2))
        
    else:
        xsplit = np.arange(0, N0+npx_side, npx_side)
        xsplit[-1] = N0
        
        ysplit = np.arange(0, N1+npx_side, npx_side)
        ysplit[-1] = N1
    
    

    x1_vals = []
    x2_vals = []
    y1_vals = []
    y2_vals = []
    dx_vals = []
    dy_vals = []
    n_nz_vals = []

    for i in range(len(xsplit)-1):
        for j in range(len(ysplit)-1):
            n = i*(len(ysplit)-1) + j
        
            x1 = xsplit[i]
            x2 = xsplit[i+1]

            y1 = ysplit[j]
            y2 = ysplit[j+1]
            
            if (x2 < x1) or (y2 < y1):
                continue
            
                    
            im_ij1 = im_sci[x1:x2, y1:y2].copy()
            im_ij2 = im_ref[x1:x2, y1:y2].copy()
            im_ij1_m = im_msci[x1:x2, y1:y2].copy()
            im_ij2_m = im_mref[x1:x2, y1:y2].copy()
            
            x1a = 0
            x2a = im_ij1.shape[0]
            y1a = 0
            y2a = im_ij1.shape[1]
            
            n_nz = np.sum( (~im_ij1_m) & (~im_ij2_m) )
            if (n_nz == 0) or (np.nansum(im_ij1) == 0.) or (np.nansum(im_ij2) == 0.):
                continue
            
            n_nz_vals.append(n_nz)

            y1 = y1 + y1a
            y2 = y1 + y2a
            x1 = x1 + x1a
            x2 = x1 + x2a

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > N0:
                x2 = N0
            if y2 > N1:
                y2 = N1
                
            if (x2-x1 == 0) or (y2-y1 == 0) or (n_nz == 0):
                continue

            x1_vals.append(x1)
            x2_vals.append(x2)
            y1_vals.append(y1)
            y2_vals.append(y2)

  
    x1_vals = np.array(x1_vals, dtype=int)
    x2_vals = np.array(x2_vals, dtype=int)
    y1_vals = np.array(y1_vals, dtype=int)
    y2_vals = np.array(y2_vals, dtype=int)
    n_nz_vals = np.array(n_nz_vals, dtype=int)
    
    x1_vals, x2_vals, y1_vals, y2_vals, nnz_vals = remove_empty_space(x1_vals, x2_vals, y1_vals, y2_vals, im_mref, im_msci)
    for i in range(len(x1_vals)):
        if (x2_vals[i] - x1_vals[i]) < npx_side:
            if x1_vals[i] == 0:
                x2_vals[i] = x1_vals[i] + npx_side
            elif x2_vals[i] == N0:
                x1_vals[i] = x2_vals[i] - npx_side
            elif (x1_vals[i] > N0/2):
                x1_vals[i] = x2_vals[i] - npx_side
            elif (x2_vals[i] < N0/2):
                x2_vals[i] = x1_vals[i] + npx_side
                
        if (y2_vals[i] - y1_vals[i]) < npx_side:
            if y1_vals[i] == 0:
                y2_vals[i] = y1_vals[i] + npx_side
            elif y2_vals[i] == N1:
                y1_vals[i] = y2_vals[i] - npx_side
            elif (y1_vals[i] > N1/2):
                y1_vals[i] = y2_vals[i] - npx_side
            elif (y2_vals[i] < N1/2):
                y2_vals[i] = y1_vals[i] + npx_side


    nnz_fracs = []
    for i in range(len(x1_vals)):
        n = i
        
        if (subset is not None) and (not (n in subset)):
            continue
    
        x1 = x1_vals[i] - npx_boundary
        x2 = x2_vals[i] + npx_boundary

        y1 = y1_vals[i] - npx_boundary
        y2 = y2_vals[i] + npx_boundary

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > N0:
            x2 = N0
        if y2 > N1:
            y2 = N1
        
        if np.nansum(im_sci[x1:x2, y1:y2]) == 0.:
            continue
        if np.nansum(im_ref[x1:x2, y1:y2]) == 0.:
            continue
        
        assert im_sci[x1:x2, y1:y2].shape == im_ref[x1:x2, y1:y2].shape, "Reference and science images must have the same shape."

            
        if (x2-x1 == 0) or (y2-y1 == 0):
            continue
        
        xc = (x1 + x2) / 2.
        yc = (y1 + y2) / 2.
        
        cutout_shape = np.array([x2-x1, y2-y1]) * u.pixel
        
        ra, dec = wcs_ref.array_index_to_world_values(xc, yc)
        ra = float(ra)
        dec = float(dec)
        new_coords = SkyCoord(ra, dec, unit=(u.deg, u.deg))
        
        ra_vals.append(ra)
        dec_vals.append(dec)
        row_inds.append(xc)
        col_inds.append(yc)
        shapes_x.append(x2-x1)
        shapes_y.append(y2-y1)
        numbers.append(n)
        
        outdir = maindir + 'output_{}/'.format(n)            
        if os.path.exists(outdir):
            continue
        
        os.makedirs(outdir, exist_ok=True)
        
        for subdir in ['input/', 'noise/', 'psf/', 'mask/', 'output/']:
            os.makedirs(outdir + subdir, exist_ok=True)    
    
        ################################################
        #Input image
        
        fname_ref = maindir + 'input/{}.fits'.format(ref_name)
        fname_sci = maindir + 'input/{}.fits'.format(sci_name)
        fname_out_r = outdir + 'input/{}.fits'.format(ref_name)
        fname_out_s = outdir + 'input/{}.fits'.format(sci_name)
        
        with fits.open(fname_ref) as hdul:
            hdu_r = hdul[0]
            im_r = hdu_r.data
            cutout_r = Cutout2D(im_r, new_coords, cutout_shape, wcs=wcs)
            
            hdu_r.data = cutout_r.data
            hdu_r.data[np.isnan(hdu_r.data)] = 0.                
            hdu_r.header.update(cutout_r.wcs.to_header())
            hdu_r.writeto(fname_out_r, overwrite=True)


        with fits.open(fname_sci) as hdul:
            hdu_s = hdul[0]
            im_s = hdu_s.data            
            cutout_s = Cutout2D(im_s, new_coords, cutout_shape, wcs=wcs) 
                        
            hdu_s.data = cutout_s.data
            hdu_s.data[np.isnan(hdu_s.data)] = 0.    
            hdu_s.header.update(cutout_s.wcs.to_header())
            hdu_s.writeto(fname_out_s, overwrite=True)
            
            
        ################################################
        #Input image with skysub
        
        if skysub:
            fname_ref = maindir + 'input/{}.skysub.fits'.format(ref_name)
            fname_sci = maindir + 'input/{}.skysub.fits'.format(sci_name)
            fname_out_r = outdir + 'input/{}.skysub.fits'.format(ref_name)
            fname_out_s = outdir + 'input/{}.skysub.fits'.format(sci_name)
            
            with fits.open(fname_ref) as hdul:
                hdu_r = hdul[0]
                im_r = hdu_r.data
                cutout_r = Cutout2D(im_r, new_coords, cutout_shape, wcs=wcs)
                
                hdu_r.data = cutout_r.data
                hdu_r.data[np.isnan(hdu_r.data)] = 0.                
                hdu_r.header.update(cutout_r.wcs.to_header())
                hdu_r.writeto(fname_out_r, overwrite=True)


            with fits.open(fname_sci) as hdul:
                hdu_s = hdul[0]
                im_s = hdu_s.data            
                cutout_s = Cutout2D(im_s, new_coords, cutout_shape, wcs=wcs) 
                            
                hdu_s.data = cutout_s.data
                hdu_s.data[np.isnan(hdu_s.data)] = 0.    
                hdu_s.header.update(cutout_s.wcs.to_header())
                hdu_s.writeto(fname_out_s, overwrite=True)
                
                
        ################################################
        #Cross-convolved image
        
        if conv_ref:
            fname_ref = maindir + 'output/{}.crossconvd.fits'.format(ref_name)
            fname_out_r = outdir + 'output/{}.crossconvd.fits'.format(ref_name)
            
            with fits.open(fname_ref) as hdul:
                hdu_r = hdul[0]
                im_r = hdu_r.data
                cutout_r = Cutout2D(im_r, new_coords, cutout_shape, wcs=wcs)
                
                hdu_r.data = cutout_r.data
                hdu_r.data[np.isnan(hdu_r.data)] = 0.                
                hdu_r.header.update(cutout_r.wcs.to_header())
                hdu_r.writeto(fname_out_r, overwrite=True)


        if conv_sci:
            fname_sci = maindir + 'output/{}.crossconvd.fits'.format(sci_name)
            fname_out_s = outdir + 'output/{}.crossconvd.fits'.format(sci_name)

            with fits.open(fname_sci) as hdul:
                hdu_s = hdul[0]
                im_s = hdu_s.data            
                cutout_s = Cutout2D(im_s, new_coords, cutout_shape, wcs=wcs) 
                            
                hdu_s.data = cutout_s.data
                hdu_s.data[np.isnan(hdu_s.data)] = 0.    
                hdu_s.header.update(cutout_s.wcs.to_header())
                hdu_s.writeto(fname_out_s, overwrite=True)
                
        ################################################
        #Cross-convolved image (masked)
        
        if conv_ref:
            fname_ref = maindir + 'output/{}.crossconvd.masked.fits'.format(ref_name)
            fname_out_r = outdir + 'output/{}.crossconvd.masked.fits'.format(ref_name)
        else:
            fname_ref = maindir + 'output/{}.masked.fits'.format(ref_name)
            fname_out_r = outdir + 'output/{}.masked.fits'.format(ref_name)
            
        with fits.open(fname_ref) as hdul:
            hdu_r = hdul[0]
            im_r = hdu_r.data
            cutout_r = Cutout2D(im_r, new_coords, cutout_shape, wcs=wcs)
            
            hdu_r.data = cutout_r.data
            hdu_r.data[np.isnan(hdu_r.data)] = 0.                
            hdu_r.header.update(cutout_r.wcs.to_header())
            hdu_r.writeto(fname_out_r, overwrite=True)


        if conv_sci:
            fname_sci = maindir + 'output/{}.crossconvd.masked.fits'.format(sci_name)
            fname_out_s = outdir + 'output/{}.crossconvd.masked.fits'.format(sci_name)
        else:
            fname_sci = maindir + 'output/{}.masked.fits'.format(sci_name)
            fname_out_s = outdir + 'output/{}.masked.fits'.format(sci_name)

        with fits.open(fname_sci) as hdul:
            hdu_s = hdul[0]
            im_s = hdu_s.data            
            cutout_s = Cutout2D(im_s, new_coords, cutout_shape, wcs=wcs) 
                        
            hdu_s.data = cutout_s.data
            hdu_s.data[np.isnan(hdu_s.data)] = 0.    
            hdu_s.header.update(cutout_s.wcs.to_header())
            hdu_s.writeto(fname_out_s, overwrite=True)
        
        ################################################
        #SFFT mask
        
        fname_ref = maindir + 'mask/{}.mask4sfft.fits'.format(ref_name)
        fname_out_r = outdir + 'mask/{}.mask4sfft.fits'.format(ref_name)
        fname_sci = maindir + 'mask/{}.mask4sfft.fits'.format(sci_name)
        fname_out_s = outdir + 'mask/{}.mask4sfft.fits'.format(sci_name)
        
        with fits.open(fname_ref) as hdul:
            hdu_r = hdul[0]
            im_r = hdu_r.data
            cutout_r = Cutout2D(im_r, new_coords, cutout_shape, wcs=wcs)
            
            hdu_r.data = cutout_r.data
            hdu_r.data[np.isnan(hdu_r.data)] = 0.                
            hdu_r.header.update(cutout_r.wcs.to_header())
            hdu_r.writeto(fname_out_r, overwrite=True)


        with fits.open(fname_sci) as hdul:
            hdu_s = hdul[0]
            im_s = hdu_s.data            
            cutout_s = Cutout2D(im_s, new_coords, cutout_shape, wcs=wcs) 
                        
            hdu_s.data = cutout_s.data
            hdu_s.data[np.isnan(hdu_s.data)] = 0.    
            hdu_s.header.update(cutout_s.wcs.to_header())
            hdu_s.writeto(fname_out_s, overwrite=True)
            
        ################################################
        #SFFT mask segmentation map
        
        fname_ref = maindir + 'mask/{}_sexseg.fits'.format(ref_name)
        fname_out_r = outdir + 'mask/{}_sexseg.fits'.format(ref_name)
        fname_sci = maindir + 'mask/{}_sexseg.fits'.format(sci_name)
        fname_out_s = outdir + 'mask/{}_sexseg.fits'.format(sci_name)
        
        with fits.open(fname_ref) as hdul:
            hdu_r = hdul[0]
            im_r = hdu_r.data
            cutout_r = Cutout2D(im_r, new_coords, cutout_shape, wcs=wcs)
            
            hdu_r.data = cutout_r.data
            hdu_r.data[np.isnan(hdu_r.data)] = 0.                
            hdu_r.header.update(cutout_r.wcs.to_header())
            hdu_r.writeto(fname_out_r, overwrite=True)


        with fits.open(fname_sci) as hdul:
            hdu_s = hdul[0]
            im_s = hdu_s.data            
            cutout_s = Cutout2D(im_s, new_coords, cutout_shape, wcs=wcs) 
                        
            hdu_s.data = cutout_s.data
            hdu_s.data[np.isnan(hdu_s.data)] = 0.    
            hdu_s.header.update(cutout_s.wcs.to_header())
            hdu_s.writeto(fname_out_s, overwrite=True)
        

        ##################################################
        #Noise image
        
        fname_ref = maindir + 'noise/{}.noise.fits'.format(ref_name)
        fname_sci = maindir + 'noise/{}.noise.fits'.format(sci_name)
        fname_out_r = outdir + 'noise/{}.noise.fits'.format(ref_name)
        fname_out_s = outdir + 'noise/{}.noise.fits'.format(sci_name)

        with fits.open(fname_ref) as hdul:            
            hdu_r = hdul[0]
            im_rn = hdu_r.data
            cutout_r = Cutout2D(im_rn, new_coords, cutout_shape, wcs=wcs)

            hdu_r.data = cutout_r.data
            hdu_r.data[np.isnan(hdu_r.data)] = 0.
            hdu_r.header.update(cutout_r.wcs.to_header())
            hdu_r.writeto(fname_out_r, overwrite=True)



        with fits.open(fname_sci) as hdul:
            hdu_s = hdul[0]
            im_sn = hdu_s.data            
            cutout_s = Cutout2D(im_sn, new_coords, cutout_shape, wcs=wcs) 
                        
            hdu_s.data = cutout_s.data
            hdu_s.data[np.isnan(hdu_s.data)] = 0.            
            hdu_s.header.update(cutout_s.wcs.to_header())    
            hdu_s.writeto(fname_out_s, overwrite=True)
        
        ##################################################
        #PSF image
        
        fname_ref = maindir + 'psf/{}.psf.fits'.format(ref_name)
        fname_sci = maindir + 'psf/{}.psf.fits'.format(sci_name)
        
        shutil.copy(fname_ref, outdir + 'psf/{}.psf.fits'.format(ref_name))
        shutil.copy(fname_sci, outdir + 'psf/{}.psf.fits'.format(sci_name))
        
        ##################################################
        #Mask image
        
        fname_ref = maindir + 'input/{}.maskin.fits'.format(ref_name)
        fname_sci = maindir + 'input/{}.maskin.fits'.format(sci_name)
        fname_out_r = outdir + 'input/{}.maskin.fits'.format(ref_name)
        fname_out_s = outdir + 'input/{}.maskin.fits'.format(sci_name)
        
        with fits.open(fname_ref) as hdul:            
            hdu_r = hdul[0]
            im_rm = hdu_r.data
            cutout_rm = Cutout2D(im_rm, new_coords, cutout_shape, wcs=wcs)

            hdu_r.data = cutout_rm.data
            hdu_r.data[np.isnan(hdu_r.data)] = 0.
            hdu_r.header.update(cutout_rm.wcs.to_header())
            hdu_r.writeto(fname_out_r, overwrite=True)


        with fits.open(fname_sci) as hdul:
            hdu_s = hdul[0]
            im_sm = hdu_s.data            
            cutout_sm = Cutout2D(im_sm, new_coords, cutout_shape, wcs=wcs) 
                        
            hdu_s.data = cutout_sm.data
            hdu_s.data[np.isnan(hdu_s.data)] = 0.            
            hdu_s.header.update(cutout_sm.wcs.to_header())
            hdu_s.writeto(fname_out_s, overwrite=True)

        im_mask_tot = cutout_rm.data.astype(bool) | cutout_sm.data.astype(bool)
        nnz_frac = np.sum(~im_mask_tot) / (cutout_rm.data.shape[0] * cutout_rm.data.shape[1])
        nnz_fracs.append(nnz_frac)

    if len(ra_vals) > 0:
        # Save the cutout information to a file
        tab = Table()
        tab['RA'] = ra_vals
        tab['DEC'] = dec_vals
        tab['ROW_INDEX'] = row_inds
        tab['COL_INDEX'] = col_inds
        tab['N0'] = shapes_x
        tab['N1'] = shapes_y
        tab['FRAC_NNZ'] = nnz_fracs
        tab['LABEL'] = numbers
        tab.write(maindir + 'cutout_info.txt', format='ascii', overwrite=True)    
    
    return



def make_split_subdir_separate(npx_side, maindir, ref_name, sci_name, npx_boundary=30, 
                               subset=None, dither=None, pp_separate=False,
                               skysub=True, conv_ref=False, conv_sci=False):    

    #Need to take into account that gkerhw pixels from the edge of the image are masked
    #So overlap the cutouts by gkerhw pixels        
    
    npx_boundary = 30

    ra_vals = []
    dec_vals = []
    row_inds = []
    col_inds = []
    shapes_x = []
    shapes_y = []
    numbers = []
    

    fname_ref = maindir + 'input/{}.fits'.format(ref_name)
    fname_sci = maindir + 'input/{}.fits'.format(sci_name)
    fname_mref = maindir + 'input/{}.maskin.fits'.format(ref_name)
    fname_msci = maindir + 'input/{}.maskin.fits'.format(sci_name)
        
    with fits.open(fname_ref) as hdul:
        wcs_ref = WCS(hdul[0].header)
        im_ref = hdul[0].data

    with fits.open(fname_sci) as hdul:
        wcs_sci = WCS(hdul[0].header)    
        im_sci = hdul[0].data
        
    with fits.open(fname_mref) as hdul:
        im_mref = hdul[0].data
    with fits.open(fname_msci) as hdul:
        im_msci = hdul[0].data

    assert im_ref.shape == im_sci.shape, "Reference and science images must have the same shape."
    N0, N1 = im_sci.shape
    wcs = wcs_ref
    
    if dither == 'dec':
        xsplit1 = np.arange(0, N0+npx_side, npx_side)
        xsplit1[-1] = N0
        nsplitx = len(xsplit1)-1

        x1 = int(npx_side/2)
        xsplit2 = np.arange(x1, N0+npx_side, npx_side)
        xsplit2[-1] = N0
        
        xsplit = np.concatenate((xsplit1, xsplit2))   
                     
        ysplit = np.arange(0, N1+npx_side, npx_side)
        ysplit[-1] = N1
        

    elif dither == 'ra':
        ysplit1 = np.arange(0, N1+npx_side, npx_side)
        ysplit1[-1] = N1
        nsplity = len(ysplit1)-1

        y1 = int(npx_side/2)
        ysplit2 = np.arange(y1, N1+npx_side, npx_side)
        ysplit2[-1] = N1
        
        xsplit = np.arange(0, N0+npx_side, npx_side)
        xsplit[-1] = N0
        
        ysplit = np.concatenate((ysplit1, ysplit2))
        
        
    elif dither == 'both':
        xsplit1 = np.arange(0, N0+npx_side, npx_side)
        xsplit1[-1] = N0
        nsplitx = len(xsplit1)-1

        x1 = int(-npx_side/2)
        xsplit2 = np.arange(x1, N0+npx_side, npx_side)
        xsplit2[-1] = N0
        
        xsplit = np.concatenate((xsplit1, xsplit2))   
                     
        ysplit1 = np.arange(0, N1+npx_side, npx_side)
        ysplit1[-1] = N1
        nsplity = len(ysplit1)-1

        y1 = int(-npx_side/2)
        ysplit2 = np.arange(y1, N1+npx_side, npx_side)
        ysplit2[-1] = N1
        
        ysplit = np.concatenate((ysplit1, ysplit2))
        
    else:
        xsplit = np.arange(0, N0+npx_side, npx_side)
        xsplit[-1] = N0
        
        ysplit = np.arange(0, N1+npx_side, npx_side)
        ysplit[-1] = N1
    
    

    x1_vals = []
    x2_vals = []
    y1_vals = []
    y2_vals = []
    dx_vals = []
    dy_vals = []
    n_nz_vals = []

    for i in range(len(xsplit)-1):
        for j in range(len(ysplit)-1):
            n = i*(len(ysplit)-1) + j
        
            x1 = xsplit[i]
            x2 = xsplit[i+1]

            y1 = ysplit[j]
            y2 = ysplit[j+1]
            
            if (x2 < x1) or (y2 < y1):
                continue
            
                    
            im_ij1 = im_sci[x1:x2, y1:y2].copy()
            im_ij2 = im_ref[x1:x2, y1:y2].copy()
            im_ij1_m = im_msci[x1:x2, y1:y2].copy()
            im_ij2_m = im_mref[x1:x2, y1:y2].copy()
            
            x1a = 0
            x2a = im_ij1.shape[0]
            y1a = 0
            y2a = im_ij1.shape[1]
            
            n_nz = np.sum( (~im_ij1_m) & (~im_ij2_m) )
            if (n_nz == 0) or (np.nansum(im_ij1) == 0.) or (np.nansum(im_ij2) == 0.):
                continue
            
            n_nz_vals.append(n_nz)

            y1 = y1 + y1a
            y2 = y1 + y2a
            x1 = x1 + x1a
            x2 = x1 + x2a

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > N0:
                x2 = N0
            if y2 > N1:
                y2 = N1
                
            if (x2-x1 == 0) or (y2-y1 == 0) or (n_nz == 0):
                continue

            x1_vals.append(x1)
            x2_vals.append(x2)
            y1_vals.append(y1)
            y2_vals.append(y2)

  
    x1_vals = np.array(x1_vals, dtype=int)
    x2_vals = np.array(x2_vals, dtype=int)
    y1_vals = np.array(y1_vals, dtype=int)
    y2_vals = np.array(y2_vals, dtype=int)
    n_nz_vals = np.array(n_nz_vals, dtype=int)
    
    x1_vals, x2_vals, y1_vals, y2_vals, nnz_vals = remove_empty_space(x1_vals, x2_vals, y1_vals, y2_vals, im_mref, im_msci)
    for i in range(len(x1_vals)):
        if (x2_vals[i] - x1_vals[i]) < npx_side:
            if x1_vals[i] == 0:
                x2_vals[i] = x1_vals[i] + npx_side
            elif x2_vals[i] == N0:
                x1_vals[i] = x2_vals[i] - npx_side
            elif (x1_vals[i] > N0/2):
                x1_vals[i] = x2_vals[i] - npx_side
            elif (x2_vals[i] < N0/2):
                x2_vals[i] = x1_vals[i] + npx_side
                
        if (y2_vals[i] - y1_vals[i]) < npx_side:
            if y1_vals[i] == 0:
                y2_vals[i] = y1_vals[i] + npx_side
            elif y2_vals[i] == N1:
                y1_vals[i] = y2_vals[i] - npx_side
            elif (y1_vals[i] > N1/2):
                y1_vals[i] = y2_vals[i] - npx_side
            elif (y2_vals[i] < N1/2):
                y2_vals[i] = y1_vals[i] + npx_side

    nnz_fracs = []
    for i in range(len(x1_vals)):
        n = i
        
        if (subset is not None) and (not (n in subset)):
            continue
    
        x1 = x1_vals[i] - npx_boundary
        x2 = x2_vals[i] + npx_boundary

        y1 = y1_vals[i] - npx_boundary
        y2 = y2_vals[i] + npx_boundary

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > N0:
            x2 = N0
        if y2 > N1:
            y2 = N1
        
        if np.nansum(im_sci[x1:x2, y1:y2]) == 0.:
            continue
        if np.nansum(im_ref[x1:x2, y1:y2]) == 0.:
            continue
        
        assert im_sci[x1:x2, y1:y2].shape == im_ref[x1:x2, y1:y2].shape, "Reference and science images must have the same shape."

            
        if (x2-x1 == 0) or (y2-y1 == 0):
            continue
        
        xc = (x1 + x2) / 2.
        yc = (y1 + y2) / 2.
        
        cutout_shape = np.array([x2-x1, y2-y1]) * u.pixel
        
        ra, dec = wcs_ref.array_index_to_world_values(xc, yc)
        ra = float(ra)
        dec = float(dec)
        new_coords = SkyCoord(ra, dec, unit=(u.deg, u.deg))
        
        ra_vals.append(ra)
        dec_vals.append(dec)
        row_inds.append(xc)
        col_inds.append(yc)
        shapes_x.append(x2-x1)
        shapes_y.append(y2-y1)
        numbers.append(n)
        
        outdir = maindir + 'output_{}/'.format(n)            
        if os.path.exists(outdir):
            continue
        
        os.makedirs(outdir, exist_ok=True)
        
        for subdir in ['input/', 'noise/', 'psf/', 'mask/', 'output/']:
            os.makedirs(outdir + subdir, exist_ok=True)    
    
        ################################################
        #Input image
        
        fname_ref = maindir + 'input/{}.fits'.format(ref_name)
        fname_sci = maindir + 'input/{}.fits'.format(sci_name)
        fname_out_r = outdir + 'input/{}.fits'.format(ref_name)
        fname_out_s = outdir + 'input/{}.fits'.format(sci_name)
        
        with fits.open(fname_ref) as hdul:
            hdu_r = hdul[0]
            im_r = hdu_r.data
            cutout_r = Cutout2D(im_r, new_coords, cutout_shape, wcs=wcs)
            
            hdu_r.data = cutout_r.data
            hdu_r.data[np.isnan(hdu_r.data)] = 0.                
            hdu_r.header.update(cutout_r.wcs.to_header())
            hdu_r.writeto(fname_out_r, overwrite=True)


        with fits.open(fname_sci) as hdul:
            hdu_s = hdul[0]
            im_s = hdu_s.data            
            cutout_s = Cutout2D(im_s, new_coords, cutout_shape, wcs=wcs) 
                        
            hdu_s.data = cutout_s.data
            hdu_s.data[np.isnan(hdu_s.data)] = 0.    
            hdu_s.header.update(cutout_s.wcs.to_header())
            hdu_s.writeto(fname_out_s, overwrite=True)
            
            
        if not pp_separate:
            ################################################
            #Input image with skysub
            
            if skysub:
                fname_ref = maindir + 'input/{}.skysub.fits'.format(ref_name)
                fname_sci = maindir + 'input/{}.skysub.fits'.format(sci_name)
                fname_out_r = outdir + 'input/{}.skysub.fits'.format(ref_name)
                fname_out_s = outdir + 'input/{}.skysub.fits'.format(sci_name)
                
                with fits.open(fname_ref) as hdul:
                    hdu_r = hdul[0]
                    im_r = hdu_r.data
                    cutout_r = Cutout2D(im_r, new_coords, cutout_shape, wcs=wcs)
                    
                    hdu_r.data = cutout_r.data
                    hdu_r.data[np.isnan(hdu_r.data)] = 0.                
                    hdu_r.header.update(cutout_r.wcs.to_header())
                    hdu_r.writeto(fname_out_r, overwrite=True)


                with fits.open(fname_sci) as hdul:
                    hdu_s = hdul[0]
                    im_s = hdu_s.data            
                    cutout_s = Cutout2D(im_s, new_coords, cutout_shape, wcs=wcs) 
                                
                    hdu_s.data = cutout_s.data
                    hdu_s.data[np.isnan(hdu_s.data)] = 0.    
                    hdu_s.header.update(cutout_s.wcs.to_header())
                    hdu_s.writeto(fname_out_s, overwrite=True)
                        
                        
            ################################################
            #Cross-convolved image
            
            if conv_ref:
                fname_ref = maindir + 'output/{}.crossconvd.fits'.format(ref_name)
                fname_out_r = outdir + 'output/{}.crossconvd.fits'.format(ref_name)
                
                with fits.open(fname_ref) as hdul:
                    hdu_r = hdul[0]
                    im_r = hdu_r.data
                    cutout_r = Cutout2D(im_r, new_coords, cutout_shape, wcs=wcs)
                    
                    hdu_r.data = cutout_r.data
                    hdu_r.data[np.isnan(hdu_r.data)] = 0.                
                    hdu_r.header.update(cutout_r.wcs.to_header())
                    hdu_r.writeto(fname_out_r, overwrite=True)


            if conv_sci:
                fname_sci = maindir + 'output/{}.crossconvd.fits'.format(sci_name)
                fname_out_s = outdir + 'output/{}.crossconvd.fits'.format(sci_name)

                with fits.open(fname_sci) as hdul:
                    hdu_s = hdul[0]
                    im_s = hdu_s.data            
                    cutout_s = Cutout2D(im_s, new_coords, cutout_shape, wcs=wcs) 
                                
                    hdu_s.data = cutout_s.data
                    hdu_s.data[np.isnan(hdu_s.data)] = 0.    
                    hdu_s.header.update(cutout_s.wcs.to_header())
                    hdu_s.writeto(fname_out_s, overwrite=True)
        
        ##################################################
        #Noise image
        
        fname_ref = maindir + 'noise/{}.noise.fits'.format(ref_name)
        fname_sci = maindir + 'noise/{}.noise.fits'.format(sci_name)
        fname_out_r = outdir + 'noise/{}.noise.fits'.format(ref_name)
        fname_out_s = outdir + 'noise/{}.noise.fits'.format(sci_name)

        with fits.open(fname_ref) as hdul:            
            hdu_r = hdul[0]
            im_rn = hdu_r.data
            cutout_r = Cutout2D(im_rn, new_coords, cutout_shape, wcs=wcs)

            hdu_r.data = cutout_r.data
            hdu_r.data[np.isnan(hdu_r.data)] = 0.
            hdu_r.header.update(cutout_r.wcs.to_header())
            hdu_r.writeto(fname_out_r, overwrite=True)



        with fits.open(fname_sci) as hdul:
            hdu_s = hdul[0]
            im_sn = hdu_s.data            
            cutout_s = Cutout2D(im_sn, new_coords, cutout_shape, wcs=wcs) 
                        
            hdu_s.data = cutout_s.data
            hdu_s.data[np.isnan(hdu_s.data)] = 0.            
            hdu_s.header.update(cutout_s.wcs.to_header())    
            hdu_s.writeto(fname_out_s, overwrite=True)
        
        ##################################################
        #PSF image
        
        fname_ref = maindir + 'psf/{}.psf.fits'.format(ref_name)
        fname_sci = maindir + 'psf/{}.psf.fits'.format(sci_name)
        
        shutil.copy(fname_ref, outdir + 'psf/{}.psf.fits'.format(ref_name))
        shutil.copy(fname_sci, outdir + 'psf/{}.psf.fits'.format(sci_name))
        
        ##################################################
        #Mask image
        
        fname_ref = maindir + 'input/{}.maskin.fits'.format(ref_name)
        fname_sci = maindir + 'input/{}.maskin.fits'.format(sci_name)
        fname_out_r = outdir + 'input/{}.maskin.fits'.format(ref_name)
        fname_out_s = outdir + 'input/{}.maskin.fits'.format(sci_name)
        
        with fits.open(fname_ref) as hdul:            
            hdu_r = hdul[0]
            im_rm = hdu_r.data
            cutout_rm = Cutout2D(im_rm, new_coords, cutout_shape, wcs=wcs)

            hdu_r.data = cutout_rm.data
            hdu_r.data[np.isnan(hdu_r.data)] = 0.
            hdu_r.header.update(cutout_rm.wcs.to_header())
            hdu_r.writeto(fname_out_r, overwrite=True)


        with fits.open(fname_sci) as hdul:
            hdu_s = hdul[0]
            im_sm = hdu_s.data            
            cutout_sm = Cutout2D(im_sm, new_coords, cutout_shape, wcs=wcs) 
                        
            hdu_s.data = cutout_sm.data
            hdu_s.data[np.isnan(hdu_s.data)] = 0.            
            hdu_s.header.update(cutout_sm.wcs.to_header())
            hdu_s.writeto(fname_out_s, overwrite=True)

        im_mask_tot = cutout_rm.data.astype(bool) | cutout_sm.data.astype(bool)
        nnz_frac = np.sum(~im_mask_tot) / (cutout_rm.data.shape[0] * cutout_rm.data.shape[1])
        nnz_fracs.append(nnz_frac)

    if len(ra_vals) > 0:
        # Save the cutout information to a file
        tab = Table()
        tab['RA'] = ra_vals
        tab['DEC'] = dec_vals
        tab['ROW_INDEX'] = row_inds
        tab['COL_INDEX'] = col_inds
        tab['N0'] = shapes_x
        tab['N1'] = shapes_y
        tab['FRAC_NNZ'] = nnz_fracs
        tab['LABEL'] = numbers
        tab.write(maindir + 'cutout_info.txt', format='ascii', overwrite=True)    
    
    return