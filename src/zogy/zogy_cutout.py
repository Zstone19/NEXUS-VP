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


def divide_cutout_by4(x1_vals, x2_vals, y1_vals, y2_vals, im_mref, im_msci, n_nz_vals, npx_min=1000, nz_thresh=.6):
    
    x1_new = []
    x2_new = []
    y1_new = []
    y2_new = []
    
    for i in range(len(x1_vals)):
        if n_nz_vals[i] > nz_thresh*npx_min*npx_min*4:
            x1 = x1_vals[i]
            x2 = x2_vals[i]
            y1 = y1_vals[i]
            y2 = y2_vals[i]
            
            dx = x2-x1
            dy = y2-y1

            #Lower left
            y1_ll = y1
            y2_ll = int(y1 + dy/2.)
            x1_ll = x1
            x2_ll = int(x1 + dx/2.)
            im_ij1_m = im_msci[x1_ll:x2_ll, y1_ll:y2_ll].copy()
            im_ij2_m = im_mref[x1_ll:x2_ll, y1_ll:y2_ll].copy()
            n_nz_ll = np.sum( (im_ij1_m == 0) & (im_ij2_m == 0) )

            #Lower right
            y1_lr = int(y1 + dy/2.)
            y2_lr = y2
            x1_lr = x1
            x2_lr = int(x1 + dx/2.)
            im_ij1_m = im_msci[x1_lr:x2_lr, y1_lr:y2_lr].copy()
            im_ij2_m = im_mref[x1_lr:x2_lr, y1_lr:y2_lr].copy()
            n_nz_lr = np.sum( (im_ij1_m == 0) & (im_ij2_m == 0) )

            #Upper left
            y1_ul = y1
            y2_ul = int(y1 + dy/2.)
            x1_ul = int(x1 + dx/2.)
            x2_ul = x2
            im_ij1_m = im_msci[x1_ul:x2_ul, y1_ul:y2_ul].copy()
            im_ij2_m = im_mref[x1_ul:x2_ul, y1_ul:y2_ul].copy()
            n_nz_ul = np.sum( (im_ij1_m == 0) & (im_ij2_m == 0) )
            
            #Upper right
            y1_ur = int(y1 + dy/2.)
            y2_ur = y2
            x1_ur = int(x1 + dx/2.)
            x2_ur = x2
            im_ij1_m = im_msci[x1_ur:x2_ur, y1_ur:y2_ur].copy()
            im_ij2_m = im_mref[x1_ur:x2_ur, y1_ur:y2_ur].copy()
            n_nz_ur = np.sum( (im_ij1_m == 0) & (im_ij2_m == 0) )
            
            #Lower both
            y1_lo = min(y1_ll, y1_lr)
            y2_lo = max(y2_ll, y2_lr)
            x1_lo = min(x1_ll, x1_lr)
            x2_lo = max(x2_ll, x2_lr)
            im_ij1_m = im_msci[x1_lo:x2_lo, y1_lo:y2_lo].copy()
            im_ij2_m = im_mref[x1_lo:x2_lo, y1_lo:y2_lo].copy()
            n_nz_lo = np.sum( (im_ij1_m == 0) & (im_ij2_m == 0) )
            
            #Upper both
            y1_u = min(y1_ul, y1_ur)
            y2_u = max(y2_ul, y2_ur)
            x1_u = min(x1_ul, x1_ur)
            x2_u = max(x2_ul, x2_ur)
            im_ij1_m = im_msci[x1_u:x2_u, y1_u:y2_u].copy()
            im_ij2_m = im_mref[x1_u:x2_u, y1_u:y2_u].copy()
            n_nz_u = np.sum( (im_ij1_m == 0) & (im_ij2_m == 0) )
            
            #Left both
            x1_l = min(x1_ll, x1_ul)
            x2_l = max(x2_ll, x2_ul)
            y1_l = min(y1_ll, y1_ul)
            y2_l = max(y2_ll, y2_ul)
            im_ij1_m = im_msci[x1_l:x2_l, y1_l:y2_l].copy()
            im_ij2_m = im_mref[x1_l:x2_l, y1_l:y2_l].copy()
            n_nz_l = np.sum( (im_ij1_m == 0) & (im_ij2_m == 0) )
            
            #Right both
            x1_r = min(x1_lr, x1_ur)
            x2_r = max(x2_lr, x2_ur)
            y1_r = min(y1_lr, y1_ur)
            y2_r = max(y2_lr, y2_ur)
            im_ij1_m = im_msci[x1_r:x2_r, y1_r:y2_r].copy()
            im_ij2_m = im_mref[x1_r:x2_r, y1_r:y2_r].copy()
            n_nz_r = np.sum( (im_ij1_m == 0) & (im_ij2_m == 0) )
            
            
            #All satisfy
            if (n_nz_ll > nz_thresh*npx_min*npx_min) and (n_nz_lr > nz_thresh*npx_min*npx_min) and (n_nz_ul > nz_thresh*npx_min*npx_min) and (n_nz_ur > nz_thresh*npx_min*npx_min):
                #Lower left
                x1_new.append(x1_ll)
                x2_new.append(x2_ll)
                y1_new.append(y1_ll)
                y2_new.append(y2_ll)

                
                #Lower right
                x1_new.append(x1_lr)
                x2_new.append(x2_lr)
                y1_new.append(y1_lr)
                y2_new.append(y2_lr)

                
                #Upper left
                x1_new.append(x1_ul)
                x2_new.append(x2_ul)
                y1_new.append(y1_ul)
                y2_new.append(y2_ul)

                
                #Upper right
                x1_new.append(x1_ur)
                x2_new.append(x2_ur)
                y1_new.append(y1_ur)
                y2_new.append(y2_ur)
                
                
             
            #Only lower satisfy
            elif (n_nz_ll > nz_thresh*npx_min*npx_min) and (n_nz_lr > nz_thresh*npx_min*npx_min) and (n_nz_u > nz_thresh*npx_min*npx_min):
                #Lower left
                x1_new.append(x1_ll)
                x2_new.append(x2_ll)
                y1_new.append(y1_ll)
                y2_new.append(y2_ll)
                
                #Lower right
                x1_new.append(x1_lr)
                x2_new.append(x2_lr)
                y1_new.append(y1_lr)
                y2_new.append(y2_lr)
                
                #Upper 
                x1_new.append( min(x1_ul, x1_ur) )
                x2_new.append( max(x2_ul, x2_ur) )
                y1_new.append( min(y1_ul, y1_ur) )
                y2_new.append( max(y2_ul, y2_ur) )

            
            #Only upper satisfy
            elif (n_nz_ul > nz_thresh*npx_min*npx_min) and (n_nz_ur > nz_thresh*npx_min*npx_min) and (n_nz_lo > nz_thresh*npx_min*npx_min):
                #Upper left
                x1_new.append(x1_ul)
                x2_new.append(x2_ul)
                y1_new.append(y1_ul)
                y2_new.append(y2_ul)
                
                #Upper right
                x1_new.append(x1_ur)
                x2_new.append(x2_ur)
                y1_new.append(y1_ur)
                y2_new.append(y2_ur)
                
                #Lower 
                x1_new.append( min(x1_ll, x1_lr) )
                x2_new.append( max(x2_ll, x2_lr) )
                y1_new.append( min(y1_ll, y1_lr) )
                y2_new.append( max(y2_ll, y2_lr) )
                
                
            #Only left satisfy
            elif (n_nz_ll > nz_thresh*npx_min*npx_min) and (n_nz_ul > nz_thresh*npx_min*npx_min) and (n_nz_r > nz_thresh*npx_min*npx_min):
                #Lower left
                x1_new.append(x1_ll)
                x2_new.append(x2_ll)
                y1_new.append(y1_ll)
                y2_new.append(y2_ll)
                
                #Upper left
                x1_new.append(x1_ul)
                x2_new.append(x2_ul)
                y1_new.append(y1_ul)
                y2_new.append(y2_ul)

                #Right 
                x1_new.append( min(x1_lr, x1_ur) )
                x2_new.append( max(x2_lr, x2_ur) )
                y1_new.append( min(y1_lr, y1_ur) )
                y2_new.append( max(y2_lr, y2_ur) )

                
            #Only right satisfy
            elif (n_nz_lr > nz_thresh*npx_min*npx_min) and (n_nz_ur > nz_thresh*npx_min*npx_min) and (n_nz_l > nz_thresh*npx_min*npx_min):
                #Lower right
                x1_new.append(x1_lr)
                x2_new.append(x2_lr)
                y1_new.append(y1_lr)
                y2_new.append(y2_lr)

                
                #Upper right
                x1_new.append(x1_ur)
                x2_new.append(x2_ur)
                y1_new.append(y1_ur)
                y2_new.append(y2_ur)

                
                #Left 
                x1_new.append( min(x1_ll, x1_ul) )
                x2_new.append( max(x2_ll, x2_ul) )
                y1_new.append( min(y1_ll, y1_ul) )
                y2_new.append( max(y2_ll, y2_ul) )
                

            #Divide lower and upper
            elif (n_nz_lo > nz_thresh*npx_min*npx_min) and (n_nz_u > nz_thresh*npx_min*npx_min):
                #Lower 
                x1_new.append( min(x1_ll, x1_lr) )
                x2_new.append( max(x2_ll, x2_lr) )
                y1_new.append( min(y1_ll, y1_lr) )
                y2_new.append( max(y2_ll, y2_lr) )

                
                #Upper 
                x1_new.append( min(x1_ul, x1_ur) )
                x2_new.append( max(x2_ul, x2_ur) )
                y1_new.append( min(y1_ul, y1_ur) )
                y2_new.append( max(y2_ul, y2_ur) )


            #Divide left and right
            elif (n_nz_l > nz_thresh*npx_min*npx_min) and (n_nz_r > nz_thresh*npx_min*npx_min):
                #Left 
                x1_new.append( min(x1_ll, x1_ul) )
                x2_new.append( max(x2_ll, x2_ul) )
                y1_new.append( min(y1_ll, y1_ul) )
                y2_new.append( max(y2_ll, y2_ul) )

                
                #Right 
                x1_new.append( min(x1_lr, x1_ur) )
                x2_new.append( max(x2_lr, x2_ur) )
                y1_new.append( min(y1_lr, y1_ur) )
                y2_new.append( max(y2_lr, y2_ur) )

            
            #None satisfy
            else:
                x1_new.append(x1_vals[i])
                x2_new.append(x2_vals[i])
                y1_new.append(y1_vals[i])
                y2_new.append(y2_vals[i])

            
        else:            
            x1_new.append(x1_vals[i])
            x2_new.append(x2_vals[i])
            y1_new.append(y1_vals[i])
            y2_new.append(y2_vals[i])
    

    x1_new = np.array(x1_new, dtype=int)
    x2_new = np.array(x2_new, dtype=int)
    y1_new = np.array(y1_new, dtype=int)
    y2_new = np.array(y2_new, dtype=int)
    
    return x1_new, x2_new, y1_new, y2_new




def make_split_subdir_separate(npx_side, maindir, ref_name, sci_name, dither=None, npx_min=1000, nz_thresh=.6):    

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
        im_mref = hdul[0].data.astype(bool)
    with fits.open(fname_msci) as hdul:
        im_msci = hdul[0].data.astype(bool)


    N0, N1 = im_sci.shape
    
    if dither == 'dec':
        xsplit1 = np.arange(0, N0+npx_side, npx_side)
        xsplit1[-1] = N0
        nsplitx = len(xsplit1)-1

        x1 = int(-npx_side/2)
        xsplit2 = np.arange(x1, N0+npx_side, npx_side)
        xsplit2[-1] = N0
        
        xsplit = np.concatenate((xsplit1, xsplit2))   
                     
        ysplit = np.arange(0, N1+npx_side, npx_side)
        ysplit[-1] = N1
        

    elif dither == 'ra':
        ysplit1 = np.arange(0, N1+npx_side, npx_side)
        ysplit1[-1] = N1
        nsplity = len(ysplit1)-1

        y1 = int(-npx_side/2)
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
        xsplit[-2] = N0
        
        ysplit = np.arange(0, N1+npx_side, npx_side)
        ysplit[-2] = N1
        
        xsplit = xsplit[:-1]
        ysplit = ysplit[:-1]
    
    
    
    x1_vals = []
    x2_vals = []
    y1_vals = []
    y2_vals = []
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
    

    
    #Account for empty space
    x1_vals, x2_vals, y1_vals, y2_vals, n_nz_vals = remove_empty_space(x1_vals, x2_vals, y1_vals, y2_vals, im_mref, im_msci)
    xc_vals = (x1_vals + x2_vals) / 2.
    yc_vals = (y1_vals + y2_vals) / 2.
    dx_vals = x2_vals - x1_vals
    dy_vals = y2_vals - y1_vals
    
    x1_vals_new = x1_vals.copy()
    x2_vals_new = x2_vals.copy()
    y1_vals_new = y1_vals.copy()
    y2_vals_new = y2_vals.copy()
    
    
    #Repeate division until number of cutouts is consistent
    nold = len(x1_vals)
    nnew = nold + 1
    while np.abs(nold - nnew) > 0:
        x1_vals_old = x1_vals_new.copy()
        x2_vals_old = x2_vals_new.copy()
        y1_vals_old = y1_vals_new.copy()
        y2_vals_old = y2_vals_new.copy()
        nold = len(x1_vals_old)
        
        #If any cutouts have (n_nz > .75*npx_min*npx_min*4), split into 4 cutouts
        x1_vals, x2_vals, y1_vals, y2_vals = divide_cutout_by4(x1_vals_old, x2_vals_old, y1_vals_old, y2_vals_old, im_mref, im_msci, n_nz_vals, npx_min=npx_min, nz_thresh=nz_thresh)
        xc_vals = (x1_vals + x2_vals) / 2.
        yc_vals = (y1_vals + y2_vals) / 2.
        dx_vals = x2_vals - x1_vals
        dy_vals = y2_vals - y1_vals
        
        #Account for empty space again
        x1_vals_new, x2_vals_new, y1_vals_new, y2_vals_new, n_nz_vals = remove_empty_space(x1_vals, x2_vals, y1_vals, y2_vals, im_mref, im_msci)
        xc_vals_new = (x1_vals_new + x2_vals_new) / 2.
        yc_vals_new = (y1_vals_new + y2_vals_new) / 2.
        dx_vals_new = x2_vals_new - x1_vals_new
        dy_vals_new = y2_vals_new - y1_vals_new
        
        nnew = len(x1_vals_new)
        
        print('Iteration: {} -> {}'.format(nold, nnew))
        
    x1_vals = x1_vals_new.copy()
    x2_vals = x2_vals_new.copy()
    y1_vals = y1_vals_new.copy()
    y2_vals = y2_vals_new.copy()
    xc_vals = xc_vals_new.copy()
    yc_vals = yc_vals_new.copy()
    dx_vals = dx_vals_new.copy()
    dy_vals = dy_vals_new.copy()

    #Get cutouts            
    for i in range(len(xc_vals)):
        xc = xc_vals[i]
        yc = yc_vals[i]           
        dx = dx_vals[i]
        dy = dy_vals[i]

        cutout_shape = np.array([dx, dy]) * u.pixel
        
        ra, dec = wcs_ref.array_index_to_world_values(xc, yc)
        ra = float(ra)
        dec = float(dec)
        new_coords = SkyCoord(ra, dec, unit=(u.deg, u.deg))
        
        ra_vals.append(ra)
        dec_vals.append(dec)
        row_inds.append(xc)
        col_inds.append(yc)
        shapes_x.append(dx)
        shapes_y.append(dy)
        numbers.append(i)
        
        outdir = maindir + 'output_{}/'.format(i)            
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
            cutout_r = Cutout2D(im_r, new_coords, cutout_shape, wcs=wcs_ref)
            
            hdu_r.data = cutout_r.data
            hdu_r.data[np.isnan(hdu_r.data)] = 0.                
            hdu_r.header.update(cutout_r.wcs.to_header())
            hdu_r.writeto(fname_out_r, overwrite=True)


        with fits.open(fname_sci) as hdul:
            hdu_s = hdul[0]
            im_s = hdu_s.data            
            cutout_s = Cutout2D(im_s, new_coords, cutout_shape, wcs=wcs_sci) 
                        
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
            im_r = hdu_r.data
            cutout_r = Cutout2D(im_r, new_coords, cutout_shape, wcs=wcs_ref)

            hdu_r.data = cutout_r.data
            hdu_r.data[np.isnan(hdu_r.data)] = 0.
            hdu_r.header.update(cutout_r.wcs.to_header())
            hdu_r.writeto(fname_out_r, overwrite=True)



        with fits.open(fname_sci) as hdul:
            hdu_s = hdul[0]
            im_s = hdu_s.data            
            cutout_s = Cutout2D(im_s, new_coords, cutout_shape, wcs=wcs_sci) 
                        
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
            im_r = hdu_r.data
            cutout_r = Cutout2D(im_r, new_coords, cutout_shape, wcs=wcs_ref)

            hdu_r.data = cutout_r.data
            hdu_r.data[np.isnan(hdu_r.data)] = 0.
            hdu_r.header.update(cutout_r.wcs.to_header())
            hdu_r.writeto(fname_out_r, overwrite=True)


        with fits.open(fname_sci) as hdul:
            hdu_s = hdul[0]
            im_s = hdu_s.data            
            cutout_s = Cutout2D(im_s, new_coords, cutout_shape, wcs=wcs_sci) 
                        
            hdu_s.data = cutout_s.data
            hdu_s.data[np.isnan(hdu_s.data)] = 0.            
            hdu_s.header.update(cutout_s.wcs.to_header())
            hdu_s.writeto(fname_out_s, overwrite=True)

    if len(ra_vals) > 0:
        # Save the cutout information to a file
        tab = Table()
        tab['RA'] = ra_vals
        tab['DEC'] = dec_vals
        tab['ROW_INDEX'] = row_inds
        tab['COL_INDEX'] = col_inds
        tab['N0'] = shapes_x
        tab['N1'] = shapes_y
        tab['LABEL'] = numbers
        tab.write(maindir + 'cutout_info.txt', format='ascii', overwrite=True)    
    
    return