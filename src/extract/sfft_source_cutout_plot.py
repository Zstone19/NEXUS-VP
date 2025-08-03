import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Ellipse
import matplotlib as mpl
from astropy.visualization.wcsaxes import add_scalebar


import os
import shutil

import numpy as np
from astropy.table import Table, vstack
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.convolution import convolve_fft

from compile_catalog import get_catalog_allcutout, make_full_catalog, get_total_catalog

import sys
sys.path.append('../pipeline/src/sfft/')
from sfft_utils import get_background

###################################################################################################
###################################################################################################

def format_flux(v, err):
    v2 = v
    err = err
    return r'{0:.2f} $\pm$ {1:.2f}'.format(v2, err)

def get_ellipse(dx, cxx, cxy, cyy, r=3):
    pos = ( -cxy*dx + np.sqrt( cyy**2 - 4*cyy*(cxx*(dx**2) - r**2) ) )/2/cyy
    neg = ( -cxy*dx - np.sqrt( cyy**2 - 4*cyy*(cxx*(dx**2) - r**2) ) )/2/cyy
    
    xout = np.concatenate([dx, [np.nan], dx])
    yout = np.concatenate([pos, [np.nan], neg])
    return xout, yout

def get_ellipse_mask(dxvals, dyvals, cxx, cxy, cyy, r=3):
    t1 = cxx * (dxvals**2)
    t2 = cyy * (dyvals**2)
    t3 = 2*cxy*dxvals*dyvals
    
    mask_tot = t1 + t2 + t3 <= (r**2)
    return mask_tot


def flux_to_mag(f, zp=8.9):
    f = np.abs(f)
    return -2.5*np.log10(f) + zp

def flux_to_mag_w_err(f, ferr, zp=8.9):
    f = np.abs(f)
    ferr = np.abs(ferr)
    
    m = flux_to_mag(f, zp)
    dm = 2.5 * ferr / (f * np.log(10))
    return m, dm

###################################################################################################
###################################################################################################

def make_source_cutouts_simple(catalog, maindir_source, fname_jwst, inds, 
                               dx_cutout=3, dy_cutout=3, unit='Jy',
                               show_flux=False, show_aperture=False, show_ct_conversion=False,
                               output_fname=None, show=False):
    
    if np.max(inds) >= len(catalog):
        inds_new = []
        for i in inds:
            if i < len(catalog):
                inds_new.append(i)
        inds = inds_new
    
    
    
    dx = dx_cutout*u.arcsec
    dy = dy_cutout*u.arcsec

    ct2microjy_jwst = 0.021153987485188146
    ct2microjy_euclid = 0.0036307805477010027

    dn2microjy_jwst = 1.0550972810012904e-05
    e2microjy_euclid = 0.0036307805477010027

    gain_euclid = 2 #From: https://www.aanda.org/articles/aa/full_html/2025/05/aa50803-24/aa50803-24.html
    ct2microjy_euclid = e2microjy_euclid / gain_euclid

    if unit == 'Jy':
        div_factor = 1e-6
    else:
        div_factor = 1



    jwst_scale = 10/3
    jwst_ps = .03
    euclid_ps = 0.1

    Nrow = min(5*3, len(inds))
    Ncol = 3
    
    Nrow_gs = Nrow // 3 + 1
    Ncol_gs = 3
    
    fig = plt.figure(figsize=(3*Ncol*Ncol_gs, 3*Nrow_gs))
    gs = GridSpec(Nrow_gs, Ncol_gs, wspace=.05, hspace=0.)


    for i, n in enumerate(inds):
        k = i // 3
        l = i % 3
        
        gs_i = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[k, l], hspace=0., wspace=0.)
        ax0 = plt.subplot(gs_i[0, 0])
        ax1 = plt.subplot(gs_i[0, 1])
        ax2 = plt.subplot(gs_i[0, 2])
        ax = np.array([ax0, ax1, ax2], dtype=object)
        
        
        cutout_num = catalog['CUTOUT'][n]
        im_in_r = fits.open(maindir_source + 'sources_{}/REF.fits'.format(cutout_num))[0].data
        im_in_s = fits.open(maindir_source + 'sources_{}/SCI.fits'.format(cutout_num))[0].data
        im_out = fits.open(maindir_source + 'sources_{}/DIFF.fits'.format(cutout_num))[0].data
        im_out_n = fits.open(maindir_source + 'sources_{}/DIFF.neg.fits'.format(cutout_num))[0].data
        
        wcs_in = WCS(fits.open(maindir_source + 'sources_{}/REF.fits'.format(cutout_num))[0].header)
        
        
        
        ra_s = catalog['RA_SCI_INPUT'][n]
        dec_s = catalog['DEC_SCI_INPUT'][n]
        xs, ys = wcs_in.wcs_world2pix(ra_s, dec_s, 0)
        
        ra_r = catalog['RA_REF_INPUT'][n]
        dec_r = catalog['DEC_REF_INPUT'][n]
        xr, yr = wcs_in.wcs_world2pix(ra_r, dec_r, 0)
        
        ra_d = catalog['RA_DIFF'][n]
        dec_d = catalog['DEC_DIFF'][n]
        xd, yd = wcs_in.wcs_world2pix(ra_d, dec_d, 0)
        
        ra_d2 = catalog['RA_DIFF_n'][n]
        dec_d2 = catalog['DEC_DIFF_n'][n]
        xd2, yd2 = wcs_in.wcs_world2pix(ra_d2, dec_d2, 0)
        
        ref_detected = True
        if np.isnan(catalog[n]['FLUX_AUTO_REF_INPUT']) or isinstance(catalog[n]['FLUX_AUTO_REF_INPUT'], np.ma.core.MaskedConstant):
            ref_detected = False
        
        sci_detected = True
        if np.isnan(catalog[n]['FLUX_AUTO_SCI_INPUT']) or isinstance(catalog[n]['FLUX_AUTO_SCI_INPUT'], np.ma.core.MaskedConstant):
            sci_detected = False

        diff_detected = True
        if (not catalog[n]['DIFF_DETECTED']) and (not catalog[n]['DIFF_DETECTED_n']):
            diff_detected = False

        x = xs
        y = ys
        ra = ra_s
        dec = dec_s
        if np.isnan(ra) or isinstance(ra, np.ma.core.MaskedConstant):
            x = xr
            y = yr
            ra = ra_r
            dec = dec_r
        if np.isnan(ra) or isinstance(ra, np.ma.core.MaskedConstant):
            x = xd
            y = yd
            ra = ra_d
            dec = dec_d
        if np.isnan(ra) or isinstance(ra, np.ma.core.MaskedConstant):
            x = xd2
            y = yd2
            ra = ra_d2
            dec = dec_d2
            
                
        ##############################################################################################################################
        #Get Kron radius

        if show_aperture:
            rad_r = catalog['KRON_RADIUS_REF'][n]
            rad_r2 = catalog['KRON_RADIUS_REF_n'][n]
            rad_s = catalog['KRON_RADIUS_SCI'][n]
            rad_s2 = catalog['KRON_RADIUS_SCI_n'][n]
            rad_d = catalog['KRON_RADIUS_DIFF'][n]
            rad_d2 = catalog['KRON_RADIUS_DIFF_n'][n]
            rad_r_in = catalog['KRON_RADIUS_REF_INPUT'][n]
            rad_s_in = catalog['KRON_RADIUS_SCI_INPUT'][n]
            
            if np.isnan(rad_r) or (rad_r == 0.) or isinstance(rad_r, np.ma.core.MaskedConstant):
                rad_r = rad_r2
            if np.isnan(rad_s) or (rad_s == 0.) or isinstance(rad_s, np.ma.core.MaskedConstant):
                rad_s = rad_s2
            if np.isnan(rad_d) or (rad_d == 0.) or isinstance(rad_d, np.ma.core.MaskedConstant):
                rad_d = rad_d2
            
        ##############################################################################################################################
        #Get fluxes/errors    
        
        if show_flux:
            m_r1 = catalog['FLUX_AUTO_REF'][n]/div_factor
            m_r2 = catalog['FLUX_AUTO_REF_n'][n]/div_factor
            merr_r1 = catalog['FLUXERR_AUTO_REF'][n]/div_factor
            merr_r2 = catalog['FLUXERR_AUTO_REF_n'][n]/div_factor
            m_r = m_r1
            merr_r = merr_r1
            # m_rvals = np.array([m_r1, m_r2])
            # m_rvals[m_rvals == 99.] = np.nan
            # m_r = np.nanmedian(m_rvals)


            m_s1 = catalog['FLUX_AUTO_SCI'][n]/div_factor
            m_s2 = catalog['FLUX_AUTO_SCI_n'][n]/div_factor
            merr_s1 = catalog['FLUXERR_AUTO_SCI'][n]/div_factor
            merr_s2 = catalog['FLUXERR_AUTO_SCI_n'][n]/div_factor
            m_s = m_s1
            merr_s = merr_s1

            m_d1 = catalog['FLUX_AUTO_DIFF'][n]/div_factor
            m_d2 = catalog['FLUX_AUTO_DIFF_n'][n]/div_factor
            merr_d1 = catalog['FLUXERR_AUTO_DIFF'][n]/div_factor
            merr_d2 = catalog['FLUXERR_AUTO_DIFF_n'][n]/div_factor
            if np.isnan(m_d1) or isinstance(m_d1, np.ma.core.MaskedConstant):
                m_d = -m_d2
                merr_d = merr_d2
            else:
                m_d = m_d1
                merr_d = merr_d1

            m_r_og = catalog['FLUX_AUTO_REF_INPUT'][n]/div_factor
            merr_r_og = catalog['FLUXERR_AUTO_REF_INPUT'][n]/div_factor
            m_rs_og = catalog['FLUX_AUTO_REF_SCIAPER'][n]/div_factor
            merr_rs_og = catalog['FLUXERR_AUTO_REF_SCIAPER'][n]/div_factor
            m_s_og = catalog['FLUX_AUTO_SCI_INPUT'][n]/div_factor
            merr_s_og = catalog['FLUXERR_AUTO_SCI_INPUT'][n]/div_factor
            m_sr_og = catalog['FLUX_AUTO_SCI_REFAPER'][n]/div_factor
            merr_sr_og = catalog['FLUXERR_AUTO_SCI_REFAPER'][n]/div_factor
        
        ##############################################################################################################################
        #Get source cutouts
        
        coord = SkyCoord(ra, dec, unit=(u.deg, u.deg))
        im_r_i = Cutout2D(im_in_r, coord, size=(dx*2, dy*2), wcs=wcs_in)
        im_s_i = Cutout2D(im_in_s, coord, size=(dx*2, dy*2), wcs=wcs_in)
        im_d_i = Cutout2D(im_out, coord, size=(dx*2, dy*2), wcs=wcs_in)
        
        ##############################################################################################################################
        #Get center of cutout

        if ra == ra_r:
            im_use = im_r_i
        elif ra == ra_s:
            im_use = im_s_i
        elif ra == ra_d:
            im_use = im_d_i
        elif ra == ra_d2:
            im_use = im_d_i
        
        xs_c, ys_c = im_use.wcs.wcs_world2pix(ra_s, dec_s, 0)
        xr_c, yr_c = im_use.wcs.wcs_world2pix(ra_r, dec_r, 0)
        xd_c, yd_c = im_use.wcs.wcs_world2pix(ra_d, dec_d, 0)
        xd2_c, yd2_c = im_use.wcs.wcs_world2pix(ra_d2, dec_d2, 0)

        im_r_i = im_r_i.data
        im_s_i = im_s_i.data
        im_d_i = im_d_i.data
        

        
        
        dx_px = int(dx.value / euclid_ps)
        dy_px = int(dy.value / euclid_ps)
        x1 = int(  max(x-dx_px, 0)  )
        x2 = int(  min(x+dx_px, im_in_r.shape[1])  )
        y1 = int(  max(y-dy_px, 0)  )
        y2 = int(  min(y+dy_px, im_in_r.shape[0])  )
        
        dxc_r = xr_c - dx_px
        dyc_r = yr_c - dy_px
        dxc_s = xs_c - dx_px
        dyc_s = ys_c - dy_px
        dxc_d = xd_c - dx_px
        dyc_d = yd_c - dy_px
        dxc_d2 = xd2_c - dx_px
        dyc_d2 = yd2_c - dy_px   
        
        if np.isnan(dxc_d):
            dxc_d_use = dxc_d2
            dyc_d_use = dyc_d2
        else:
            dxc_d_use = dxc_d
            dyc_d_use = dyc_d   
        
        ##############################################################################################################################
        #Correct if on the edge of the image

        if ((x2-x1) < dx_px*2) or ((y2-y1) < dy_px*2):   
            im_r_new = np.full((dy_px*2, dx_px*2), np.nan)
            im_r_new[:im_r_i.shape[0], :im_r_i.shape[1]] = im_r_i
            im_r_i = im_r_new.copy()
            
            im_s_new = np.full((dy_px*2, dx_px*2), np.nan)
            im_s_new[:im_s_i.shape[0], :im_s_i.shape[1]] = im_s_i
            im_s_i = im_s_new.copy()
            
            im_d_new = np.full((dy_px*2, dx_px*2), np.nan)
            im_d_new[:im_d_i.shape[0], :im_d_i.shape[1]] = im_d_i
            im_d_i = im_d_new.copy()              

            dx_empty = dx_px*2 - (x2-x1)
            dy_empty = dy_px*2 - (y2-y1)   
            
        else:
            dx_empty = 0
            dy_empty = 0     
            dx_empty_jwst = 0
            dy_empty_jwst = 0
            
        ##############################################################################################################################
        #Choose normalization

        norm_r = ImageNormalize(im_r_i, stretch=AsinhStretch())
        norm_s = ImageNormalize(im_s_i, stretch=AsinhStretch())
        norm_d = ImageNormalize(im_d_i, stretch=AsinhStretch())
        
        norm = norm_r
        
        ##############################################################################################################################
        #Plot cutouts with fluxes
        
        ax[0].imshow(im_r_i, norm=norm, cmap='Greys_r', origin='lower', interpolation='none')
        ax[1].imshow(im_s_i, norm=norm, cmap='Greys_r', origin='lower', interpolation='none')
        ax[2].imshow(im_d_i, norm=norm, cmap='Greys_r', origin='lower', interpolation='none')    
        
        if show_flux:
            ax[0].text(.01, .98, format_flux(m_r_og, merr_r_og, div_factor), color='orange', fontsize=12, va='top', ha='left', transform=ax[0].transAxes, fontweight='bold')
            ax[0].text(.01, .9, format_flux(m_rs_og, merr_rs_og, div_factor), color='r', fontsize=12, va='top', ha='left', transform=ax[0].transAxes, fontweight='bold')
            
            ax[1].text(.01, .98, format_flux(m_sr_og, merr_sr_og, div_factor), color='orange', fontsize=12, va='top', ha='left', transform=ax[1].transAxes, fontweight='bold')
            ax[1].text(.01, .9, format_flux(m_s_og, merr_s_og, div_factor), color='r', fontsize=12, va='top', ha='left', transform=ax[1].transAxes, fontweight='bold')
            
            ax[2].text(.01, .98, format_flux(m_r, merr_r, div_factor), color='orange', fontsize=12, va='top', ha='left', transform=ax[2].transAxes, fontweight='bold')
            ax[2].text(.01, .9, format_flux(m_s, merr_s, div_factor), color='r', fontsize=12, va='top', ha='left', transform=ax[2].transAxes, fontweight='bold')
            ax[2].text(.01, .82, format_flux(m_d, merr_d, div_factor), color='m', fontsize=12, va='top', ha='left', transform=ax[2].transAxes, fontweight='bold')
        
        ##############################################################################################################################
        #Cts to microJy
        
        if show_ct_conversion:
            ax[0].text(.98, .02, r'1 ct = {:.2e} $\mu$Jy'.format(ct2microjy_euclid), fontsize=10, color='y', fontweight='bold', va='bottom', ha='right', transform=ax[0].transAxes)
            ax[1].text(.98, .02, r'1 DN = {:.2e} $\mu$Jy'.format(dn2microjy_jwst), fontsize=10, color='y', fontweight='bold', va='bottom', ha='right', transform=ax[1].transAxes)

        ##############################################################################################################################
        #Make scalebar for 1" in diff image
        x2 = .98
        x1 = (x2*im_r_i.shape[1] - 1/euclid_ps) / im_r_i.shape[1]
        y = .85*im_r_i.shape[0] 
        ax[0].hlines(y=y, xmin=x1, xmax=x2, transform=ax[0].get_yaxis_transform(), color='y', lw=2)
        ax[0].text((x1+x2)/2, (y/im_r_i.shape[0] + .05)*im_r_i.shape[0], '1"', color='y', fontsize=11, va='center', ha='left', transform=ax[0].get_yaxis_transform(), fontweight='bold')
        
        ###############################################################################################################################
        #Draw arrow on all images
        
        xend = dxc_d_use + dx_px
        yend = dyc_d_use + dy_px
        
        xstart = xend + im_r_i.shape[1]/10.
        ystart = yend + im_r_i.shape[0]/10.
        
        for a in ax:
            a.annotate('', xy=(xend, yend), xytext=(xstart, ystart),
                        arrowprops=dict(arrowstyle='->', lw=2, color='m'))
            
        ###############################################################################################################################
        #Print index
        ax[0].text(.05, .95, str(n), fontsize=12, color='y', fontweight='bold', va='top', ha='left', transform=ax[0].transAxes)
        
        
        ##############################################################################################################################
        #Draw apertures        
        
        if show_aperture:
            dxvals = np.linspace(-dx_px, dx_px, 7500)[::2]
            cxx_s_in, cxy_s_in, cyy_s_in = catalog['CXX_SCI_INPUT'][n], catalog['CXY_SCI_INPUT'][n], catalog['CYY_SCI_INPUT'][n] 
            cxx_r_in, cxy_r_in, cyy_r_in = catalog['CXX_REF_INPUT'][n], catalog['CXY_REF_INPUT'][n], catalog['CYY_REF_INPUT'][n]
            cxx_s, cxy_s, cyy_s = catalog['CXX_SCI'][n], catalog['CXY_SCI'][n], catalog['CYY_SCI'][n]
            cxx_r, cxy_r, cyy_r = catalog['CXX_REF'][n], catalog['CXY_REF'][n], catalog['CYY_REF'][n]
            
            if diff_detected:
                if catalog['DIFF_DETECTED'][n]:
                    cxx_d, cxy_d, cyy_d = catalog['CXX_DIFF'][n], catalog['CXY_DIFF'][n], catalog['CYY_DIFF'][n]
                if catalog['DIFF_DETECTED_n'][n]:
                    cxx_d2, cxy_d2, cyy_d2 = catalog['CXX_DIFF_n'][n], catalog['CXY_DIFF_n'][n], catalog['CYY_DIFF_n'][n]
                    
                if np.isnan(dxc_d):
                    dxc_d_use = dxc_d2
                    dyc_d_use = dyc_d2
                    
                    cxx_d_use = cxx_d2
                    cxy_d_use = cxy_d2
                    cyy_d_use = cyy_d2 
                else:
                    dxc_d_use = dxc_d
                    dyc_d_use = dyc_d 
                    
                    cxx_d_use = cxx_d
                    cxy_d_use = cxy_d
                    cyy_d_use = cyy_d      
            
            
            dxvals_r = np.zeros_like(im_r_i)
            dyvals_r = np.zeros_like(im_r_i)
            dxvals_s = np.zeros_like(im_s_i)
            dyvals_s = np.zeros_like(im_s_i)
            for j in range(im_r_i.shape[0]):
                dyvals_r[j] = j - (dy_px + dyc_r)
                dxvals_s[j] = j - (dy_px + dyc_s)
            for j in range(im_r_i.shape[1]):
                dxvals_r[:,j] = j - (dx_px + dxc_r)
                dyvals_s[:,j] = j - (dy_px + dyc_s)

                
            if diff_detected:
                dxvals_d = np.zeros_like(im_d_i)
                dyvals_d = np.zeros_like(im_d_i)
                for j in range(im_r_i.shape[0]):
                    dyvals_d[j] = j - (dy_px + dyc_d_use)
                for j in range(im_r_i.shape[1]):
                    dxvals_d[:,j] = j - (dx_px + dxc_d_use)
                

            emask_r_in = get_ellipse_mask(dxvals_r, dyvals_r, cxx_r_in, cxy_r_in, cyy_r_in, r=rad_r_in)
            emask_r = get_ellipse_mask(dxvals_r, dyvals_r, cxx_r, cxy_r, cyy_r, r=rad_r)
            # emask_r[np.isnan(im_r_i)] = False
            # emask_r[np.isnan(im_r_i)] = False
            # emask_r[im_r_i == 0.] = False
            # emask_r[im_r_i == 0.] = False
            

            ex_s_in, ey_s_in = get_ellipse(dxvals, cxx_s_in, cxy_s_in, cyy_s_in, r=rad_s_in)
            ex_r_in, ey_r_in = get_ellipse(dxvals, cxx_r_in, cxy_r_in, cyy_r_in, r=rad_r_in)
            ex_s, ey_s = get_ellipse(dxvals, cxx_s, cxy_s, cyy_s, r=rad_s)
            ex_r, ey_r = get_ellipse(dxvals, cxx_r, cxy_r, cyy_r, r=rad_r)
            if diff_detected:
                ex_d, ey_d = get_ellipse(dxvals, cxx_d_use, cxy_d_use, cyy_d_use, r=rad_d)
                
            xi_r = ex_r + (dx_px + dxc_r)
            yi_r = ey_r + (dy_px + dyc_r)

            xi_s = ex_s + (dx_px + dxc_s)
            yi_s = ey_s + (dy_px + dyc_s)
            
            xi_r_in = ex_r_in + (dx_px + dxc_r)
            yi_r_in = ey_r_in + (dy_px + dyc_r)
            
            xi_s_in = ex_s_in + (dx_px + dxc_s)
            yi_s_in = ey_s_in + (dy_px + dyc_s)
            
            if diff_detected:
                xi_d = ex_d + (dx_px + dxc_d_use)
                yi_d = ey_d + (dy_px + dyc_d_use)
            
            
            for a in [ax[0], ax[1]]:        
                a.plot(xi_r_in, yi_r_in, color='orange', lw=1)
                a.plot(xi_s_in, yi_s_in, color='r', lw=1)
                if diff_detected:
                    a.plot(xi_d, yi_d, color='m', lw=1)
                

            ax[2].plot(xi_r, yi_r, color='orange', lw=2)
            ax[2].plot(xi_s, yi_s, color='r', lw=2)
            if diff_detected:
                ax[2].plot(xi_d, yi_d, color='m', lw=2)
        
        ##############################################################################################################################
        #Draw checks and xs

        if ref_detected:
            ax[0].text(.05, .05, '✓', color='orange', fontsize=24, va='bottom', ha='left', transform=ax[0].transAxes, fontweight='bold')
        else:
            ax[0].text(.05, .05, '×', color='orange', fontsize=24, va='bottom', ha='left', transform=ax[0].transAxes, fontweight='bold')

            
        if sci_detected:
            ax[1].text(.05, .05, '✓', color='r', fontsize=24, va='bottom', ha='left', transform=ax[1].transAxes, fontweight='bold')
        else:
            ax[1].text(.05, .05, '×', color='r', fontsize=24, va='bottom', ha='left', transform=ax[1].transAxes, fontweight='bold')
            
            
        # if diff_detected:
        #     ax[2].text(.05, .05, '✓', color='m', fontsize=24, va='bottom', ha='left', transform=ax[2].transAxes, fontweight='bold')
        # else:
        #     ax[2].text(.05, .05, '×', color='m', fontsize=24, va='bottom', ha='left', transform=ax[2].transAxes, fontweight='bold')
        
        ##############################################################################################################################
        #Set axis limits and ticks
        
        for j in range(3):
            ax[j].set_xticks([])
            ax[j].set_yticks([])

            ax[j].set_xlim(0, dx_px*2)
            ax[j].set_ylim(0, dy_px*2)        
        
        ##############################################################################################################################
        #Set axis labels and titles
            
        if k == 0:
            ax[0].set_title('REF (Euclid)', fontsize=11, color='orange', fontweight='bold')
            ax[1].set_title('SCI (JWST)\nResampled', fontsize=11, color='red', fontweight='bold')
            ax[2].set_title('DIFF', fontsize=11, color='m', fontweight='bold')
    
    if output_fname is not None:
        plt.savefig(output_fname, dpi=300, bbox_inches='tight')
        
    if show:
        plt.show()
    
    plt.cla()
    plt.clf()
    plt.close()
    
    return


def make_source_cutouts_simple_pubcat(catalog, catalog_euc, catalog_nex, maindir_source, fname_jwst, inds, 
                                      dx_cutout=3, dy_cutout=3, unit='Jy',
                                      show_flux=False, show_ct_conversion=False,
                                      output_fname=None, show=False):
    
    cmap = mpl.cm.Greys_r
    cmap.set_bad(color='c', alpha=.25)
    
    if unit == 'Jy':
        div_factor = 1e-6
        zp = 8.9
    else:
        div_factor = 1
        zp = 23.9
        
    FLUB_FACTOR = 1e40
    
    if np.max(inds) >= len(catalog):
        inds_new = []
        for i in inds:
            if i < len(catalog):
                inds_new.append(i)
        inds = inds_new    
    
    
    with fits.open(fname_jwst) as hdu:
        im_jwst = hdu[0].data
        hdr_jwst = hdu[0].header
        wcs_jwst = WCS(hdr_jwst)    
        
    im_jwst *= 10**( (zp - hdr_jwst['MAG_ZP']) / 2.5 )
    # im_jwst *= FLUB_FACTOR
    
    dx = dx_cutout*u.arcsec
    dy = dy_cutout*u.arcsec

    ct2microjy_jwst = 0.021153987485188146
    ct2microjy_euclid = 0.0036307805477010027

    dn2microjy_jwst = 1.0550972810012904e-05
    e2microjy_euclid = 0.0036307805477010027

    gain_euclid = 2 #From: https://www.aanda.org/articles/aa/full_html/2025/05/aa50803-24/aa50803-24.html
    ct2microjy_euclid = e2microjy_euclid / gain_euclid



    jwst_scale = 10/3
    jwst_ps = .03
    euclid_ps = 0.1

    Ncol_gs = 2
    Ncol = 4

    Nrow = min(Ncol*Ncol_gs, len(inds))
    Nrow_gs = Nrow // Ncol_gs + 1
    
    fig = plt.figure(figsize=(3*Ncol*Ncol_gs, 3*Nrow_gs))
    gs = GridSpec(Nrow_gs, Ncol_gs, wspace=.05, hspace=0.)


    for i, n in enumerate(inds):
        k = i // Ncol_gs
        l = i % Ncol_gs
        
        gs_i = GridSpecFromSubplotSpec(1, Ncol, subplot_spec=gs[k, l], hspace=0., wspace=0.)
        
        cutout_num = catalog['CUTOUT'][n]
        
        with fits.open(maindir_source + 'sources_{}/REF.fits'.format(cutout_num)) as hdul:
            im_in_r = hdul[0].data
            hdr_r = hdul[0].header

        with fits.open(maindir_source + 'sources_{}/SCI.fits'.format(cutout_num)) as hdul:
            im_in_s = hdul[0].data
            hdr_s = hdul[0].header

        with fits.open(maindir_source + 'sources_{}/DIFF.fits'.format(cutout_num)) as hdul:
            im_out = hdul[0].data
            hdr_d = hdul[0].header

        with fits.open(maindir_source + 'sources_{}/DIFF.neg.fits'.format(cutout_num)) as hdul:
            im_out_n = hdul[0].data        
            hdr_d2 = hdul[0].header

        wcs_in = WCS(hdr_r)

        ra_s = catalog['RA_NEXUS'][n]
        dec_s = catalog['DEC_NEXUS'][n]
        xs, ys = wcs_in.wcs_world2pix(ra_s, dec_s, 0)
        
        ra_r = catalog['RA_EUCLID'][n]
        dec_r = catalog['DEC_EUCLID'][n]
        xr, yr = wcs_in.wcs_world2pix(ra_r, dec_r, 0)
        
        ra_d = catalog['RA_DIFF'][n]
        dec_d = catalog['DEC_DIFF'][n]
        xd, yd = wcs_in.wcs_world2pix(ra_d, dec_d, 0)
        
        ra_d2 = catalog['RA_DIFF_n'][n]
        dec_d2 = catalog['DEC_DIFF_n'][n]
        xd2, yd2 = wcs_in.wcs_world2pix(ra_d2, dec_d2, 0)
        
        fn = catalog['FLUX_NEXUS'][n]
        fe = catalog['FLUX_EUCLID'][n]
        
        ref_detected = True
        if np.isnan(catalog[n]['FLUX_EUCLID']) or isinstance(catalog[n]['FLUX_EUCLID'], np.ma.core.MaskedConstant) or (fe < 0.):
            ref_detected = False
        
        sci_detected = True
        if np.isnan(catalog[n]['FLUX_NEXUS']) or isinstance(catalog[n]['FLUX_NEXUS'], np.ma.core.MaskedConstant) or (fn < 0.):
            sci_detected = False

        diff_detected = True
        if (not catalog[n]['DIFF_DETECTED']) and (not catalog[n]['DIFF_DETECTED_n']):
            diff_detected = False

        x = xd
        y = yd
        ra = ra_d
        dec = dec_d
        im_use = 'diff'
        if np.isnan(ra) or isinstance(ra, np.ma.core.MaskedConstant):
            x = xd2
            y = yd2
            ra = ra_d2
            dec = dec_d2
            im_use = 'diff_n'
        if np.isnan(ra) or isinstance(ra, np.ma.core.MaskedConstant):
            x = xr
            y = yr
            ra = ra_r
            dec = dec_r
            im_use = 'ref'
        if np.isnan(ra) or isinstance(ra, np.ma.core.MaskedConstant):
            x = xs
            y = ys
            ra = ra_s
            dec = dec_s
            im_use = 'sci'
            
        ra_obj = ra
        dec_obj = dec
        
        ##############################################################################################################################
        #If close to edge, correct center

        close_top = False
        close_bottom = False
        close_left = False
        close_right = False
        if x + dx.value/euclid_ps > im_in_r.shape[1]:
            close_right = True
            print(f'{i} Close to right edge')
        if x - dx.value/euclid_ps < 0:
            close_left = True
            print(f'{i} Close to left edge')
        if y + dy.value/euclid_ps > im_in_r.shape[0]:
            close_top = True
            print(f'{i} Close to top edge')
        if y - dy.value/euclid_ps < 0:
            close_bottom = True
            print(f'{i} Close to bottom edge')
            
            
        rac, decc = wcs_in.wcs_pix2world(x, y, 0)
            
        if close_right:
            x -= np.abs(x - im_in_r.shape[1] + dx.value/euclid_ps)
            rac, decc = wcs_in.wcs_pix2world(x, y, 0)
        if close_left:
            x += np.abs(x - dx.value/euclid_ps)
            rac, decc = wcs_in.wcs_pix2world(x, y, 0)
        if close_top:
            y -= np.abs(y - im_in_r.shape[0] + dy.value/euclid_ps)
            rac, decc = wcs_in.wcs_pix2world(x, y, 0)
        if close_bottom:
            y += np.abs(y - dy.value/euclid_ps)
            rac, decc = wcs_in.wcs_pix2world(x, y, 0)
            
        ###############################################################################################################################
        # Get all sources in image
        
        ra_min = rac - dx.value/3600.
        ra_max = rac + dx.value/3600.
        dec_min = decc - dy.value/3600.
        dec_max = decc + dy.value/3600.
        
        coords_euc = SkyCoord(catalog_euc['RIGHT_ASCENSION'].data, catalog_euc['DECLINATION'].data, unit=(u.deg, u.deg))
        coords_nex = SkyCoord(catalog_nex['RA'].data, catalog_nex['DEC'].data, unit=(u.deg, u.deg))
        coord_center = SkyCoord([rac], [decc], unit=(u.deg, u.deg))
        
        idx_euc, d2d_euc, d3d_euc = coords_euc.match_to_catalog_sky(coord_center)
        idx_nex, d2d_nex, d3d_nex = coords_nex.match_to_catalog_sky(coord_center)
        
        mask_euc = d2d_euc.arcsec < np.sqrt((2*dx.value)**2 + (2*dy.value)**2)
        mask_nex = d2d_nex.arcsec < np.sqrt((2*dx.value)**2 + (2*dy.value)**2)
        
        # mask_euc = ((catalog_euc['RIGHT_ASCENSION'] > ra_min) & (catalog_euc['RIGHT_ASCENSION'] < ra_max) & \
        #             (catalog_euc['DECLINATION'] > dec_min) & (catalog_euc['DECLINATION'] < dec_max))
        # mask_nex = ((catalog_nex['RA'] > ra_min) & (catalog_nex['RA'] < ra_max) & \
        #             (catalog_nex['DEC'] > dec_min) & (catalog_nex['DEC'] < dec_max))
        
        cat_euc_i = catalog_euc[mask_euc].copy()
        cat_nex_i = catalog_nex[mask_nex].copy()     
        
        # print('Euclid sources in cutout:', len(cat_euc_i))
        # print('Nexus sources in cutout:', len(cat_nex_i))        
                        
        ##############################################################################################################################
        #Get fluxes/errors    
        
        if show_flux:
            if ref_detected:
                f_r = catalog['FLUX_EUCLID'][n]
                ferr_r = catalog['FLUXERR_EUCLID'][n]
                m_r, merr_r = flux_to_mag_w_err(f_r, ferr_r, zp=23.9)

            if sci_detected:
                f_s = catalog['FLUX_NEXUS'][n]
                ferr_s = catalog['FLUXERR_NEXUS'][n]
                m_s, merr_s = flux_to_mag_w_err(f_s, ferr_s, zp=23.9)

            if diff_detected:
                f_d1 = catalog['FLUX_DIFF'][n]/div_factor
                f_d2 = catalog['FLUX_DIFF_n'][n]/div_factor
                ferr_d1 = catalog['FLUXERR_DIFF'][n]/div_factor
                ferr_d2 = catalog['FLUXERR_DIFF_n'][n]/div_factor
                if np.isnan(f_d1) or isinstance(f_d1, np.ma.core.MaskedConstant) or (f_d1 < 0.):
                    f_d = -f_d2
                    ferr_d = ferr_d2
                else:
                    f_d = f_d1
                    ferr_d = ferr_d1
                    
                m_d, merr_d = flux_to_mag_w_err(f_d, ferr_d, zp=zp)
        
        ##############################################################################################################################
        #Get source cutouts
        
        coord = SkyCoord(rac, decc, unit=(u.deg, u.deg))
        im_r_i = Cutout2D(im_in_r, coord, size=(dx*2, dy*2), wcs=wcs_in)
        im_s_i = Cutout2D(im_in_s, coord, size=(dx*2, dy*2), wcs=wcs_in)
        im_d_i = Cutout2D(im_out, coord, size=(dx*2, dy*2), wcs=wcs_in)

        ###############################################################################################################################
        #Get JWST cutout

        dx_px_jwst = int(dx.value / jwst_ps)
        dy_px_jwst = int(dy.value / jwst_ps)

        im_jwst_i = Cutout2D(im_jwst, coord, size=(dx*2, dy*2), wcs=wcs_jwst)
        
        ##############################################################################################################################
        #Get center of cutout

        if im_use == 'ref':
            im_use = im_r_i
            wcs_use = im_use.wcs
        elif im_use == 'sci':
            im_use = im_s_i
            wcs_use = im_use.wcs
        elif im_use == 'diff':
            im_use = im_d_i
            wcs_use = im_use.wcs
        elif im_use == 'diff_n':
            im_use = im_d_i
            wcs_use = im_use.wcs
        
        xs_c, ys_c = im_use.wcs.wcs_world2pix(ra_s, dec_s, 0)
        xr_c, yr_c = im_use.wcs.wcs_world2pix(ra_r, dec_r, 0)
        xd_c, yd_c = im_use.wcs.wcs_world2pix(ra_d, dec_d, 0)
        xd2_c, yd2_c = im_use.wcs.wcs_world2pix(ra_d2, dec_d2, 0)  
        
        xs_c_jwst, ys_c_jwst = im_jwst_i.wcs.wcs_world2pix(ra_s, dec_s, 0)      
        xr_c_jwst, yr_c_jwst = im_jwst_i.wcs.wcs_world2pix(ra_r, dec_r, 0)
        xd_c_jwst, yd_c_jwst = im_jwst_i.wcs.wcs_world2pix(ra_d, dec_d, 0)
        xd2_c_jwst, yd2_c_jwst = im_jwst_i.wcs.wcs_world2pix(ra_d2, dec_d2, 0)
        
        xcat_euc, ycat_euc = im_use.wcs.wcs_world2pix(cat_euc_i['RIGHT_ASCENSION'].data, cat_euc_i['DECLINATION'].data, 0)
        xcat_nex, ycat_nex = im_use.wcs.wcs_world2pix(cat_nex_i['RA'].data, cat_nex_i['DEC'].data, 0)
        xcat_euc_jwst, ycat_euc_jwst = im_jwst_i.wcs.wcs_world2pix(cat_euc_i['RIGHT_ASCENSION'].data, cat_euc_i['DECLINATION'].data, 0)
        xcat_nex_jwst, ycat_nex_jwst = im_jwst_i.wcs.wcs_world2pix(cat_nex_i['RA'].data, cat_nex_i['DEC'].data, 0)

        
        
        dx_px = int(dx.value / euclid_ps)
        dy_px = int(dy.value / euclid_ps)
        dx_px_jwst = int(dx.value / jwst_ps)
        dy_px_jwst = int(dy.value / jwst_ps)
        
        dxc_r = xr_c - dx_px
        dyc_r = yr_c - dy_px
        dxc_s = xs_c - dx_px
        dyc_s = ys_c - dy_px
        dxc_d = xd_c - dx_px
        dyc_d = yd_c - dy_px
        dxc_d2 = xd2_c - dx_px
        dyc_d2 = yd2_c - dy_px   
        
        dxc_r_jwst = xr_c_jwst - dx_px_jwst
        dyc_r_jwst = yr_c_jwst - dy_px_jwst
        dxc_s_jwst = xs_c_jwst - dx_px_jwst
        dyc_s_jwst = ys_c_jwst - dy_px_jwst
        dxc_d_jwst = xd_c_jwst - dx_px_jwst
        dyc_d_jwst = yd_c_jwst - dy_px_jwst
        dxc_d2_jwst = xd2_c_jwst - dx_px_jwst
        dyc_d2_jwst = yd2_c_jwst - dy_px_jwst   
        
        if np.isnan(dxc_d):
            dxc_d_use = dxc_d2
            dyc_d_use = dyc_d2
            dxc_d_jwst_use = dxc_d2_jwst
            dyc_d_jwst_use = dyc_d2_jwst
        else:
            dxc_d_use = dxc_d
            dyc_d_use = dyc_d   
            dxc_d_jwst_use = dxc_d_jwst
            dyc_d_jwst_use = dyc_d_jwst
            
        ##############################################################################################################################
        #Choose normalization

        norm_r = ImageNormalize(im_r_i.data, stretch=AsinhStretch())
        norm_s = ImageNormalize(im_s_i.data, stretch=AsinhStretch())
        norm_d = ImageNormalize(im_d_i.data, stretch=AsinhStretch())        
        norm_jwst = ImageNormalize(im_jwst_i.data, stretch=AsinhStretch())
        
        ##############################################################################################################################
        #Plot cutouts with fluxes
        ax = np.zeros(Ncol, dtype=object) 
        for j in range(Ncol):
            if j != 2:
                ax[j] = plt.subplot(gs_i[0, j], projection=wcs_use)
            else:
                ax[j] = plt.subplot(gs_i[0, j], projection=im_jwst_i.wcs)

        ax[0].imshow(im_r_i.data, cmap=cmap, origin='lower', interpolation='auto', aspect='equal', norm=norm_r)
        ax[1].imshow(im_s_i.data, cmap=cmap, origin='lower', interpolation='auto', aspect='equal', norm=norm_r)
        ax[2].imshow(im_jwst_i.data, cmap=cmap, origin='lower', interpolation='auto', aspect='equal', norm=norm_jwst)
        ax[3].imshow(im_d_i.data, cmap=cmap, origin='lower', interpolation='auto', aspect='equal', norm=norm_r)  

        if show_flux:
            if ref_detected:
                ax[0].text(.01, .98, format_flux(m_r, merr_r), color='orange', fontsize=12, va='top', ha='left', transform=ax[0].transAxes, fontweight='bold')

            if sci_detected:
                ax[1].text(.01, .98, format_flux(m_s, merr_s), color='r', fontsize=12, va='top', ha='left', transform=ax[1].transAxes, fontweight='bold')
            
            # ax[3].text(.01, .98, format_flux(m_r, merr_r, div_factor), color='orange', fontsize=12, va='top', ha='left', transform=ax[3].transAxes, fontweight='bold')
            # ax[3].text(.01, .9, format_flux(m_s, merr_s, div_factor), color='r', fontsize=12, va='top', ha='left', transform=ax[3].transAxes, fontweight='bold')
            # ax[3].text(.01, .82, format_flux(m_d, merr_d, div_factor), color='m', fontsize=12, va='top', ha='left', transform=ax[3].transAxes, fontweight='bold')
        
        ##############################################################################################################################
        # Draw locations of catalog/DIFF sources
        
        if len(cat_euc_i) > 0:
            for j in range(Ncol):
                ax[j].errorbar(cat_euc_i['RIGHT_ASCENSION'].data, cat_euc_i['DECLINATION'].data, ms=7.5, color='orange', fmt='x', transform=ax[j].get_transform('world'))
        if len(cat_nex_i) > 0:
            for j in range(Ncol):
                ax[j].errorbar(cat_nex_i['RA'].data, cat_nex_i['DEC'].data, ms=7.5, color='r', fmt='x', transform=ax[j].get_transform('world'))

        if diff_detected:
            for j in range(Ncol):
                ax[j].errorbar([ra_obj], [dec_obj], ms=7.5, color='m', fmt='x', transform=ax[j].get_transform('world'))


        ##############################################################################################################################
        #Cts to microJy
        
        if show_ct_conversion:
            ax[0].text(.98, .02, r'1 ct = {:.2e} $\mu$Jy'.format(ct2microjy_euclid), fontsize=10, color='y', fontweight='bold', va='bottom', ha='right', transform=ax[0].transAxes)
            ax[1].text(.98, .02, r'1 DN = {:.2e} $\mu$Jy'.format(dn2microjy_jwst), fontsize=10, color='y', fontweight='bold', va='bottom', ha='right', transform=ax[1].transAxes)

        ##############################################################################################################################
        #Make scalebar for 1" in JWST image
        
        add_scalebar(ax[2], 1 * u.arcsec, label='1"', color='y', fontproperties={'size': 11, 'weight': 'bold'}, size_vertical=1)
        
        # x2 = .95
        # x1 = (x2*im_jwst_i.shape[1] - 1/jwst_ps) / im_jwst_i.shape[1]
        # y = .85*im_jwst_i.shape[0] 
        # ax[2].hlines(y=y, xmin=x1, xmax=x2, transform=ax[2].get_yaxis_transform(), color='y', lw=2)
        # ax[2].text((x1+x2)/2, (y/im_jwst_i.shape[0] + .05)*im_jwst_i.shape[0], '1"', color='y', fontsize=11, va='center', ha='left', transform=ax[2].get_yaxis_transform(), fontweight='bold')
        
        ###############################################################################################################################
        #Draw arrow on all images
        
        # xend = dxc_d_use + dx_px
        # yend = dyc_d_use + dy_px
        # xstart = xend + im_r_i.data.shape[1]/7.
        # ystart = yend + im_r_i.data.shape[0]/7.
        
        # for a in ax[[0,1,3]]:
        #     a.annotate('', xy=(xend, yend), xytext=(xstart, ystart),
        #                 arrowprops=dict(arrowstyle='->', lw=3, color='m', mutation_scale=20, alpha=.5))
            
        
        # xend_jwst = dxc_d_jwst_use + dx_px_jwst
        # yend_jwst = dyc_d_jwst_use + dy_px_jwst
        # xstart_jwst = xend_jwst + im_jwst_i.data.shape[1]/7.
        # ystart_jwst = yend_jwst + im_jwst_i.data.shape[0]/7.
        # ax[2].annotate('', xy=(xend_jwst, yend_jwst), xytext=(xstart_jwst, ystart_jwst),
        #                 arrowprops=dict(arrowstyle='->', lw=3, color='m', mutation_scale=20, alpha=.5))
            
        ###############################################################################################################################
        #Print index

        ax[0].text(.05, .05, str(n), fontsize=20, color='y', fontweight='bold', va='bottom', ha='left', transform=ax[0].transAxes)
        
        ##############################################################################################################################
        #Draw checks and xs

        # if ref_detected:
        #     ax[0].text(.95, .05, '✓', color='orange', fontsize=24, va='bottom', ha='right', transform=ax[0].transAxes, fontweight='bold')
        # else:
        #     ax[0].text(.95, .05, '×', color='orange', fontsize=24, va='bottom', ha='right', transform=ax[0].transAxes, fontweight='bold')

            
        # if sci_detected:
        #     ax[1].text(.95, .05, '✓', color='r', fontsize=24, va='bottom', ha='right', transform=ax[1].transAxes, fontweight='bold')
        # else:
        #     ax[1].text(.95, .05, '×', color='r', fontsize=24, va='bottom', ha='right', transform=ax[1].transAxes, fontweight='bold')
            
            
        # if diff_detected:
        #     ax[2].text(.05, .05, '✓', color='m', fontsize=24, va='bottom', ha='left', transform=ax[2].transAxes, fontweight='bold')
        # else:
        #     ax[2].text(.05, .05, '×', color='m', fontsize=24, va='bottom', ha='left', transform=ax[2].transAxes, fontweight='bold')
        
        ##############################################################################################################################
        #Set axis limits and ticks
        
        for j in range(Ncol):
            ra = ax[j].coords['ra']
            dec = ax[j].coords['dec']
            
            ra.set_axislabel('')
            dec.set_axislabel('')
            ra.set_ticklabel_visible(False)
            dec.set_ticklabel_visible(False)
            ra.set_ticks(np.array([])*u.deg)
            dec.set_ticks(np.array([])*u.deg)
            

        for j in [0,1,3]:
            ax[j].set_xlim(0, im_r_i.data.shape[1])
            ax[j].set_ylim(0, im_r_i.data.shape[0]) 
            
        ax[2].set_xlim(0, im_jwst_i.data.shape[1])
        ax[2].set_ylim(0, im_jwst_i.data.shape[0])
        
        ##############################################################################################################################
        #Set axis labels and titles
            
        if k == 0:
            ax[0].set_title('REF (Euclid)', fontsize=16, color='orange', fontweight='bold')
            ax[1].set_title('SCI (JWST)\nResampled', fontsize=16, color='red', fontweight='bold')
            ax[2].set_title('SCI (JWST)\nNative', fontsize=16, color='m', fontweight='bold')
            ax[3].set_title('DIFF', fontsize=16, color='m', fontweight='bold')
            
        del im_in_r, im_in_s, im_out, im_out_n, wcs_in
        del im_r_i, im_s_i, im_d_i, im_jwst_i, im_use
        del xs, ys, xr, yr, xd, yd, xd2, yd2
        del ra_s, dec_s, ra_r, dec_r, ra_d, dec_d, ra_d2, dec_d2
        del x, y, ra, dec, ra_obj, dec_obj, rac, decc
        del norm_r, norm_s, norm_d, norm_jwst
        
        
        
    
    if output_fname is not None:
        plt.savefig(output_fname)#, dpi=700, bbox_inches='tight')
        
    if show:
        plt.show()
    
    plt.cla()
    plt.clf()
    plt.close()
    
    return




def make_source_cutouts(catalog, maindir_source, maindir_diff, fname_jwst, inds, 
                        dx_cutout=3, dy_cutout=3, unit='Jy',
                        dx_cutout_big=6, dy_cutout_big=6,
                        output_fname=None, show=False):
    
    im_in_s_native = fits.open(fname_jwst)[0].data
    hdr_jwst = fits.open(fname_jwst)[0].header

    #Need to change ZP of JWST
    if unit == 'Jy':
        im_in_s_native *= 10**( (8.9 - hdr_jwst['MAG_ZP']) / 2.5 )
    elif unit == 'micro-Jy':
        im_in_s_native *= 10**( (8.9 - hdr_jwst['MAG_ZP']) / 2.5 ) / 1e-6
    
    
    
    dx = dx_cutout*u.arcsec
    dy = dy_cutout*u.arcsec

    dx_big = dx_cutout_big*u.arcsec
    dy_big = dy_cutout_big*u.arcsec


    ct2microjy_jwst = 0.021153987485188146
    ct2microjy_euclid = 0.0036307805477010027

    dn2microjy_jwst = 1.0550972810012904e-05
    e2microjy_euclid = 0.0036307805477010027

    gain_euclid = 2 #From: https://www.aanda.org/articles/aa/full_html/2025/05/aa50803-24/aa50803-24.html
    ct2microjy_euclid = e2microjy_euclid / gain_euclid

    if unit == 'Jy':
        div_factor = 1e-6
    else:
        div_factor = 1



    jwst_scale = 10/3
    jwst_ps = .03

    Nrow = min(5, len(inds))   
    Ncol = 5

    fig, ax = plt.subplots(Nrow, Ncol, figsize=(3*Ncol, 3*Nrow))

    tmpdir = 'tmp/'
    os.makedirs(tmpdir, exist_ok=True)

    for i, n in enumerate(inds):
        cutout_num = catalog['CUTOUT'][n]
        im_in_r = fits.open(maindir_source + 'sources_{}/REF.fits'.format(cutout_num))[0].data
        im_in_s = fits.open(maindir_source + 'sources_{}/SCI.fits'.format(cutout_num))[0].data
        im_out = fits.open(maindir_source + 'sources_{}/DIFF.fits'.format(cutout_num))[0].data
        im_out_n = fits.open(maindir_source + 'sources_{}/DIFF.neg.fits'.format(cutout_num))[0].data
        
        im_decorr_bkg = fits.open(maindir_diff + 'output_{}/output/nexus_F115W.decorrbkg.fits'.format(cutout_num))[0].data[30:-30, 30:-30]
        im_decorr_bkg = im_decorr_bkg[ (im_in_r != 0.) & (im_in_s != 0.) ]
        
        wcs_in = WCS(fits.open(maindir_source + 'sources_{}/REF.fits'.format(cutout_num))[0].header)
        
        
        
        ra_s = catalog['RA_SCI_INPUT'][n]
        dec_s = catalog['DEC_SCI_INPUT'][n]
        xs, ys = wcs_in.wcs_world2pix(ra_s, dec_s, 0)
        
        ra_r = catalog['RA_REF_INPUT'][n]
        dec_r = catalog['DEC_REF_INPUT'][n]
        xr, yr = wcs_in.wcs_world2pix(ra_r, dec_r, 0)
        
        ra_d = catalog['RA_DIFF'][n]
        dec_d = catalog['DEC_DIFF'][n]
        xd, yd = wcs_in.wcs_world2pix(ra_d, dec_d, 0)
        
        ra_d2 = catalog['RA_DIFF_n'][n]
        dec_d2 = catalog['DEC_DIFF_n'][n]
        xd2, yd2 = wcs_in.wcs_world2pix(ra_d2, dec_d2, 0)
        
        ref_detected = True
        if np.isnan(catalog[n]['FLUX_AUTO_REF_INPUT']) or isinstance(catalog[n]['FLUX_AUTO_REF_INPUT'], np.ma.core.MaskedConstant):
            ref_detected = False
        
        sci_detected = True
        if np.isnan(catalog[n]['FLUX_AUTO_SCI_INPUT']) or isinstance(catalog[n]['FLUX_AUTO_SCI_INPUT'], np.ma.core.MaskedConstant):
            sci_detected = False

        diff_detected = True
        if (not catalog[n]['DIFF_DETECTED']) and (not catalog[n]['DIFF_DETECTED_n']):
            diff_detected = False

        x = xs
        y = ys
        ra = ra_s
        dec = dec_s
        if np.isnan(ra) or isinstance(ra, np.ma.core.MaskedConstant):
            x = xr
            y = yr
            ra = ra_r
            dec = dec_r
        if np.isnan(ra) or isinstance(ra, np.ma.core.MaskedConstant):
            x = xd
            y = yd
            ra = ra_d
            dec = dec_d
        if np.isnan(ra) or isinstance(ra, np.ma.core.MaskedConstant):
            x = xd2
            y = yd2
            ra = ra_d2
            dec = dec_d2
            
                
        ##############################################################################################################################
        #Get Kron radius

        rad_r = catalog['KRON_RADIUS_REF'][n]
        rad_r2 = catalog['KRON_RADIUS_REF_n'][n]
        rad_s = catalog['KRON_RADIUS_SCI'][n]
        rad_s2 = catalog['KRON_RADIUS_SCI_n'][n]
        rad_d = catalog['KRON_RADIUS_DIFF'][n]
        rad_d2 = catalog['KRON_RADIUS_DIFF_n'][n]
        rad_r_in = catalog['KRON_RADIUS_REF_INPUT'][n]
        rad_s_in = catalog['KRON_RADIUS_SCI_INPUT'][n]
        
        if np.isnan(rad_r) or (rad_r == 0.) or isinstance(rad_r, np.ma.core.MaskedConstant):
            rad_r = rad_r2
        if np.isnan(rad_s) or (rad_s == 0.) or isinstance(rad_s, np.ma.core.MaskedConstant):
            rad_s = rad_s2
        if np.isnan(rad_d) or (rad_d == 0.) or isinstance(rad_d, np.ma.core.MaskedConstant):
            rad_d = rad_d2
            
        ##############################################################################################################################
        #Get fluxes/errors    
        
        m_r1 = catalog['FLUX_AUTO_REF'][n]/div_factor
        m_r2 = catalog['FLUX_AUTO_REF_n'][n]/div_factor
        merr_r1 = catalog['FLUXERR_AUTO_REF'][n]/div_factor
        merr_r2 = catalog['FLUXERR_AUTO_REF_n'][n]/div_factor
        m_r = m_r1
        merr_r = merr_r1
        # m_rvals = np.array([m_r1, m_r2])
        # m_rvals[m_rvals == 99.] = np.nan
        # m_r = np.nanmedian(m_rvals)


        m_s1 = catalog['FLUX_AUTO_SCI'][n]/div_factor
        m_s2 = catalog['FLUX_AUTO_SCI_n'][n]/div_factor
        merr_s1 = catalog['FLUXERR_AUTO_SCI'][n]/div_factor
        merr_s2 = catalog['FLUXERR_AUTO_SCI_n'][n]/div_factor
        m_s = m_s1
        merr_s = merr_s1

        m_d1 = catalog['FLUX_AUTO_DIFF'][n]/div_factor
        m_d2 = catalog['FLUX_AUTO_DIFF_n'][n]/div_factor
        merr_d1 = catalog['FLUXERR_AUTO_DIFF'][n]/div_factor
        merr_d2 = catalog['FLUXERR_AUTO_DIFF_n'][n]/div_factor
        if np.isnan(m_d1) or isinstance(m_d1, np.ma.core.MaskedConstant):
            m_d = -m_d2
            merr_d = merr_d2
        else:
            m_d = m_d1
            merr_d = merr_d1

        m_r_og = catalog['FLUX_AUTO_REF_INPUT'][n]/div_factor
        merr_r_og = catalog['FLUXERR_AUTO_REF_INPUT'][n]/div_factor
        m_rs_og = catalog['FLUX_AUTO_REF_SCIAPER'][n]/div_factor
        merr_rs_og = catalog['FLUXERR_AUTO_REF_SCIAPER'][n]/div_factor
        m_s_og = catalog['FLUX_AUTO_SCI_INPUT'][n]/div_factor
        merr_s_og = catalog['FLUXERR_AUTO_SCI_INPUT'][n]/div_factor
        m_sr_og = catalog['FLUX_AUTO_SCI_REFAPER'][n]/div_factor
        merr_sr_og = catalog['FLUXERR_AUTO_SCI_REFAPER'][n]/div_factor
        
        ##############################################################################################################################
        #Get source cutouts
        
        coord = SkyCoord(ra, dec, unit=(u.deg, u.deg))
        im_r_i = Cutout2D(im_in_r, coord, size=(dx*2, dy*2), wcs=wcs_in)
        im_s_i = Cutout2D(im_in_s, coord, size=(dx*2, dy*2), wcs=wcs_in)
        im_d_i = Cutout2D(im_out, coord, size=(dx*2, dy*2), wcs=wcs_in)
        
        ##############################################################################################################################
        #Get local background
        
        im_r_i_big = Cutout2D(im_in_r, coord, size=(dx_big*2, dy_big*2), wcs=wcs_in)
        im_s_i_big = Cutout2D(im_in_s, coord, size=(dx_big*2, dy_big*2), wcs=wcs_in)
        im_d_i_big = Cutout2D(im_out, coord, size=(dx_big*2, dy_big*2), wcs=wcs_in)
        
        fname_tmp_r = tmpdir + 'cutout_big_REF.fits'
        fits.writeto(fname_tmp_r, im_r_i_big.data, header=im_r_i_big.wcs.to_header(), overwrite=True)
        fname_tmp_s = tmpdir + 'cutout_big_SCI.fits'
        fits.writeto(fname_tmp_s, im_s_i_big.data, header=im_s_i_big.wcs.to_header(), overwrite=True)
        

        boundary_mask = np.ones(im_r_i_big.shape).astype(bool)
        boundary_mask[30:-30, 30:-30] = False
        
        bkg_mask_lsci = get_background(fname_tmp_s).T    
        bkg_mask_lref = get_background(fname_tmp_r).T  
        bkg_mask = np.logical_and.reduce((bkg_mask_lsci, bkg_mask_lref, ~boundary_mask))
        
        im_decorr_bkg_local = im_d_i_big.data[bkg_mask & (im_r_i_big.data != 0.) & (im_s_i_big.data != 0.)].copy()
        os.remove(fname_tmp_r)
        os.remove(fname_tmp_s)
        
        ##############################################################################################################################
        #Get center of cutout
        
        if ra == ra_r:
            im_use = im_r_i
        elif ra == ra_s:
            im_use = im_s_i
        elif ra == ra_d:
            im_use = im_d_i
        elif ra == ra_d2:
            im_use = im_d_i
        
        xs_c, ys_c = im_use.wcs.wcs_world2pix(ra_s, dec_s, 0)
        xr_c, yr_c = im_use.wcs.wcs_world2pix(ra_r, dec_r, 0)
        xd_c, yd_c = im_use.wcs.wcs_world2pix(ra_d, dec_d, 0)
        xd2_c, yd2_c = im_use.wcs.wcs_world2pix(ra_d2, dec_d2, 0)

        im_r_i = im_r_i.data
        im_s_i = im_s_i.data
        im_d_i = im_d_i.data
        

        
        
        dx_px = int(dx.value / .1)
        dy_px = int(dy.value / .1)
        x1 = int(  max(x-dx_px, 0)  )
        x2 = int(  min(x+dx_px, im_in_r.shape[1])  )
        y1 = int(  max(y-dy_px, 0)  )
        y2 = int(  min(y+dy_px, im_in_r.shape[0])  )
        
        dxc_r = xr_c - dx_px
        dyc_r = yr_c - dy_px
        dxc_s = xs_c - dx_px
        dyc_s = ys_c - dy_px
        dxc_d = xd_c - dx_px
        dyc_d = yd_c - dy_px
        dxc_d2 = xd2_c - dx_px
        dyc_d2 = yd2_c - dy_px     
        
        ##############################################################################################################################
        #Cutout JWST native resolution image

        dx_px_jwst = int(dx.value / jwst_ps)
        dy_px_jwst = int(dy.value / jwst_ps)

        im_jwst_i = Cutout2D(im_in_s_native, coord, size=(dx*2, dy*2), wcs=WCS(hdr_jwst)).data

        ##############################################################################################################################
        #Correct if on the edge of the image

        if ((x2-x1) < dx_px*2) or ((y2-y1) < dy_px*2):   
            im_r_new = np.full((dy_px*2, dx_px*2), np.nan)
            im_r_new[:im_r_i.shape[0], :im_r_i.shape[1]] = im_r_i
            im_r_i = im_r_new.copy()
            
            im_s_new = np.full((dy_px*2, dx_px*2), np.nan)
            im_s_new[:im_s_i.shape[0], :im_s_i.shape[1]] = im_s_i
            im_s_i = im_s_new.copy()
            
            im_d_new = np.full((dy_px*2, dx_px*2), np.nan)
            im_d_new[:im_d_i.shape[0], :im_d_i.shape[1]] = im_d_i
            im_d_i = im_d_new.copy()              

            dx_empty = dx_px*2 - (x2-x1)
            dy_empty = dy_px*2 - (y2-y1)   
            
            dx_empty_jwst = int(  np.floor(dx_empty * jwst_scale)  )
            dy_empty_jwst = int(  np.floor(dy_empty * jwst_scale)  )
                    
            im_jwst_new = np.full((dy_px_jwst*2, dx_px_jwst*2), np.nan)
            ind1 = im_jwst_i.shape[0] - dy_empty_jwst
            ind2 = im_jwst_i.shape[1] - dx_empty_jwst
            im_jwst_new[:ind1, :ind2] = im_jwst_i[dy_empty_jwst:, dx_empty_jwst:]
            im_jwst_i = im_jwst_new.copy()
            
            
        else:
            dx_empty = 0
            dy_empty = 0     
            dx_empty_jwst = 0
            dy_empty_jwst = 0
            
        ##############################################################################################################################
        #Choose normalization

        norm_r = ImageNormalize(im_r_i, stretch=AsinhStretch())
        norm_s = ImageNormalize(im_s_i, stretch=AsinhStretch())
        norm_d = ImageNormalize(im_d_i, stretch=AsinhStretch())
        norm_jwst = ImageNormalize(im_jwst_i, stretch=AsinhStretch())
        
        norm = norm_jwst
        
        ##############################################################################################################################
        #Plot cutouts with fluxes
        
        ax[i, 0].imshow(im_r_i, norm=norm, cmap='Greys_r', aspect='equal', origin='lower', interpolation='none')
        ax[i, 0].text(.01, .98, format_flux(m_r_og, merr_r_og, div_factor), color='orange', fontsize=12, va='top', ha='left', transform=ax[i, 0].transAxes, fontweight='bold')
        ax[i, 0].text(.01, .9, format_flux(m_rs_og, merr_rs_og, div_factor), color='r', fontsize=12, va='top', ha='left', transform=ax[i, 0].transAxes, fontweight='bold')
        
        ax[i, 1].imshow(im_s_i, norm=norm, cmap='Greys_r', aspect='equal', origin='lower', interpolation='none')
        ax[i, 1].text(.01, .98, format_flux(m_sr_og, merr_sr_og, div_factor), color='orange', fontsize=12, va='top', ha='left', transform=ax[i, 1].transAxes, fontweight='bold')
        ax[i, 1].text(.01, .9, format_flux(m_s_og, merr_s_og, div_factor), color='r', fontsize=12, va='top', ha='left', transform=ax[i, 1].transAxes, fontweight='bold')
        
        ax[i, 2].imshow(im_jwst_i, norm=norm, cmap='Greys_r', aspect='equal', origin='lower', interpolation='none')
        
        ax[i, 3].imshow(im_d_i, norm=norm, cmap='Greys_r', aspect='equal', origin='lower', interpolation='none')    
        ax[i, 3].text(.01, .98, format_flux(m_r, merr_r, div_factor), color='orange', fontsize=12, va='top', ha='left', transform=ax[i, 3].transAxes, fontweight='bold')
        ax[i, 3].text(.01, .9, format_flux(m_s, merr_s, div_factor), color='r', fontsize=12, va='top', ha='left', transform=ax[i, 3].transAxes, fontweight='bold')
        ax[i, 3].text(.01, .82, format_flux(m_d, merr_d, div_factor), color='m', fontsize=12, va='top', ha='left', transform=ax[i, 3].transAxes, fontweight='bold')
        
        ##############################################################################################################################
        #Plot background fluxes
        
        if i == 0:
            label1 = 'Background (DIFF, whole cutout)'
            label1a = 'Background (DIFF, {}"x{}" cutout)'.format(dx_big.value, dy_big.value)
            label2 = 'REF Flux (REF Aper)'
            label3 = 'SCI Flux (REF Aper)'
            label4 = 'DIFF Flux (REF Aper)'
            label5 = 'DIFF Flux (DIFF Aper)'
        else:
            label1 = label2 = label3 = label4 = None
        
        
        bmax = np.nanmax([.1, np.abs(m_r_og), np.abs(m_sr_og), np.abs(m_d), np.abs(m_r)])
        bins = np.linspace(-bmax, bmax, 50)
        p16, p50, p84 = np.nanpercentile(im_decorr_bkg.flatten()/div_factor, [16, 50, 84])
        p16_l, p50_l, p84_l = np.nanpercentile(im_decorr_bkg_local.flatten()/div_factor, [16, 50, 84])
        
        ax[i,4].hist(im_decorr_bkg.flatten()/div_factor, bins=bins, color='c', lw=1, alpha=.5, label=label1, density=True)
        ax[i,4].hist(im_decorr_bkg.flatten()/div_factor, bins=bins, color='k', lw=.5, histtype='step', density=True)
        ax[i,4].hist(im_decorr_bkg_local.flatten()/div_factor, bins=bins, color='g', lw=1, alpha=.5, label=label1a, density=True)
        ax[i,4].hist(im_decorr_bkg_local.flatten()/div_factor, bins=bins, color='k', lw=.5, histtype='step', density=True)
        
        ax[i,4].axvline(m_r_og, color='orange', lw=1, ls='-', label=label2)
        ax[i,4].axvline(m_sr_og, color='r', lw=1, ls='-', label=label3)
        ax[i,4].axvline(m_r, color='orange', lw=1, ls='--', label=label4)
        if diff_detected:
            ax[i,4].axvline(m_d, color='m', lw=1, ls='--', label=label5)
            
        txt = r'${{ {:.2f} }}_{{- {:.2f} }}^{{ + {:.2f} }}$'.format(p50, (p50-p16), (p84-p50))
        ax[i,4].text(.99, .98, txt, color='c', fontsize=10, va='top', ha='right', transform=ax[i, 4].transAxes, fontweight='bold')
        txt = r'${{ {:.2f} }}_{{- {:.2f} }}^{{ + {:.2f} }}$'.format(p50_l, (p50_l-p16_l), (p84_l-p50_l))
        ax[i,4].text(.99, .88, txt, color='g', fontsize=10, va='top', ha='right', transform=ax[i, 4].transAxes, fontweight='bold')
        
        ##############################################################################################################################
        #Cts to microJy
        ax[i,0].text(.98, .02, r'1 ct = {:.2e} $\mu$Jy'.format(ct2microjy_euclid), fontsize=10, color='y', fontweight='bold', va='bottom', ha='right', transform=ax[i, 0].transAxes)
        ax[i,1].text(.98, .02, r'1 DN = {:.2e} $\mu$Jy'.format(dn2microjy_jwst), fontsize=10, color='y', fontweight='bold', va='bottom', ha='right', transform=ax[i, 1].transAxes)

        ##############################################################################################################################
        #Make scalebar for 1" in diff image
        x2 = .98
        x1 = (x2*im_jwst_i.shape[1] - 1/jwst_ps) / im_jwst_i.shape[1]
        y = .85*im_jwst_i.shape[0] 
        ax[i, 2].hlines(y=y, xmin=x1, xmax=x2, transform=ax[i,2].get_yaxis_transform(), color='y', lw=2)
        ax[i, 2].text((x1+x2)/2, (y/im_jwst_i.shape[0] + .05)*im_jwst_i.shape[0], '1"', color='y', fontsize=11, va='center', ha='left', transform=ax[i,2].get_yaxis_transform(), fontweight='bold')
        
        ##############################################################################################################################
        #Draw apertures        
        dxvals = np.linspace(-dx_px, dx_px, 7500)[::2]
        cxx_s_in, cxy_s_in, cyy_s_in = catalog['CXX_SCI_INPUT'][n], catalog['CXY_SCI_INPUT'][n], catalog['CYY_SCI_INPUT'][n] 
        cxx_r_in, cxy_r_in, cyy_r_in = catalog['CXX_REF_INPUT'][n], catalog['CXY_REF_INPUT'][n], catalog['CYY_REF_INPUT'][n]
        cxx_s, cxy_s, cyy_s = catalog['CXX_SCI'][n], catalog['CXY_SCI'][n], catalog['CYY_SCI'][n]
        cxx_r, cxy_r, cyy_r = catalog['CXX_REF'][n], catalog['CXY_REF'][n], catalog['CYY_REF'][n]
        
        if diff_detected:
            if catalog['DIFF_DETECTED'][n]:
                cxx_d, cxy_d, cyy_d = catalog['CXX_DIFF'][n], catalog['CXY_DIFF'][n], catalog['CYY_DIFF'][n]
            if catalog['DIFF_DETECTED_n'][n]:
                cxx_d2, cxy_d2, cyy_d2 = catalog['CXX_DIFF_n'][n], catalog['CXY_DIFF_n'][n], catalog['CYY_DIFF_n'][n]
                
            if np.isnan(dxc_d):
                dxc_d_use = dxc_d2
                dyc_d_use = dyc_d2
                
                cxx_d_use = cxx_d2
                cxy_d_use = cxy_d2
                cyy_d_use = cyy_d2 
            else:
                dxc_d_use = dxc_d
                dyc_d_use = dyc_d 
                
                cxx_d_use = cxx_d
                cxy_d_use = cxy_d
                cyy_d_use = cyy_d      
        
        
        dxvals_r = np.zeros_like(im_r_i)
        dyvals_r = np.zeros_like(im_r_i)
        dxvals_s = np.zeros_like(im_s_i)
        dyvals_s = np.zeros_like(im_s_i)
        for j in range(im_r_i.shape[0]):
            dyvals_r[j] = j - (dy_px + dyc_r)
            dxvals_s[j] = j - (dy_px + dyc_s)
        for j in range(im_r_i.shape[1]):
            dxvals_r[:,j] = j - (dx_px + dxc_r)
            dyvals_s[:,j] = j - (dy_px + dyc_s)

            
        if diff_detected:
            dxvals_d = np.zeros_like(im_d_i)
            dyvals_d = np.zeros_like(im_d_i)
            for j in range(im_r_i.shape[0]):
                dyvals_d[j] = j - (dy_px + dyc_d_use)
            for j in range(im_r_i.shape[1]):
                dxvals_d[:,j] = j - (dx_px + dxc_d_use)
            

        emask_r_in = get_ellipse_mask(dxvals_r, dyvals_r, cxx_r_in, cxy_r_in, cyy_r_in, r=rad_r_in)
        emask_r = get_ellipse_mask(dxvals_r, dyvals_r, cxx_r, cxy_r, cyy_r, r=rad_r)
        # emask_r[np.isnan(im_r_i)] = False
        # emask_r[np.isnan(im_r_i)] = False
        # emask_r[im_r_i == 0.] = False
        # emask_r[im_r_i == 0.] = False
        
        # if i == 4:        
        #     sum1 = np.nansum(im_r_i[emask_r_in]/div_factor)
        #     sum2 = np.nansum(im_s_i[emask_r_in]/div_factor)
        #     sum3 = np.nansum(im_d_i[emask_r]/div_factor)
        #     print('For the bottom object (using the orange aperture):')
        #     print('Sum Euclid: {:.2f}'.format(sum1))
        #     print('Sum JWST: {:.2f}'.format(sum2))
        #     print('Sum DIFF: {:.2f}'.format(sum3))
        

        ex_s_in, ey_s_in = get_ellipse(dxvals, cxx_s_in, cxy_s_in, cyy_s_in, r=rad_s_in)
        ex_r_in, ey_r_in = get_ellipse(dxvals, cxx_r_in, cxy_r_in, cyy_r_in, r=rad_r_in)
        ex_s, ey_s = get_ellipse(dxvals, cxx_s, cxy_s, cyy_s, r=rad_s)
        ex_r, ey_r = get_ellipse(dxvals, cxx_r, cxy_r, cyy_r, r=rad_r)
        if diff_detected:
            ex_d, ey_d = get_ellipse(dxvals, cxx_d_use, cxy_d_use, cyy_d_use, r=rad_d)
            
        xi_r = ex_r + (dx_px + dxc_r)
        yi_r = ey_r + (dy_px + dyc_r)

        xi_s = ex_s + (dx_px + dxc_s)
        yi_s = ey_s + (dy_px + dyc_s)
        
        xi_r_in = ex_r_in + (dx_px + dxc_r)
        yi_r_in = ey_r_in + (dy_px + dyc_r)
        
        xi_s_in = ex_s_in + (dx_px + dxc_s)
        yi_s_in = ey_s_in + (dy_px + dyc_s)
        
        if diff_detected:
            xi_d = ex_d + (dx_px + dxc_d_use)
            yi_d = ey_d + (dy_px + dyc_d_use)
        
        
        for a in [ax[i,0], ax[i,1]]:        
            a.plot(xi_r_in, yi_r_in, color='orange', lw=1)
            a.plot(xi_s_in, yi_s_in, color='r', lw=1)
            if diff_detected:
                a.plot(xi_d, yi_d, color='m', lw=1)
            

        ax[i,3].plot(xi_r, yi_r, color='orange', lw=2)
        ax[i,3].plot(xi_s, yi_s, color='r', lw=2)
        if diff_detected:
            ax[i,3].plot(xi_d, yi_d, color='m', lw=2)
            
        ax[i,2].plot(xi_r_in*jwst_scale, yi_r_in*jwst_scale, color='orange', lw=1)
        ax[i,2].plot(xi_s_in*jwst_scale, yi_s_in*jwst_scale, color='r', lw=1)
        if diff_detected:
            ax[i,2].plot(xi_d*jwst_scale, yi_d*jwst_scale, color='m', lw=1)



        #Put point where center is
        # if ra == ra_s:
        #     ax[i,0].scatter(dx_px + dxc_r, dy_px + dyc_r, s=5, color='orange')
        #     ax[i,0].scatter(dx_px + dxc_s, dy_px + dyc_s, s=5, color='r')
        #     ax[i,1].scatter(dx_px + dxc_r, dy_px + dyc_r, s=5, color='orange')
        #     ax[i,1].scatter(dx_px + dxc_s, dy_px + dyc_s, s=5, color='r')
        #     # ax[i,2].scatter(dx_px_jwst, dy_px_jwst, s=5, color='orange')
        #     # ax[i,2].scatter(dx_px_jwst + (xr-x)*(10/3), (yr-y1)*(10/3), s=5, color='r')
        #     ax[i,3].scatter(dx_px + dxc_r, dy_px + dyc_r, s=5, color='orange')
        #     ax[i,3].scatter(dx_px + dxc_s, dy_px + dyc_s, s=5, color='r')
        # if ra == ra_r:
        #     ax[i,0].scatter(dx_px + dxc_r, dy_px + dyc_r, s=5, color='orange')
        #     ax[i,0].scatter(dx_px + dxc_s, dy_px + dyc_s, s=5, color='r')        
        #     ax[i,1].scatter(dx_px + dxc_r, dy_px + dyc_r, s=5, color='orange')
        #     ax[i,1].scatter(dx_px + dxc_s, dy_px + dyc_s, s=5, color='r')    
        #     ax[i,3].scatter(dx_px + dxc_r, dy_px + dyc_r, s=5, color='orange')
        #     ax[i,3].scatter(dx_px + dxc_s, dy_px + dyc_s, s=5, color='r')
        # if (ra == ra_d) or (ra == ra_d2):
        #     ax[i,0].scatter(dx_px + dxc_d, dy_px + dyc_d, s=5, color='darkgreen')
        #     ax[i,1].scatter(dx_px + dxc_d, dy_px + dyc_d, s=5, color='darkgreen')
        #     ax[i,3].scatter(dx_px + dxc_d, dy_px + dyc_d, s=5, color='darkgreen')
        
        ##############################################################################################################################
        #Draw checks and xs
            
        if ref_detected:
            ax[i,0].text(.05, .05, '✓', color='orange', fontsize=24, va='bottom', ha='left', transform=ax[i,0].transAxes, fontweight='bold')
        else:
            ax[i,0].text(.05, .05, '×', color='orange', fontsize=24, va='bottom', ha='left', transform=ax[i,0].transAxes, fontweight='bold')

            
        if sci_detected:
            ax[i,1].text(.05, .05, '✓', color='r', fontsize=24, va='bottom', ha='left', transform=ax[i,1].transAxes, fontweight='bold')
        else:
            ax[i,1].text(.05, .05, '×', color='r', fontsize=24, va='bottom', ha='left', transform=ax[i,1].transAxes, fontweight='bold')
            
            
        if diff_detected:
            ax[i,3].text(.05, .05, '✓', color='m', fontsize=24, va='bottom', ha='left', transform=ax[i,3].transAxes, fontweight='bold')
        else:
            ax[i,3].text(.05, .05, '×', color='m', fontsize=24, va='bottom', ha='left', transform=ax[i,3].transAxes, fontweight='bold')
        
        ##############################################################################################################################
        #Set axis limits and ticks
        
        for j in range(4):
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            
        ax[i,4].set_yticks([])


        for j in [0,1,3]:
            ax[i,j].set_xlim(0, dx_px*2)
            ax[i,j].set_ylim(0, dy_px*2)
            
        ax[i,2].set_xlim(0, dx_px*2 * jwst_scale)
        ax[i,2].set_ylim(0, dy_px*2 * jwst_scale)
        
        
    ##############################################################################################################################
    #Set axis labels and titles
        
    ax[0,4].legend(loc='lower left', bbox_to_anchor=(0, 1), fontsize=10)
    ax[-1,4].set_xlabel(r'Flux [$\mu$Jy]', fontsize=18)

    ax[0, 0].set_title('REF (Euclid)', fontsize=11, color='orange', fontweight='bold')
    ax[0, 1].set_title('SCI (JWST)\nResampled', fontsize=11, color='red', fontweight='bold')
    ax[0, 2].set_title('SCI (JWST)\nNative', fontsize=11, color='red', fontweight='bold')
    ax[0, 3].set_title('DIFF', fontsize=11, color='m', fontweight='bold')

    plt.subplots_adjust(wspace=0, hspace=.1)
    
    if output_fname is not None:
        plt.savefig(output_fname, dpi=300, bbox_inches='tight')
        
    if show:
        plt.show()
    
    plt.cla()
    plt.clf()
    plt.close()

    shutil.rmtree(tmpdir)
    
    return