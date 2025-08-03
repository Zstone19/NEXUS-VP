import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch, make_lupton_rgb, make_rgb, LuptonAsinhZscaleStretch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
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



def format_flux(v, err):
    v2 = v
    err = err
    return r'{0:.2f} $\pm$ {1:.2f} uJy'.format(v2, err)

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



def make_source_cutouts_simple_pubcat(catalog, catalog_euc, catalog_nex, inds,
                                      fname_ref, fname_sci, fname_diff, fname_jwst, fname_mask, 
                                      euc_band='J', dx_cutout=3, dy_cutout=3, unit='Jy',
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
    
    if np.max(inds) >= len(catalog):
        inds_new = []
        for i in inds:
            if i < len(catalog):
                inds_new.append(i)
        inds = inds_new    
        

    for i in inds:
        assert (not np.isnan(catalog['RA_'+euc_band][i])) | (not np.isnan(catalog['RA_n_'+euc_band][i])), 'Source with index {} is not found in DIFF image in {} band!'.format(i, euc_band)
    
    
    with fits.open(fname_jwst) as hdu:
        im_jwst = hdu[0].data
        hdr_jwst = hdu[0].header
        wcs_jwst = WCS(hdr_jwst)    
        
    im_jwst *= 10**( (zp - hdr_jwst['MAG_ZP']) / 2.5 )
    
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
        
        with fits.open(fname_ref) as hdul:
            im_in_r = hdul[0].data
            hdr_r = hdul[0].header

        with fits.open(fname_sci) as hdul:
            im_in_s = hdul[0].data
            hdr_s = hdul[0].header

        with fits.open(fname_diff) as hdul:
            im_out = hdul[0].data
            hdr_d = hdul[0].header
            
        with fits.open(fname_mask) as hdul:
            im_mask = hdul[0].data.astype(bool)
            
        hdr_d2 = hdr_d.copy()


        im_in_r[im_mask] = np.nan
        im_in_s[im_mask] = np.nan
        im_out[im_mask] = np.nan

        wcs_in = WCS(hdr_r)

        
        ra_d = catalog['RA_'+euc_band][n]
        dec_d = catalog['DEC_'+euc_band][n]
        xd, yd = wcs_in.wcs_world2pix(ra_d, dec_d, 0)
        
        ra_d2 = catalog['RA_n_'+euc_band][n]
        dec_d2 = catalog['DEC_n_'+euc_band][n]
        xd2, yd2 = wcs_in.wcs_world2pix(ra_d2, dec_d2, 0)
        
        # fn = catalog['FLUX_NEXUS'][n]
        # fe = catalog['FLUX_EUCLID'][n]
        
        # ref_detected = True
        # if np.isnan(catalog[n]['FLUX_EUCLID']) or isinstance(catalog[n]['FLUX_EUCLID'], np.ma.core.MaskedConstant) or (fe < 0.):
        #     ref_detected = False
        
        # sci_detected = True
        # if np.isnan(catalog[n]['FLUX_NEXUS']) or isinstance(catalog[n]['FLUX_NEXUS'], np.ma.core.MaskedConstant) or (fn < 0.):
        #     sci_detected = False


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
        
        cat_euc_i = catalog_euc[mask_euc].copy()
        cat_nex_i = catalog_nex[mask_nex].copy()     
        
        # print('Euclid sources in cutout:', len(cat_euc_i))
        # print('Nexus sources in cutout:', len(cat_nex_i))        
                        
        ##############################################################################################################################
        #Get fluxes/errors    
        
        # if ref_detected:
        #     f_r = catalog['FLUX_EUCLID'][n]
        #     ferr_r = catalog['FLUXERR_EUCLID'][n]
        #     m_r, merr_r = flux_to_mag_w_err(f_r, ferr_r, zp=23.9)

        # if sci_detected:
        #     f_s = catalog['FLUX_NEXUS'][n]
        #     ferr_s = catalog['FLUXERR_NEXUS'][n]
        #     m_s, merr_s = flux_to_mag_w_err(f_s, ferr_s, zp=23.9)

        f_d1 = catalog['AperFlux_'+euc_band][n]/div_factor
        f_d2 = catalog['AperFlux_n_'+euc_band][n]/div_factor
        ferr_d1 = catalog['AperFluxErr_'+euc_band][n]/div_factor
        ferr_d2 = catalog['AperFluxErr_n_'+euc_band][n]/div_factor
        if np.isnan(f_d1) or isinstance(f_d1, np.ma.core.MaskedConstant) or (f_d1 < 0.):
            f_d = f_d2
            ferr_d = ferr_d2
        else:
            f_d = f_d1
            ferr_d = ferr_d1
            
        # m_d, merr_d = flux_to_mag_w_err(f_d, ferr_d, zp=zp)
        m_d = f_d
        merr_d = ferr_d
            
        vmax = 1.5*f_d
        
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
        
        # xs_c, ys_c = im_use.wcs.wcs_world2pix(ra_s, dec_s, 0)
        # xr_c, yr_c = im_use.wcs.wcs_world2pix(ra_r, dec_r, 0)
        xd_c, yd_c = im_use.wcs.wcs_world2pix(ra_d, dec_d, 0)
        xd2_c, yd2_c = im_use.wcs.wcs_world2pix(ra_d2, dec_d2, 0)  
        
        # xs_c_jwst, ys_c_jwst = im_jwst_i.wcs.wcs_world2pix(ra_s, dec_s, 0)      
        # xr_c_jwst, yr_c_jwst = im_jwst_i.wcs.wcs_world2pix(ra_r, dec_r, 0)
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
        
        # dxc_r = xr_c - dx_px
        # dyc_r = yr_c - dy_px
        # dxc_s = xs_c - dx_px
        # dyc_s = ys_c - dy_px
        dxc_d = xd_c - dx_px
        dyc_d = yd_c - dy_px
        dxc_d2 = xd2_c - dx_px
        dyc_d2 = yd2_c - dy_px   
        
        # dxc_r_jwst = xr_c_jwst - dx_px_jwst
        # dyc_r_jwst = yr_c_jwst - dy_px_jwst
        # dxc_s_jwst = xs_c_jwst - dx_px_jwst
        # dyc_s_jwst = ys_c_jwst - dy_px_jwst
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

        bright_mask_r = np.isfinite(im_r_i.data) & (im_r_i.data > vmax)
        bright_mask_s = np.isfinite(im_s_i.data) & (im_s_i.data > vmax)
        bright_mask_d = np.isfinite(im_d_i.data) & (im_d_i.data > vmax)

        norm_r = ImageNormalize(im_r_i.data[~bright_mask_r], interval=ZScaleInterval(), stretch=AsinhStretch())
        norm_s = ImageNormalize(im_s_i.data[~bright_mask_s], interval=ZScaleInterval(), stretch=AsinhStretch())
        norm_d = ImageNormalize(im_d_i.data[~bright_mask_d], interval=ZScaleInterval(), stretch=AsinhStretch())        
        norm_jwst = ImageNormalize(im_jwst_i.data, interval=ZScaleInterval(), stretch=AsinhStretch())
        
        ##############################################################################################################################
        #Plot cutouts with fluxes
        ax = np.zeros(Ncol, dtype=object) 
        for j in range(Ncol):
            if j != 2:
                ax[j] = plt.subplot(gs_i[0, j], projection=wcs_use)
            else:
                ax[j] = plt.subplot(gs_i[0, j], projection=im_jwst_i.wcs)

        ax[0].imshow(im_r_i.data, cmap=cmap, origin='lower', interpolation='auto', aspect='equal', norm=norm_s)
        ax[1].imshow(im_s_i.data, cmap=cmap, origin='lower', interpolation='auto', aspect='equal', norm=norm_s)
        ax[2].imshow(im_jwst_i.data, cmap=cmap, origin='lower', interpolation='auto', aspect='equal', norm=norm_jwst)
        ax[3].imshow(im_d_i.data, cmap=cmap, origin='lower', interpolation='auto', aspect='equal', norm=norm_s)  

        if show_flux:
        #     if ref_detected:
        #         ax[0].text(.01, .98, format_flux(m_r, merr_r), color='orange', fontsize=12, va='top', ha='left', transform=ax[0].transAxes, fontweight='bold')

        #     if sci_detected:
        #         ax[1].text(.01, .98, format_flux(m_s, merr_s), color='r', fontsize=12, va='top', ha='left', transform=ax[1].transAxes, fontweight='bold')
            
            # ax[3].text(.01, .98, format_flux(m_r, merr_r, div_factor), color='orange', fontsize=12, va='top', ha='left', transform=ax[3].transAxes, fontweight='bold')
            # ax[3].text(.01, .9, format_flux(m_s, merr_s, div_factor), color='r', fontsize=12, va='top', ha='left', transform=ax[3].transAxes, fontweight='bold')
            ax[3].text(.01, .98, format_flux(m_d, merr_d), color='m', fontsize=12, va='top', ha='left', transform=ax[3].transAxes, fontweight='bold')
        
        ##############################################################################################################################
        # Draw locations of catalog/DIFF sources
        
        if len(cat_euc_i) > 0:
            for j in range(Ncol):
                ax[j].errorbar(cat_euc_i['RIGHT_ASCENSION'].data, cat_euc_i['DECLINATION'].data, ms=7.5, color='orange', fmt='x', transform=ax[j].get_transform('world'))
        if len(cat_nex_i) > 0:
            for j in range(Ncol):
                ax[j].errorbar(cat_nex_i['RA'].data, cat_nex_i['DEC'].data, ms=7.5, color='r', fmt='x', transform=ax[j].get_transform('world'))

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
            
        del im_in_r, im_in_s, im_out, wcs_in
        del im_r_i, im_s_i, im_d_i, im_jwst_i, im_use
        del xd, yd, xd2, yd2
        del ra_d, dec_d, ra_d2, dec_d2
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

def make_source_cutouts_simple_pubcat_multiband(catalog, catalog_euc, catalog_nex, inds,
                                      fname_ref_y, fname_sci_y, fname_diff_y, fname_jwst_y, fname_mask_y,
                                      fname_ref_j, fname_sci_j, fname_diff_j, fname_jwst_j, fname_mask_j,
                                      fname_ref_h, fname_sci_h, fname_diff_h, fname_jwst_h, fname_mask_h,
                                      dx_cutout=3, dy_cutout=3, unit='Jy', show_flux=False,
                                      output_fname=None, show=False):
        
    
    cmap = mpl.cm.Greys_r
    cmap.set_bad(color='c', alpha=.25)
    
    if unit == 'Jy':
        div_factor = 1e-6
        zp = 8.9
    else:
        div_factor = 1
        zp = 23.9
    
    if np.max(inds) >= len(catalog):
        inds_new = []
        for i in inds:
            if i < len(catalog):
                inds_new.append(i)
        inds = inds_new    
    
    euc_bands = 'YJH'
    
    ########################################
    #Y
    with fits.open(fname_jwst_y) as hdu:
        im_jwst_y = hdu[0].data
        hdr_jwst_y = hdu[0].header
        wcs_jwst_y = WCS(hdr_jwst_y)
    #J
    with fits.open(fname_jwst_j) as hdu:
        im_jwst_j = hdu[0].data
        hdr_jwst_j = hdu[0].header
        wcs_jwst_j = WCS(hdr_jwst_j)    
    #H
    with fits.open(fname_jwst_h) as hdu:
        im_jwst_h = hdu[0].data
        hdr_jwst_h = hdu[0].header
        wcs_jwst_h = WCS(hdr_jwst_h)
    
    im_jwst_y *= 10**( (zp - hdr_jwst_y['MAG_ZP']) / 2.5 )
    im_jwst_j *= 10**( (zp - hdr_jwst_j['MAG_ZP']) / 2.5 )
    im_jwst_h *= 10**( (zp - hdr_jwst_h['MAG_ZP']) / 2.5 )
    ims_jwst = [im_jwst_y, im_jwst_j, im_jwst_h]
    wcss_jwst = [wcs_jwst_y, wcs_jwst_j, wcs_jwst_h]

    ########################################
    
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
    
    
    ####################################################################
    #Get images
    
    #Y
    with fits.open(fname_ref_y) as hdul:
        im_in_r_y = hdul[0].data
        hdr_r_y = hdul[0].header
    with fits.open(fname_sci_y) as hdul:
        im_in_s_y = hdul[0].data
        hdr_s_y = hdul[0].header
    with fits.open(fname_diff_y) as hdul:
        im_out_y = hdul[0].data
        hdr_d_y = hdul[0].header
    with fits.open(fname_mask_y) as hdul:
        im_mask_y = hdul[0].data.astype(bool)
    hdr_d2_y = hdr_d_y.copy()
    
    #J
    with fits.open(fname_ref_j) as hdul:
        im_in_r_j = hdul[0].data
        hdr_r_j = hdul[0].header
    with fits.open(fname_sci_j) as hdul:
        im_in_s_j = hdul[0].data
        hdr_s_j = hdul[0].header
    with fits.open(fname_diff_j) as hdul:
        im_out_j = hdul[0].data
        hdr_d_j = hdul[0].header
    with fits.open(fname_mask_j) as hdul:
        im_mask_j = hdul[0].data.astype(bool)
    hdr_d2_j = hdr_d_j.copy()
    
    #H
    with fits.open(fname_ref_h) as hdul:
        im_in_r_h = hdul[0].data
        hdr_r_h = hdul[0].header
    with fits.open(fname_sci_h) as hdul:
        im_in_s_h = hdul[0].data
        hdr_s_h = hdul[0].header
    with fits.open(fname_diff_h) as hdul:
        im_out_h = hdul[0].data
        hdr_d_h = hdul[0].header
    with fits.open(fname_mask_h) as hdul:
        im_mask_h = hdul[0].data.astype(bool)
    hdr_d2_h = hdr_d_h.copy()
    
    ims_in_r = [im_in_r_y, im_in_r_j, im_in_r_h]
    ims_in_s = [im_in_s_y, im_in_s_j, im_in_s_h]
    ims_out = [im_out_y, im_out_j, im_out_h]
    ims_mask = [im_mask_y, im_mask_j, im_mask_h]
    hdrs_r = [hdr_r_y, hdr_r_j, hdr_r_h]
    hdrs_s = [hdr_s_y, hdr_s_j, hdr_s_h]
    hdrs_d = [hdr_d_y, hdr_d_j, hdr_d_h]
    hdrs_d2 = [hdr_d2_y, hdr_d2_j, hdr_d2_h]

    for i in range(len(ims_in_r)):
        ims_in_r[i][ims_mask[i]] = 0.
        ims_in_s[i][ims_mask[i]] = 0.
        ims_out[i][ims_mask[i]] = 0.
        
    wcss_in = [WCS(h) for h in hdrs_r]
    
    ####################################################################

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
        
        ####################################################################
        #Get source positions
        xs = []
        ys = []
        ras = []
        decs = []
        ims_use = []
        
        ras_obj = []
        decs_obj = []
        
        for i, b in enumerate(euc_bands):        
            ra_d = catalog['RA_'+b][n]
            dec_d = catalog['DEC_'+b][n]
            xd, yd = wcss_in[i].wcs_world2pix(ra_d, dec_d, 0)
            
            ra_d2 = catalog['RA_n_'+b][n]
            dec_d2 = catalog['DEC_n_'+b][n]
            xd2, yd2 = wcss_in[i].wcs_world2pix(ra_d2, dec_d2, 0)

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

            ra_obj = ra
            dec_obj = dec
            
            xs.append(x)
            ys.append(y)
            ras.append(ra)
            decs.append(dec)
            ims_use.append(im_use)
            ras_obj.append(ra_obj)
            decs_obj.append(dec_obj)
        
        ##############################################################################################################################
        #If close to edge, correct center
        
        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            
            if np.isnan(x) and np.isnan(y):
                continue
            
            close_top = False
            close_bottom = False
            close_left = False
            close_right = False
            if x + dx.value/euclid_ps > ims_in_r[i].shape[1]:
                close_right = True
                print(f'{i} Close to right edge')
            if x - dx.value/euclid_ps < 0:
                close_left = True
                print(f'{i} Close to left edge')
            if y + dy.value/euclid_ps > ims_in_r[i].shape[0]:
                close_top = True
                print(f'{i} Close to top edge')
            if y - dy.value/euclid_ps < 0:
                close_bottom = True
                print(f'{i} Close to bottom edge')
                
                
            rac, decc = wcss_in[i].wcs_pix2world(x, y, 0)
                
            if close_right:
                x -= np.abs(x - ims_in_r[i].shape[1] + dx.value/euclid_ps)
                rac, decc = wcss_in[i].wcs_pix2world(x, y, 0)
            if close_left:
                x += np.abs(x - dx.value/euclid_ps)
                rac, decc = wcss_in[i].wcs_pix2world(x, y, 0)
            if close_top:
                y -= np.abs(y - ims_in_r[i].shape[0] + dy.value/euclid_ps)
                rac, decc = wcss_in[i].wcs_pix2world(x, y, 0)
            if close_bottom:
                y += np.abs(y - dy.value/euclid_ps)
                rac, decc = wcss_in[i].wcs_pix2world(x, y, 0)
                
        xc = x
        yc = y
            
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
        
        cat_euc_i = catalog_euc[mask_euc].copy()
        cat_nex_i = catalog_nex[mask_nex].copy()           
                        
        ##############################################################################################################################
        #Get fluxes/errors    

        ms_d = []
        merrs_d = []

        for b in euc_bands:

            f_d1 = catalog['AperFlux_'+b][n]/div_factor
            f_d2 = catalog['AperFlux_n_'+b][n]/div_factor
            ferr_d1 = catalog['AperFluxErr_'+b][n]/div_factor
            ferr_d2 = catalog['AperFluxErr_n_'+b][n]/div_factor
            if np.isnan(f_d1) or isinstance(f_d1, np.ma.core.MaskedConstant) or (f_d1 < 0.):
                f_d = f_d2
                ferr_d = ferr_d2
            else:
                f_d = f_d1
                ferr_d = ferr_d1
                
            # m_d, merr_d = flux_to_mag_w_err(f_d, ferr_d, zp=zp)
            m_d = f_d
            merr_d = ferr_d
            
            ms_d.append(m_d)
            merrs_d.append(merr_d)
            
        ##############################################################################################################################
        #Get source cutouts
        ims_r_i = []
        ims_s_i = []
        ims_d_i = []
        
        coord = SkyCoord(rac, decc, unit=(u.deg, u.deg))
        for i in range(len(ims_in_r)):        
            im_r_i = Cutout2D(ims_in_r[i], coord, size=(dx*2, dy*2), wcs=wcss_in[i])
            im_s_i = Cutout2D(ims_in_s[i], coord, size=(dx*2, dy*2), wcs=wcss_in[i])
            im_d_i = Cutout2D(ims_out[i], coord, size=(dx*2, dy*2), wcs=wcss_in[i])
            
            ims_r_i.append(im_r_i)
            ims_s_i.append(im_s_i)
            ims_d_i.append(im_d_i)

        ###############################################################################################################################
        #Get JWST cutout

        dx_px_jwst = int(dx.value / jwst_ps)
        dy_px_jwst = int(dy.value / jwst_ps)

        ims_jwst_i = []
        for i in range(len(ims_jwst)):
            im_jwst_i = Cutout2D(ims_jwst[i], coord, size=(dx*2, dy*2), wcs=wcss_jwst[i])
            ims_jwst_i.append(im_jwst_i)
        
        ##############################################################################################################################
        #Get center of cutout

        if im_use == 'diff':
            im_use = ims_d_i[0]
            wcs_use = im_use.wcs
        elif im_use == 'diff_n':
            im_use = ims_d_i[0]
            wcs_use = im_use.wcs
        
        # xs_c, ys_c = im_use.wcs.wcs_world2pix(ra_s, dec_s, 0)
        # xr_c, yr_c = im_use.wcs.wcs_world2pix(ra_r, dec_r, 0)
        xd_c, yd_c = im_use.wcs.wcs_world2pix(ra_d, dec_d, 0)
        xd2_c, yd2_c = im_use.wcs.wcs_world2pix(ra_d2, dec_d2, 0)  
        
        # xs_c_jwst, ys_c_jwst = im_jwst_i.wcs.wcs_world2pix(ra_s, dec_s, 0)      
        # xr_c_jwst, yr_c_jwst = im_jwst_i.wcs.wcs_world2pix(ra_r, dec_r, 0)
        xd_c_jwst, yd_c_jwst = ims_jwst_i[0].wcs.wcs_world2pix(ra_d, dec_d, 0)
        xd2_c_jwst, yd2_c_jwst = ims_jwst_i[0].wcs.wcs_world2pix(ra_d2, dec_d2, 0)
        
        xcat_euc, ycat_euc = im_use.wcs.wcs_world2pix(cat_euc_i['RIGHT_ASCENSION'].data, cat_euc_i['DECLINATION'].data, 0)
        xcat_nex, ycat_nex = im_use.wcs.wcs_world2pix(cat_nex_i['RA'].data, cat_nex_i['DEC'].data, 0)
        xcat_euc_jwst, ycat_euc_jwst = ims_jwst_i[0].wcs.wcs_world2pix(cat_euc_i['RIGHT_ASCENSION'].data, cat_euc_i['DECLINATION'].data, 0)
        xcat_nex_jwst, ycat_nex_jwst = ims_jwst_i[0].wcs.wcs_world2pix(cat_nex_i['RA'].data, cat_nex_i['DEC'].data, 0)

        
        
        dx_px = int(dx.value / euclid_ps)
        dy_px = int(dy.value / euclid_ps)
        dx_px_jwst = int(dx.value / jwst_ps)
        dy_px_jwst = int(dy.value / jwst_ps)
        
        # dxc_r = xr_c - dx_px
        # dyc_r = yr_c - dy_px
        # dxc_s = xs_c - dx_px
        # dyc_s = ys_c - dy_px
        dxc_d = xd_c - dx_px
        dyc_d = yd_c - dy_px
        dxc_d2 = xd2_c - dx_px
        dyc_d2 = yd2_c - dy_px   
        
        # dxc_r_jwst = xr_c_jwst - dx_px_jwst
        # dyc_r_jwst = yr_c_jwst - dy_px_jwst
        # dxc_s_jwst = xs_c_jwst - dx_px_jwst
        # dyc_s_jwst = ys_c_jwst - dy_px_jwst
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

        # bright_mask_r = np.isfinite(im_r_i.data) & (im_r_i.data > vmax)
        # bright_mask_s = np.isfinite(im_s_i.data) & (im_s_i.data > vmax)
        # bright_mask_d = np.isfinite(im_d_i.data) & (im_d_i.data > vmax)

        norm_r = ImageNormalize(ims_r_i[0].data, interval=ZScaleInterval(), stretch=AsinhStretch())
        norm_s = ImageNormalize(ims_s_i[0].data, interval=ZScaleInterval(), stretch=AsinhStretch())
        norm_d = ImageNormalize(ims_d_i[0].data, interval=ZScaleInterval(), stretch=AsinhStretch())        
        # norm_jwst = ImageNormalize(im_jwst_i.data, interval=ZScaleInterval(), stretch=AsinhStretch())
        
        ##############################################################################################################################
        #Plot cutouts with fluxes
        ax = np.zeros(Ncol, dtype=object) 
        for j in range(Ncol):
            if j != 2:
                ax[j] = plt.subplot(gs_i[0, j], projection=wcs_use)
            else:
                ax[j] = plt.subplot(gs_i[0, j], projection=ims_jwst_i[0].wcs)
                
                
        #stretch = LuptonAsinhZscaleStretch(im_s_i[0].data)
                
        im_r_tot = make_lupton_rgb(ims_r_i[0].data, ims_r_i[1].data, ims_r_i[2].data, stretch=.05, Q=15)
        im_s_tot = make_lupton_rgb(ims_s_i[0].data, ims_s_i[1].data, ims_s_i[2].data, stretch=.05, Q=15)
        im_jwst_tot = make_lupton_rgb(ims_jwst_i[0].data, ims_jwst_i[1].data, ims_jwst_i[2].data, stretch=.01, Q=20)
        im_d_tot = make_lupton_rgb(ims_d_i[0].data, ims_d_i[1].data, ims_d_i[2].data, stretch=1e-8, Q=80)

        ax[0].imshow(im_r_tot, cmap=cmap, origin='lower', interpolation='auto', aspect='equal')
        ax[1].imshow(im_s_tot, cmap=cmap, origin='lower', interpolation='auto', aspect='equal')
        ax[2].imshow(im_jwst_tot, cmap=cmap, origin='lower', interpolation='auto', aspect='equal')
        
        ax[3].imshow(np.full(ims_d_i[0].data.shape, 1e5), cmap='Greys', origin='lower', interpolation='auto', aspect='equal')
        ax[3].imshow(ims_d_i[0].data, cmap='Reds', origin='lower', interpolation='auto', aspect='equal', alpha=.5, norm=norm_d)  
        ax[3].imshow(ims_d_i[1].data, cmap='Greens', origin='lower', interpolation='auto', aspect='equal', alpha=.5, norm=norm_d)
        ax[3].imshow(ims_d_i[2].data, cmap='Blues', origin='lower', interpolation='auto', aspect='equal', alpha=.5, norm=norm_d)

        # if show_flux:
        #     ax[3].text(.01, .98, format_flux(m_d, merr_d), color='m', fontsize=12, va='top', ha='left', transform=ax[3].transAxes, fontweight='bold')
        
        ##############################################################################################################################
        # Draw locations of catalog/DIFF sources
        
        if len(cat_euc_i) > 0:
            for j in range(Ncol):
                ax[j].errorbar(cat_euc_i['RIGHT_ASCENSION'].data, cat_euc_i['DECLINATION'].data, ms=7.5, color='orange', fmt='x', transform=ax[j].get_transform('world'))
        if len(cat_nex_i) > 0:
            for j in range(Ncol):
                ax[j].errorbar(cat_nex_i['RA'].data, cat_nex_i['DEC'].data, ms=7.5, color='r', fmt='x', transform=ax[j].get_transform('world'))

        for j in range(Ncol):
            ax[j].errorbar(ras_obj, decs_obj, ms=7.5, color='m', fmt='x', transform=ax[j].get_transform('world'))

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
            
        del ims_r_i, ims_s_i, ims_d_i, ims_jwst_i, ims_use
        del xs, ys, ras, decs, ras_obj, decs_obj, rac, decc
        # del norm_r, norm_s, norm_d, norm_jwst
        
        
        
    
    if output_fname is not None:
        plt.savefig(output_fname)#, dpi=700, bbox_inches='tight')
        
    if show:
        plt.show()
    
    plt.cla()
    plt.clf()
    plt.close()
    
    return