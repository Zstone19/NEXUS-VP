import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, ZScaleInterval, AsinhStretch
from matplotlib.patches import Circle
from astropy.visualization.wcsaxes import add_scalebar
import matplotlib.patheffects as PathEffects
 

import warnings
import numpy as np
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats

from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D


###############################################################################

def get_fluxes(im_diff, im_var, im_mask, x, y, fname_apcorr, band, ps_grid=.03, neg=False, r1=1, r2=2, subtract_sky=True):
    circ_ap = CircularAperture(zip(x, y), r=r1/ps_grid)
    if subtract_sky:
        ann_ap = CircularAnnulus(zip(x, y), r_in=r1/ps_grid, r_out=r2/ps_grid)

    circ_npx = circ_ap.area
    if subtract_sky:
        ann_npx = ann_ap.area
    
    circ_stats = ApertureStats(im_diff, circ_ap, error=np.sqrt(im_var), mask=im_mask, sum_method='exact')
    circ_fluxes = circ_stats.sum
    circ_fluxerrs = circ_stats.sum_err
    if subtract_sky:
        ann_stats = ApertureStats(im_diff, ann_ap, error=np.sqrt(im_var), mask=im_mask, sum_method='exact')
        ann_fluxes = ann_stats.sum
        ann_fluxerrs = ann_stats.sum_err
    
    if neg:
        circ_fluxes *= -1
        if subtract_sky:
            ann_fluxes *= -1
        
    ###############################################################
    # Correct for aperture

    corrdat = Table.read(fname_apcorr, format='ascii.commented_header')
    ap_corr = corrdat['ApertureCorrection'][corrdat['Band'] == band]
    ann_corr = corrdat['SkyCorrection'][corrdat['Band'] == band]

    circ_corr_fluxes = circ_fluxes * ap_corr
    circ_corr_fluxerrs = circ_fluxerrs * ap_corr
    if subtract_sky:
        ann_corr_fluxes = ann_fluxes * ann_corr
        ann_corr_fluxerrs = ann_fluxerrs * ann_corr
    
    ################################################################
    # Subtract background
    
    if subtract_sky:
        sky_flux_perpx = ann_corr_fluxes / ann_npx
        sky_fluxerr_perpx = ann_corr_fluxerrs / ann_npx

        circ_skyfluxes = sky_flux_perpx * circ_npx
        circ_skyfluxerrs = sky_fluxerr_perpx * circ_npx

        circ_fluxes_skysub = circ_corr_fluxes - circ_skyfluxes
        circ_fluxerrs_skysub = np.sqrt(circ_corr_fluxerrs**2 + circ_skyfluxerrs**2)
    else:
        circ_fluxes_skysub = circ_corr_fluxes
        circ_fluxerrs_skysub = circ_corr_fluxerrs
    
    return circ_fluxes_skysub, circ_fluxerrs_skysub


def flux_to_mag(flux, flux_err, zp=23.90):    
    mag = -2.5 * np.log10(np.abs(flux)) + zp
    mag_err = 1.0857 * flux_err / np.abs(flux)
    return mag, mag_err

def nexstack_to_ujy(f, zp=28.0865):
    f_ujy = f * 10**( (23.9 - zp)/2.5 )
    return f_ujy    

################################################################################

def single_object_plot(ind, imdir_sfft, imdir_sub, catdat, band, 
                       aper=0.2, dx_arcsec=3., dy_arcsec=3., show_title=True,
                       show_position=True, show_mag=True, show_aperture=True, show_scalebar=True,
                       show=False, output_fname=None, ax_in=None):
    
    if band in ['F090W', 'F115W', 'F150W', 'F200W', 'F277W']:
        ps = .031
    else:
        ps = .063
        
        
    r_vals = np.array([0.2, 0.3, 0.5])
    ind_diff = [3,4,5]
    ind_nexcat = [0,1,2]    
    
    flux_ind_nex = ind_nexcat[ np.argwhere(r_vals == aper).flatten()[0] ]
    flux_ind_diff = ind_diff[ np.argwhere(r_vals == aper).flatten()[0] ]
        
    f_b1_r = catdat['FLUX_APER_WideStack'].data[ind, flux_ind_nex]
    if np.isnan(f_b1_r):
        b1_exist = False
    else:
        b1_exist = True


    if ind >= len(catdat):
        return


    
                
    #Get image filenames
    fname_diff_b1_sub = imdir_sub + 'output/nexus_deep01_{}.subdiff.fits'.format(band)
        

    fname_b1_r = imdir_sfft + 'input/nexus_wide01_{}.fits'.format(band)
    fname_b1_s = imdir_sfft + 'input/nexus_deep01_{}.fits'.format(band)
            
    fname_diff_b1_sfft = imdir_sfft + 'output/nexus_deep01_{}.sfftdiff.decorr.combined.fits'.format(band)
    fname_snr_b1 = imdir_sfft + 'output/nexus_deep01_{}.sfftdiff.snr.combined.fits'.format(band)
    
    # fname_noise_b1_r = imdir_sfft + 'noise/nexus_wide01_{}.noise.fits'.format(band)
    # fname_noise_b1_s = imdir_sfft + 'noise/nexus_deep01_{}.noise.fits'.format(band)
    
    
    #Load images
    with warnings.catch_warnings(action="ignore"):
        
        if b1_exist:
            with fits.open(fname_b1_r) as hdul:
                im_b1_r = hdul[0].data
                wcs_b1_r = WCS(hdul[0].header)
            with fits.open(fname_b1_s) as hdul:
                im_b1_s = hdul[0].data
                wcs_b1_s = WCS(hdul[0].header)            
            
                
            with fits.open(fname_diff_b1_sfft) as hdul:
                im_diff_b1_sfft = hdul[0].data
                wcs_diff_b1_sfft = WCS(hdul[0].header)
                
            with fits.open(fname_diff_b1_sub) as hdul:
                im_diff_b1_sub = hdul[0].data
                wcs_diff_b1_sub = WCS(hdul[0].header)
                
                
            im_mask_b1 = (im_b1_r == 0.) | (im_b1_s == 0.) | np.isnan(im_b1_r) | np.isnan(im_b1_s)
                                  
            with fits.open(fname_snr_b1) as hdul:
                im_snr_b1 = hdul[0].data
                wcs_snr_b1 = WCS(hdul[0].header)
       
    #Mask images
    if b1_exist:
        im_b1_r[im_mask_b1] = np.nan
        im_b1_s[im_mask_b1] = np.nan
        im_diff_b1_sub[im_mask_b1] = np.nan
        im_diff_b1_sfft[im_mask_b1] = np.nan
        im_snr_b1[im_mask_b1] = np.nan

        
    #Get RA/DEC for transient candidate
    ra_tot = catdat['RA_WideStack'][ind]
    dec_tot = catdat['DEC_WideStack'][ind]

    coord_tot = SkyCoord(ra_tot, dec_tot, unit=(u.deg, u.deg))
    dx = dx_arcsec*u.arcsec
    dy = dy_arcsec*u.arcsec



    #Get cutouts of all images
    if b1_exist:
        cutout_b1_r = Cutout2D(im_b1_r, coord_tot, size=(dx, dy), wcs=wcs_b1_r)
        cutout_b1_s = Cutout2D(im_b1_s, coord_tot, size=(dx, dy), wcs=wcs_b1_s)
        cutout_b1_subdiff = Cutout2D(im_diff_b1_sub, coord_tot, size=(dx, dy), wcs=wcs_diff_b1_sub)
        cutout_b1_sfftdiff = Cutout2D(im_diff_b1_sfft, coord_tot, size=(dx, dy), wcs=wcs_diff_b1_sfft)
        # cutout_b1_mask = Cutout2D(im_mask_b1, coord_tot, size=(dx, dy), wcs=wcs_b1_r)    
        cutout_b1_snr = Cutout2D(im_snr_b1, coord_tot, size=(dx, dy), wcs=wcs_snr_b1)
            
    if b1_exist:
        if np.all(cutout_b1_r.data == 0.) or np.all(cutout_b1_s.data == 0.) or np.all(cutout_b1_subdiff.data == 0.) or np.all(np.isnan(cutout_b1_r.data)) or np.all(np.isnan(cutout_b1_s.data)) or np.all(np.isnan(cutout_b1_subdiff.data)):
            b1_exist = False

    
    #Get normalization for each image
    if b1_exist:
        norm_b1 = ImageNormalize(cutout_b1_r.data, interval=ZScaleInterval())#, stretch=AsinhStretch())

    #Get XY coordinates of the object in each image
    if b1_exist:
        x_tot_b1, y_tot_b1 = cutout_b1_r.wcs.all_world2pix(ra_tot, dec_tot, 0)
    

    #Get fluxes and errors
    fr = nexstack_to_ujy(catdat['FLUX_APER_WideStack'].data[ind, flux_ind_nex])
    ferrr = nexstack_to_ujy(catdat['FLUXERR_APER_WideStack'].data[ind, flux_ind_nex])
    mag_b1_r, magerr_b1_r = flux_to_mag( fr, ferrr )

    fs = nexstack_to_ujy(catdat['FLUX_APER_DeepStack'].data[ind, flux_ind_nex])
    ferrs = nexstack_to_ujy(catdat['FLUXERR_APER_DeepStack'].data[ind, flux_ind_nex])
    mag_b1_s, magerr_b1_s = flux_to_mag( fs, ferrs )
            
            
    #Get mags of SFFT difference
    if b1_exist:
        f = catdat['FLUX_APER_DiffStackSFFT'].data[ind, flux_ind_diff]
        ferr = catdat['FLUXERR_APER_DiffStackSFFT'].data[ind, flux_ind_diff]
        mag_b1_sfftdiff, magerr_b1_sfftdiff = flux_to_mag(f, ferr)
        dm_b1_sfftdiff = -2.5*np.log10( 1. + f/fr )
        
        
        
        f = catdat['FLUX_APER_DiffStackSub'].data[ind, flux_ind_diff]
        ferr = catdat['FLUXERR_APER_DiffStackSub'].data[ind, flux_ind_diff]
        mag_b1_subdiff, magerr_b1_subdiff = flux_to_mag(f, ferr)
        dm_b1_subdiff = -2.5*np.log10( 1. + f/fr )        
        

    else:
        mag_b1_sfftdiff = np.nan
        magerr_b1_sfftdiff = np.nan
        dm_b1_sfftdiff = np.nan     
        
        mag_b1_subdiff = np.nan
        magerr_b1_subdiff = np.nan
        dm_b1_subdiff = np.nan  
        
    
    assert np.abs(dm_b1_sfftdiff) > .25     
        
    
    ################################################
    colors = ['c', 'm']
    
    nrow = 1
    ncol = 5
    
    if ax_in is None:
        fig = plt.figure(figsize=(9,6)) 

        ax = np.atleast_2d( np.zeros((nrow, ncol), dtype=object) )
        for i in range(nrow):
            for j in range(ncol):
                ax[i, j] = fig.add_subplot(nrow, ncol, i*ncol + j + 1)

    else:
        ax = ax_in

    if b1_exist:
        ax[0,0].imshow(cutout_b1_r.data, origin='lower', cmap='Greys_r', norm=norm_b1, interpolation='none')
        ax[0,1].imshow(cutout_b1_s.data, origin='lower', cmap='Greys_r', norm=norm_b1, interpolation='none')
        ax[0,2].imshow(cutout_b1_subdiff.data, origin='lower', cmap='Greys_r', norm=norm_b1, interpolation='none')
        ax[0,3].imshow(cutout_b1_sfftdiff.data, origin='lower', cmap='Greys_r', norm=norm_b1, interpolation='none')
        ax[0,4].imshow(cutout_b1_snr.data, origin='lower', cmap='coolwarm', interpolation='none', vmin=-5, vmax=5)
        

    txt = ax[0,0].text(.05, .05, ind, color='y', fontsize=14, transform=ax[0,0].transAxes, ha='left', va='bottom', fontweight='bold')
    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])

    if show_mag:
        txt = ax[0,0].text(.05, .95, '{:.2f} $\pm$ {:.2f}'.format(mag_b1_r, magerr_b1_r), color='c', fontsize=10, transform=ax[0,0].transAxes, ha='left', va='top', fontweight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
        txt = ax[0,1].text(.05, .95, '{:.2f} $\pm$ {:.2f}'.format(mag_b1_s, magerr_b1_s), color='c', fontsize=10, transform=ax[0,1].transAxes, ha='left', va='top', fontweight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
        
        txt = ax[0,2].text(.05, .95, '{:.2f} $\pm$ {:.2f}'.format(mag_b1_subdiff, magerr_b1_subdiff), color='c', fontsize=10, transform=ax[0,2].transAxes, ha='left', va='top', fontweight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
        txt = ax[0,3].text(.05, .95, '{:.2f} $\pm$ {:.2f}'.format(mag_b1_sfftdiff, magerr_b1_sfftdiff), color='c', fontsize=10, transform=ax[0,3].transAxes, ha='left', va='top', fontweight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])

        
        #DiffMag
        txt = ax[0,2].text(.95, .05, '$\Delta$m: {:.2f}'.format(dm_b1_subdiff), color='r', fontsize=10, transform=ax[0,2].transAxes, ha='right', va='bottom', fontweight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
        txt = ax[0,3].text(.95, .05, '$\Delta$m: {:.2f}'.format(dm_b1_sfftdiff), color='r', fontsize=10, transform=ax[0,3].transAxes, ha='right', va='bottom', fontweight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])

        # #SNR
        # if difftype == 'sfft':
        #     txt = ax[0,4].text(.95, .05, 'SNR: {:.2f}'.format(snr_b1), color='r', fontsize=10, transform=ax[0,4].transAxes, ha='right', va='bottom', fontweight='bold')
        #     txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
        #     txt = ax[1,4].text(.95, .05, 'SNR: {:.2f}'.format(snr_b2), color='r', fontsize=10, transform=ax[1,4].transAxes, ha='right', va='bottom', fontweight='bold')
        #     txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
            
        #     txt = ax[0,4].text(.05, .95, 'Multi-SNR: {:.2f}'.format(multisnr_b1), color='r', fontsize=10, transform=ax[0,4].transAxes, ha='left', va='top', fontweight='bold')
        #     txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
        #     txt = ax[1,4].text(.05, .95, 'Multi-SNR: {:.2f}'.format(multisnr_b2), color='r', fontsize=10, transform=ax[1,4].transAxes, ha='left', va='top', fontweight='bold')
        #     txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])


    #F200W
    if b1_exist:
        for i in range(ncol):
            if show_position:
                ax[0,i].scatter(x_tot_b1, y_tot_b1, marker='x', s=50, color='red')    
            
            if show_aperture:
                r = aper/ps
                circ = Circle((x_tot_b1, y_tot_b1), r, color='red', fill=False, lw=1)
                ax[0,i].add_patch(circ)
            

    if show_scalebar:
        if b1_exist:
            x2 = .98
            x1 = (x2*cutout_b1_s.data.shape[1] - 1/ps) / cutout_b1_s.data.shape[1]
            y = .1*cutout_b1_s.data.shape[0] 
            line = ax[0,1].hlines(y=y, xmin=x1, xmax=x2, transform=ax[0,1].get_yaxis_transform(), color='y', lw=2)
            line.set_path_effects([PathEffects.withStroke(linewidth=6, foreground='k')])
            txt = ax[0,1].text((x1+x2)/2, (y/cutout_b1_s.data.shape[0] + .05)*cutout_b1_s.data.shape[0], '1"', color='y', fontsize=11, va='center', ha='left', transform=ax[0,1].get_yaxis_transform(), fontweight='bold')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])



    for i in range(nrow):
        for j in range(ncol):
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

    if b1_exist:
        for i in range(ncol):
            ax[0,i].set_xlim(0, cutout_b1_r.data.shape[1])
            ax[0,i].set_ylim(0, cutout_b1_r.data.shape[0])


    if show_title:
        ax[0,0].set_title('Wide Epoch 1', fontsize=15)
        ax[0,1].set_title('Deep Epoch 1', fontsize=15)
        ax[0,2].set_title('Direct Sub DIFF', fontsize=15)
        ax[0,3].set_title('SFFT DIFF Decorr', fontsize=15)
        ax[0,4].set_title('SFFT DIFF Decorr SNR', fontsize=15)

    ax[0,0].set_ylabel(band, fontsize=18, color='c', fontweight='bold')
    plt.subplots_adjust(hspace=.01, wspace=.01)

    if show:
        plt.show()
    
    if output_fname is not None:
        plt.savefig(output_fname, bbox_inches='tight', dpi=300)
    
    return