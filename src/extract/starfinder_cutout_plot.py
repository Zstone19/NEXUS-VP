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

################################################################################

def single_object_plot(ind, imdir, catdat, fname_nexus_cat, apcorr_dir, bands, 
                       r1_band1=1., r1_band2=2., dx_arcsec=3., dy_arcsec=3., 
                       subtract_sky=True, catcoord=False, difftype='sub',
                       show_position=True, show_mag=True, show_aperture=True, show_scalebar=True,
                       title_r=None, title_s=None,
                       show=False, output_fname=None, ax_in=None):
    
    if bands[0] in ['F090W', 'F115W', 'F150W', 'F200W', 'F277W']:
        ps1 = .03
    else:
        ps1 = .06
        
    if bands[1] in ['F090W', 'F115W', 'F150W', 'F200W', 'F277W']:
        ps2 = .03
    else:
        ps2 = .06
        
    if subtract_sky:
        suffix = '_bkgsub'
    else:
        suffix = ''
        
    if catcoord:
        suffix += '_catcoord'
    

    r2_band1 = 2*r1_band1
    r2_band2 = 2*r1_band2

    r1_vals = np.array([.07, .1, .15, .2, .3, .5])
    r2_vals = 2*r1_vals
        
    flux_ind1 = np.argwhere(r1_band1 == r1_vals).flatten()[0]
    flux_ind2 = np.argwhere(r1_band2 == r1_vals).flatten()[0]
        
    f_b1_r = catdat['Flux_{}_REF{}'.format(bands[0], suffix)].data[ind, flux_ind1]
    f_b1_s = catdat['Flux_{}_SCI{}'.format(bands[0], suffix)].data[ind, flux_ind1]
    f_b2_r = catdat['Flux_{}_REF{}'.format(bands[1], suffix)].data[ind, flux_ind2]
    f_b2_s = catdat['Flux_{}_SCI{}'.format(bands[1], suffix)].data[ind, flux_ind2]
    if np.isnan(f_b1_r) or np.isnan(f_b1_s):
        b1_exist = False
    else:
        b1_exist = True
    if np.isnan(f_b2_r) or np.isnan(f_b2_s):
        b2_exist = False
    else:
        b2_exist = True
    
    
    #Get catalogs
    fname_apcorr1 = apcorr_dir + 'aperture_corrections_r{:.2f}.txt'.format(r1_band1)
    fname_apcorr2 = apcorr_dir + 'aperture_corrections_r{:.2f}.txt'.format(r1_band2)
    nexus_catdat = Table.read(fname_nexus_cat)


    if ind >= len(catdat):
        return

    if difftype == 'sfft':
        ncutout1 = catdat['Cutout_F200W'][ind]
        ncutout2 = catdat['Cutout_F444W'][ind]  
    
    imdir_full_b1 = imdir + '{}/'.format(bands[0])
    imdir_full_b2 = imdir + '{}/'.format(bands[1])
    if difftype == 'sfft':
        imdir_full_b1 += 'output_{}/'.format(ncutout1)
        imdir_full_b2 += 'output_{}/'.format(ncutout2)
        
    if ncutout1 == -1:
        b1_exist = False
    if ncutout2 == -1:
        b2_exist = False
    
    #Get image filenames
    if difftype == 'sub':
        fname_b1_r = imdir_full_b1 + 'input/nexus_wide01_{}.fits'.format(bands[0], bands[0])
        fname_b1_s = imdir_full_b1 + 'input/nexus_deep01_{}.fits'.format(bands[0], bands[0])
        fname_b2_r = imdir_full_b2 + 'input/nexus_wide01_{}.fits'.format(bands[1], bands[1])
        fname_b2_s = imdir_full_b2 + 'input/nexus_deep01_{}.fits'.format(bands[1], bands[1])
        
        fname_diff_b1 = imdir_full_b1 + 'output/nexus_deep01_{}.subdiff.fits'.format(bands[0], bands[0])
        fname_diff_b2 = imdir_full_b2 + 'output/nexus_deep01_{}.subdiff.fits'.format(bands[1], bands[1])
        
    elif difftype == 'sfft':
        fname_b1_r_og = imdir_full_b1 + 'input/nexus_wide01_{}.fits'.format(bands[0], bands[0])
        fname_b1_s_og = imdir_full_b1 + 'input/nexus_deep01_{}.fits'.format(bands[0], bands[0])
        fname_b2_r_og = imdir_full_b2 + 'input/nexus_wide01_{}.fits'.format(bands[1], bands[1])
        fname_b2_s_og = imdir_full_b2 + 'input/nexus_deep01_{}.fits'.format(bands[1], bands[1])
        
        fname_b1_r = imdir_full_b1 + 'input/nexus_wide01_{}.skysub.fits'.format(bands[0], bands[0])
        fname_b1_s = imdir_full_b1 + 'input/nexus_deep01_{}.skysub.fits'.format(bands[0], bands[0])
        fname_b2_r = imdir_full_b2 + 'input/nexus_wide01_{}.skysub.fits'.format(bands[1], bands[1])
        fname_b2_s = imdir_full_b2 + 'input/nexus_deep01_{}.skysub.fits'.format(bands[1], bands[1])
        
        fname_diff_b1 = imdir_full_b1 + 'output/nexus_deep01_{}.sfftdiff.fits'.format(bands[0], bands[0])
        fname_diff_b2 = imdir_full_b2 + 'output/nexus_deep01_{}.sfftdiff.fits'.format(bands[1], bands[1])
        
        fname2_diff_b1 = imdir_full_b1 + 'output/nexus_deep01_{}.sfftdiff.decorr.fits'.format(bands[0], bands[0])
        fname2_diff_b2 = imdir_full_b2 + 'output/nexus_deep01_{}.sfftdiff.decorr.fits'.format(bands[1], bands[1])
        
        fname_snr_b1 = imdir_full_b1 + 'output/nexus_deep01_{}.sfftdiff.decorr.snr.fits'.format(bands[0], bands[0])
        fname_snr_b2 = imdir_full_b2 + 'output/nexus_deep01_{}.sfftdiff.decorr.snr.fits'.format(bands[1], bands[1])
        
        fname_noise_b1_r = imdir_full_b1 + 'noise/nexus_wide01_{}.noise.fits'.format(bands[0], bands[0])
        fname_noise_b1_s = imdir_full_b1 + 'noise/nexus_deep01_{}.noise.fits'.format(bands[0], bands[0])
        fname_noise_b2_r = imdir_full_b2 + 'noise/nexus_wide01_{}.noise.fits'.format(bands[1], bands[1])
        fname_noise_b2_s = imdir_full_b2 + 'noise/nexus_deep01_{}.noise.fits'.format(bands[1], bands[1])
    
    
    #Load images
    with warnings.catch_warnings(action="ignore"):
        
        if b1_exist:
            with fits.open(fname_b1_r_og) as hdul:
                im_b1_r_og = hdul[0].data
            with fits.open(fname_b1_s_og) as hdul:
                im_b1_s_og = hdul[0].data
                
            with fits.open(fname_b1_r) as hdul:
                im_b1_r = hdul[0].data
                wcs_b1_r = WCS(hdul[0].header)
            with fits.open(fname_b1_s) as hdul:
                im_b1_s = hdul[0].data
                wcs_b1_s = WCS(hdul[0].header)
                
            with fits.open(fname_noise_b1_r) as hdul:
                im_b1_r_noise = hdul[0].data
                wcs_b1_r_noise = WCS(hdul[0].header)
            with fits.open(fname_noise_b1_s) as hdul:
                im_b1_s_noise = hdul[0].data
                wcs_b1_s_noise = WCS(hdul[0].header)
            
            
                
            with fits.open(fname_diff_b1) as hdul:
                im_diff_b1 = hdul[0].data
                wcs_diff_b1 = WCS(hdul[0].header)
                
            im_mask_b1 = (im_b1_r_og == 0.) | (im_b1_s_og == 0.) | np.isnan(im_b1_r_og) | np.isnan(im_b1_s_og)
            
            
            if difftype == 'sfft':
                with fits.open(fname2_diff_b1) as hdul:
                    im2_diff_b1 = hdul[0].data
                    wcs2_diff_b1 = WCS(hdul[0].header)                    
                with fits.open(fname_snr_b1) as hdul:
                    im_snr_b1 = hdul[0].data
                    wcs_snr_b1 = WCS(hdul[0].header)


        if b2_exist:
            with fits.open(fname_b2_r_og) as hdul:
                im_b2_r_og = hdul[0].data
            with fits.open(fname_b2_s_og) as hdul:
                im_b2_s_og = hdul[0].data        
            
            with fits.open(fname_b2_r) as hdul:
                im_b2_r = hdul[0].data
                wcs_b2_r = WCS(hdul[0].header)
            with fits.open(fname_b2_s) as hdul:
                im_b2_s = hdul[0].data
                wcs_b2_s = WCS(hdul[0].header)
                
            with fits.open(fname_noise_b2_r) as hdul:
                im_b2_r_noise = hdul[0].data
                wcs_b2_r_noise = WCS(hdul[0].header)
            with fits.open(fname_noise_b2_s) as hdul:
                im_b2_s_noise = hdul[0].data
                wcs_b2_s_noise = WCS(hdul[0].header)
                
            with fits.open(fname_diff_b2) as hdul:
                im_diff_b2 = hdul[0].data
                wcs_diff_b2 = WCS(hdul[0].header)
            
            im_mask_b2 = (im_b2_r_og == 0.) | (im_b2_s_og == 0.) | np.isnan(im_b2_r_og) | np.isnan(im_b2_s_og)
            
            if difftype == 'sfft':
                with fits.open(fname2_diff_b2) as hdul:
                    im2_diff_b2 = hdul[0].data
                    wcs2_diff_b2 = WCS(hdul[0].header)
                with fits.open(fname_snr_b2) as hdul:
                    im_snr_b2 = hdul[0].data
                    wcs_snr_b2 = WCS(hdul[0].header)
       
    #Mask images
    if b1_exist:
        im_b1_r[im_mask_b1] = np.nan
        im_b1_s[im_mask_b1] = np.nan
        im_diff_b1[im_mask_b1] = np.nan
        if difftype == 'sfft':
            im2_diff_b1[im_mask_b1] = np.nan
            im_snr_b1[im_mask_b1] = np.nan
    
    if b2_exist:
        im_b2_r[im_mask_b2] = np.nan
        im_b2_s[im_mask_b2] = np.nan
        im_diff_b2[im_mask_b2] = np.nan
        if difftype == 'sfft':
            im2_diff_b2[im_mask_b2] = np.nan
            im_snr_b2[im_mask_b2] = np.nan

        
        
    #Get coordinates of NEXUS catalog objects
    coord_nexus = SkyCoord(nexus_catdat['RA'].data, nexus_catdat['DEC'].data, unit=(u.deg, u.deg))

    #Get RA/DEC for transient candidate
    if catcoord:
        ra_tot = catdat['NEXUS_RA'][ind]
        dec_tot = catdat['NEXUS_DEC'][ind]
    else:
        ra_tot = catdat['RA'][ind]
        dec_tot = catdat['DEC'][ind]

    coord_tot = SkyCoord(ra_tot, dec_tot, unit=(u.deg, u.deg))
    dx = dx_arcsec*u.arcsec
    dy = dy_arcsec*u.arcsec



    #Get cutouts of all images
    if b1_exist:
        cutout_b1_r = Cutout2D(im_b1_r, coord_tot, size=(dx, dy), wcs=wcs_b1_r)
        cutout_b1_s = Cutout2D(im_b1_s, coord_tot, size=(dx, dy), wcs=wcs_b1_s)
        cutout_b1_r_noise = Cutout2D(im_b1_r_noise, coord_tot, size=(dx, dy), wcs=wcs_b1_r_noise)
        cutout_b1_s_noise = Cutout2D(im_b1_s_noise, coord_tot, size=(dx, dy), wcs=wcs_b1_s_noise)
        cutout_b1_diff = Cutout2D(im_diff_b1, coord_tot, size=(dx, dy), wcs=wcs_diff_b1)
        cutout_b1_mask = Cutout2D(im_mask_b1, coord_tot, size=(dx, dy), wcs=wcs_b1_r)

    if b2_exist:
        cutout_b2_r = Cutout2D(im_b2_r, coord_tot, size=(dx, dy), wcs=wcs_b2_r)
        cutout_b2_s = Cutout2D(im_b2_s, coord_tot, size=(dx, dy), wcs=wcs_b2_s)
        cutout_b2_r_noise = Cutout2D(im_b2_r_noise, coord_tot, size=(dx, dy), wcs=wcs_b2_r_noise)
        cutout_b2_s_noise = Cutout2D(im_b2_s_noise, coord_tot, size=(dx, dy), wcs=wcs_b2_s_noise)
        cutout_b2_diff = Cutout2D(im_diff_b2, coord_tot, size=(dx, dy), wcs=wcs_diff_b2)
        cutout_b2_mask = Cutout2D(im_mask_b2, coord_tot, size=(dx, dy), wcs=wcs_b2_r)
        
    
    if difftype == 'sfft':
        if b1_exist:
            cutout_b1_diff2 = Cutout2D(im2_diff_b1, coord_tot, size=(dx, dy), wcs=wcs2_diff_b1)
            cutout_b1_snr = Cutout2D(im_snr_b1, coord_tot, size=(dx, dy), wcs=wcs_snr_b1)
        if b2_exist:
            cutout_b2_diff2 = Cutout2D(im2_diff_b2, coord_tot, size=(dx, dy), wcs=wcs2_diff_b2)
            cutout_b2_snr = Cutout2D(im_snr_b2, coord_tot, size=(dx, dy), wcs=wcs_snr_b2)
            
    if b1_exist:
        if np.all(cutout_b1_r.data == 0.) or np.all(cutout_b1_s.data == 0.) or np.all(cutout_b1_diff.data == 0.) or np.all(np.isnan(cutout_b1_r.data)) or np.all(np.isnan(cutout_b1_s.data)) or np.all(np.isnan(cutout_b1_diff.data)):
            b1_exist = False

    if b2_exist:
        if np.all(cutout_b2_r.data == 0.) or np.all(cutout_b2_s.data == 0.) or np.all(cutout_b2_diff.data == 0.) or np.all(np.isnan(cutout_b2_r.data)) or np.all(np.isnan(cutout_b2_s.data)) or np.all(np.isnan(cutout_b2_diff.data)):
            b2_exist = False

    
    #Get normalization for each image
    if b1_exist:
        norm_b1 = ImageNormalize(cutout_b1_r.data, interval=ZScaleInterval())#, stretch=AsinhStretch())
        norm_b1_diff = ImageNormalize(cutout_b1_diff.data, interval=ZScaleInterval())#, stretch=AsinhStretch())
    if b2_exist:
        norm_b2 = ImageNormalize(cutout_b2_r.data, interval=ZScaleInterval())#, stretch=AsinhStretch())
        norm_b2_diff = ImageNormalize(cutout_b2_diff.data, interval=ZScaleInterval())#, stretch=AsinhStretch())
    

    #Get XY coordinates of the object in each image
    if b1_exist:
        x_tot_b1, y_tot_b1 = cutout_b1_r.wcs.all_world2pix(ra_tot, dec_tot, 0)
        x_nexus_b1, y_nexus_b1 = cutout_b1_r.wcs.all_world2pix(coord_nexus.ra.deg, coord_nexus.dec.deg, 0)
    if b2_exist:
        x_tot_b2, y_tot_b2 = cutout_b2_r.wcs.all_world2pix(ra_tot, dec_tot, 0)
        x_nexus_b2, y_nexus_b2 = cutout_b2_r.wcs.all_world2pix(coord_nexus.ra.deg, coord_nexus.dec.deg, 0)

    #Get fluxes and errors
    if catcoord:
        mag_b1_r, magerr_b1_r = flux_to_mag( catdat['Flux_{}_REF_bkgsub_catcoord'.format(bands[0], suffix)][ind, flux_ind1], catdat['FluxErr_{}_REF_bkgsub_catcoord'.format(bands[0], suffix)][ind, flux_ind1] )
        mag_b1_s, magerr_b1_s = flux_to_mag( catdat['Flux_{}_SCI_bkgsub_catcoord'.format(bands[0], suffix)][ind, flux_ind1], catdat['FluxErr_{}_SCI_bkgsub_catcoord'.format(bands[0], suffix)][ind, flux_ind1] )
        mag_b2_r, magerr_b2_r = flux_to_mag( catdat['Flux_{}_REF_bkgsub_catcoord'.format(bands[1], suffix)][ind, flux_ind2], catdat['FluxErr_{}_REF_bkgsub_catcoord'.format(bands[1], suffix)][ind, flux_ind2] )
    else:
        mag_b1_r, magerr_b1_r = flux_to_mag( catdat['Flux_{}_REF_bkgsub'.format(bands[0], suffix)][ind, flux_ind1], catdat['FluxErr_{}_REF_bkgsub_catcoord'.format(bands[0], suffix)][ind, flux_ind1] )
        mag_b1_s, magerr_b1_s = flux_to_mag( catdat['Flux_{}_SCI_bkgsub'.format(bands[0], suffix)][ind, flux_ind1], catdat['FluxErr_{}_SCI_bkgsub_catcoord'.format(bands[0], suffix)][ind, flux_ind1] )
        mag_b2_r, magerr_b2_r = flux_to_mag( catdat['Flux_{}_REF_bkgsub'.format(bands[1], suffix)][ind, flux_ind2], catdat['FluxErr_{}_REF_bkgsub_catcoord'.format(bands[1], suffix)][ind, flux_ind2] )
    
    # if difftype == 'sfft':
    #     if catcoord:
    #         snr_b1 = catdat['SFFT_SNR_{}_catcoord'.format(bands[0])][ind, flux_ind]
    #     else:
    #         snr_b1 = catdat['SFFT_SNR_{}'.format(bands[0])][ind, flux_ind]
            
    #     multisnr_b1 = catdat['MultiDiff_SNR_{}{}'.format(bands[0], suffix)][ind, flux_ind]


    mag_b1_r, magerr_b1_r = flux_to_mag( catdat['Flux_{}_REF{}'.format(bands[0], suffix)][ind, flux_ind1], catdat['FluxErr_{}_REF{}'.format(bands[0], suffix)][ind, flux_ind1] )
    mag_b2_s, magerr_b2_s = flux_to_mag( catdat['Flux_{}_SCI{}'.format(bands[1], suffix)][ind, flux_ind2], catdat['FluxErr_{}_SCI{}'.format(bands[1], suffix)][ind, flux_ind2] )    
    # if difftype == 'sfft':
    #     if catcoord:
    #         snr_b2 = catdat['SFFT_SNR_{}_catcoord'.format(bands[1])][ind, flux_ind]
    #     else:
    #         snr_b2 = catdat['SFFT_SNR_{}'.format(bands[1])][ind, flux_ind]
            
    #     multisnr_b2 = catdat['MultiDiff_SNR_{}{}'.format(bands[1], suffix)][ind, flux_ind]
            
            
    #Get mags of SFFT difference
    if difftype == 'sfft':
        if b1_exist:
            f1, ferr1 = get_fluxes(cutout_b1_diff.data, (cutout_b1_diff.data/cutout_b1_snr.data)**2, 
                                cutout_b1_mask.data, [x_tot_b1], [y_tot_b1], fname_apcorr1, bands[0], 
                                ps_grid=ps1, neg=False, r1=r1_band1, r2=r2_band1, subtract_sky=subtract_sky)
            
            mag_b1_diff, magerr_b1_diff = flux_to_mag(f1[0], ferr1[0])
            dm_b1 = -2.5*np.log10( 1 + np.abs(f1[0])/catdat['Flux_{}_REF{}'.format(bands[0], suffix)][ind, flux_ind1] )
        else:
            mag_b1_diff = np.nan
            magerr_b1_diff = np.nan
            dm_b1 = np.nan


        if b2_exist:
            f2, ferr2 = get_fluxes(cutout_b2_diff.data, (cutout_b2_diff.data/cutout_b2_snr.data)**2,
                                cutout_b2_mask.data, [x_tot_b2], [y_tot_b2], fname_apcorr2, bands[1], 
                                ps_grid=ps2, neg=False, r1=r1_band2, r2=r2_band2, subtract_sky=subtract_sky)
            
            mag_b2_diff, magerr_b2_diff = flux_to_mag(f2[0], ferr2[0])
            dm_b2 = -2.5*np.log10( 1 + np.abs(f2[0])/catdat['Flux_{}_REF{}'.format(bands[1], suffix)][ind, flux_ind2] )
        else:
            mag_b2_diff = np.nan
            magerr_b2_diff = np.nan
            dm_b2 = np.nan
            
        
        
        
    #Get mags of raw difference
    if b1_exist:
        f1, ferr1 = get_fluxes(cutout_b1_s.data - cutout_b1_r.data, cutout_b1_r_noise.data**2 + cutout_b1_s_noise.data**2, 
                            cutout_b1_mask.data, [x_tot_b1], [y_tot_b1], fname_apcorr1, bands[0], 
                            ps_grid=ps1, neg=False, r1=r1_band1, r2=r2_band1, subtract_sky=subtract_sky)
        
        mag_b1_rawdiff, magerr_b1_rawdiff = flux_to_mag(f1[0], ferr1[0])
        dm_b1_rawdiff = -2.5*np.log10( 1 + np.abs(f1[0])/catdat['Flux_{}_REF{}'.format(bands[0], suffix)][ind, flux_ind1] )
    else:
        mag_b1_rawdiff = np.nan
        magerr_b1_rawdiff = np.nan
        dm_b1_rawdiff = np.nan


    if b2_exist:
        f2, ferr2 = get_fluxes(cutout_b2_s.data - cutout_b2_r.data, cutout_b2_r_noise.data**2 + cutout_b2_s_noise.data**2, 
                            cutout_b2_mask.data, [x_tot_b2], [y_tot_b2], fname_apcorr2, bands[1], 
                            ps_grid=ps2, neg=False, r1=r1_band2, r2=r2_band2, subtract_sky=subtract_sky)
        
        mag_b2_rawdiff, magerr_b2_rawdiff = flux_to_mag(f2[0], ferr2[0])
        dm_b2_rawdiff = -2.5*np.log10( 1 + np.abs(f2[0])/catdat['Flux_{}_REF{}'.format(bands[1], suffix)][ind, flux_ind2] )
    else:
        mag_b2_rawdiff = np.nan
        magerr_b2_rawdiff = np.nan
        dm_b2_rawdiff = np.nan
    
    ################################################
    colors = ['c', 'm']
    
    if difftype == 'sfft':
        nrow = 2
        ncol = 5
    else:
        nrow = 2
        ncol = 3
    
    if ax_in is None:
        fig = plt.figure(figsize=(9,6)) 

        ax = np.zeros((nrow, ncol), dtype=object)
        for i in range(nrow):
            for j in range(ncol):
                ax[i, j] = fig.add_subplot(nrow, ncol, i*ncol + j + 1)

    else:
        ax = ax_in

    if b1_exist:
        ax[0,0].imshow(cutout_b1_r.data, origin='lower', cmap='Greys_r', norm=norm_b1, interpolation='none')
        ax[0,1].imshow(cutout_b1_s.data, origin='lower', cmap='Greys_r', norm=norm_b1, interpolation='none')
        ax[0,2].imshow(cutout_b1_s.data - cutout_b1_r.data, origin='lower', cmap='Greys_r', norm=norm_b1, interpolation='none')
        if difftype == 'sfft':
            ax[0,3].imshow(cutout_b1_diff2.data, origin='lower', cmap='Greys_r', norm=norm_b1, interpolation='none')
            ax[0,4].imshow(cutout_b1_snr.data, origin='lower', cmap='coolwarm', interpolation='none', vmin=-5, vmax=5)
        

    if b2_exist:
        ax[1,0].imshow(cutout_b2_r.data, origin='lower', cmap='Greys_r', norm=norm_b2, interpolation='none')
        ax[1,1].imshow(cutout_b2_s.data, origin='lower', cmap='Greys_r', norm=norm_b2, interpolation='none')
        ax[1,2].imshow(cutout_b2_s.data - cutout_b2_r.data, origin='lower', cmap='Greys_r', norm=norm_b2, interpolation='none')
        if difftype == 'sfft':
            ax[1,3].imshow(cutout_b2_diff2.data, origin='lower', cmap='Greys_r', norm=norm_b2, interpolation='none')
            ax[1,4].imshow(cutout_b2_snr.data, origin='lower', cmap='coolwarm', interpolation='none', vmin=-5, vmax=5)


    txt = ax[1,0].text(.05, .05, catdat['Index'][ind], color='y', fontsize=14, transform=ax[1,0].transAxes, ha='left', va='bottom', fontweight='bold')
    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])

    if show_mag:
        txt = ax[0,0].text(.05, .95, '{:.2f} $\pm$ {:.2f}'.format(mag_b1_r, magerr_b1_r), color='c', fontsize=10, transform=ax[0,0].transAxes, ha='left', va='top', fontweight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
        txt = ax[0,1].text(.05, .95, '{:.2f} $\pm$ {:.2f}'.format(mag_b1_s, magerr_b1_s), color='c', fontsize=10, transform=ax[0,1].transAxes, ha='left', va='top', fontweight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
        
        if (difftype == 'sfft'):
            txt = ax[0,2].text(.05, .95, '{:.2f} $\pm$ {:.2f}'.format(mag_b1_rawdiff, magerr_b1_rawdiff), color='c', fontsize=10, transform=ax[0,2].transAxes, ha='left', va='top', fontweight='bold')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
            
            txt = ax[0,3].text(.05, .95, '{:.2f} $\pm$ {:.2f}'.format(mag_b1_diff, magerr_b1_diff), color='c', fontsize=10, transform=ax[0,3].transAxes, ha='left', va='top', fontweight='bold')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
        else:
            txt = ax[0,2].text(.05, .95, '{:.2f} $\pm$ {:.2f}'.format(mag_b1_diff, magerr_b1_diff), color='c', fontsize=10, transform=ax[0,2].transAxes, ha='left', va='top', fontweight='bold')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])

        txt = ax[1,0].text(.05, .95, '{:.2f} $\pm$ {:.2f}'.format(mag_b2_r, magerr_b2_r), color='m', fontsize=10, transform=ax[1,0].transAxes, ha='left', va='top', fontweight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
        txt = ax[1,1].text(.05, .95, '{:.2f} $\pm$ {:.2f}'.format(mag_b2_s, magerr_b2_s), color='m', fontsize=10, transform=ax[1,1].transAxes, ha='left', va='top', fontweight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
        
        if (difftype == 'sfft'):
            txt = ax[1,2].text(.05, .95, '{:.2f} $\pm$ {:.2f}'.format(mag_b2_rawdiff, magerr_b2_rawdiff), color='m', fontsize=10, transform=ax[1,2].transAxes, ha='left', va='top', fontweight='bold')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])

            txt = ax[1,3].text(.05, .95, '{:.2f} $\pm$ {:.2f}'.format(mag_b2_diff, magerr_b2_diff), color='m', fontsize=10, transform=ax[1,3].transAxes, ha='left', va='top', fontweight='bold')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
        else:
            txt = ax[1,2].text(.05, .95, '{:.2f} $\pm$ {:.2f}'.format(mag_b2_diff, magerr_b2_diff), color='m', fontsize=10, transform=ax[1,2].transAxes, ha='left', va='top', fontweight='bold')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
        
        #DiffMag
        if (difftype == 'sfft'):
            txt = ax[0,2].text(.95, .05, '$\Delta$m: {:.2f}'.format(np.abs(dm_b1_rawdiff)), color='r', fontsize=10, transform=ax[0,2].transAxes, ha='right', va='bottom', fontweight='bold')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
            txt = ax[1,2].text(.95, .05, '$\Delta$m: {:.2f}'.format(np.abs(dm_b2_rawdiff)), color='r', fontsize=10, transform=ax[1,2].transAxes, ha='right', va='bottom', fontweight='bold')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
            
            txt = ax[0,3].text(.95, .05, '$\Delta$m: {:.2f}'.format(np.abs(dm_b1)), color='r', fontsize=10, transform=ax[0,3].transAxes, ha='right', va='bottom', fontweight='bold')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
            txt = ax[1,3].text(.95, .05, '$\Delta$m: {:.2f}'.format(np.abs(dm_b2)), color='r', fontsize=10, transform=ax[1,3].transAxes, ha='right', va='bottom', fontweight='bold')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
        else:
            txt = ax[0,2].text(.95, .05, '$\Delta$m: {:.2f}'.format(np.abs(dm_b1)), color='r', fontsize=10, transform=ax[0,2].transAxes, ha='right', va='bottom', fontweight='bold')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
            txt = ax[1,2].text(.95, .05, '$\Delta$m: {:.2f}'.format(np.abs(dm_b2)), color='r', fontsize=10, transform=ax[1,2].transAxes, ha='right', va='bottom', fontweight='bold')
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
            
            # if i<2:
            #     ax[0,i].scatter(x_nexus_b1, y_nexus_b1, marker='o', s=50, color='gold', edgecolor='black', linewidth=1.5)
            
            if show_aperture:
                r = r1_band1/ps1
                circ = Circle((x_tot_b1, y_tot_b1), r, color='red', fill=False, lw=1)
                ax[0,i].add_patch(circ)
                
                if (i < 2) or (subtract_sky and (i in [2,3]) ):
                    r = r2_band1/ps1
                    circ = Circle((x_tot_b1, y_tot_b1), r, color='r', fill=False, ls='--')
                    ax[0,i].add_patch(circ)

    #F444W
    if b2_exist:
        for i in range(ncol):    
            if show_position:
                ax[1,i].scatter(x_tot_b2, y_tot_b2, marker='x', s=50, color='red')    

            # if i<2:
            #     ax[1,i].scatter(x_nexus_b2, y_nexus_b2, marker='o', s=50, color='gold', edgecolor='black', linewidth=1.5)

            if show_aperture:
                r = r1_band2/ps2
                circ = Circle((x_tot_b2, y_tot_b2), r, color='red', fill=False, lw=1)
                ax[1,i].add_patch(circ)

                if (i < 2) or (subtract_sky and (i in [2,3]) ):
                    r = r2_band2/ps2
                    circ = Circle((x_tot_b2, y_tot_b2), r, color='r', fill=False, ls='--')
                    ax[1,i].add_patch(circ)
            

    if show_scalebar:
        if b1_exist:
            x2 = .98
            x1 = (x2*cutout_b1_s.data.shape[1] - 1/ps1) / cutout_b1_s.data.shape[1]
            y = .1*cutout_b1_s.data.shape[0] 
            line = ax[0,1].hlines(y=y, xmin=x1, xmax=x2, transform=ax[0,1].get_yaxis_transform(), color='y', lw=2)
            line.set_path_effects([PathEffects.withStroke(linewidth=6, foreground='k')])
            txt = ax[0,1].text((x1+x2)/2, (y/cutout_b1_s.data.shape[0] + .05)*cutout_b1_s.data.shape[0], '1"', color='y', fontsize=11, va='center', ha='left', transform=ax[0,1].get_yaxis_transform(), fontweight='bold')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])

        if b2_exist:        
            x2 = .98
            x1 = (x2*cutout_b2_s.data.shape[1] - 1/ps2) / cutout_b2_s.data.shape[1]
            y = .1*cutout_b2_s.data.shape[0] 
            line = ax[1,1].hlines(y=y, xmin=x1, xmax=x2, transform=ax[1,1].get_yaxis_transform(), color='y', lw=2)
            line.set_path_effects([PathEffects.withStroke(linewidth=6, foreground='k')])
            txt = ax[1,1].text((x1+x2)/2, (y/cutout_b2_s.data.shape[0] + .05)*cutout_b2_s.data.shape[0], '1"', color='y', fontsize=11, va='center', ha='left', transform=ax[1,1].get_yaxis_transform(), fontweight='bold')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])


    for i in range(nrow):
        for j in range(ncol):
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

    if b1_exist:
        for i in range(ncol):
            ax[0,i].set_xlim(0, cutout_b1_r.data.shape[1])
            ax[0,i].set_ylim(0, cutout_b1_r.data.shape[0])

    if b2_exist:
        for i in range(ncol):
            ax[1,i].set_xlim(0, cutout_b2_r.data.shape[1])
            ax[1,i].set_ylim(0, cutout_b2_r.data.shape[0])


    ax[0,0].set_title(title_r, fontsize=15)
    ax[0,1].set_title(title_s, fontsize=15)
    ax[0,2].set_title('Raw DIFF', fontsize=15)
    if difftype == 'sfft':
        ax[0,3].set_title('SFFT DIFF Decorr', fontsize=15)
        ax[0,4].set_title('SFFT DIFF Decorr SNR', fontsize=15)

    ax[0,0].set_ylabel(bands[0], fontsize=18, color='c', fontweight='bold')
    ax[1,0].set_ylabel(bands[1], fontsize=18, color='m', fontweight='bold')


    plt.subplots_adjust(hspace=.01, wspace=.01)

    if show:
        plt.show()
    
    if output_fname is not None:
        plt.savefig(output_fname, bbox_inches='tight', dpi=300)
    
    return