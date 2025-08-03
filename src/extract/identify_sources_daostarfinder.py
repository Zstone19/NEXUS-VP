import numpy as np
import os
import warnings
from functools import partial
import multiprocessing as mp
from p_tqdm import p_map 
from tqdm import tqdm

from astropy.table import Table, vstack
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.nddata import Cutout2D

from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats
from astropy.stats import sigma_clipped_stats

##########################################################################################################################################
##########################################################################################################################################
# Utility

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

    if fname_apcorr is not None:
        corrdat = Table.read(fname_apcorr, format='ascii.commented_header')
        ap_corr = corrdat['ApertureCorrection'][corrdat['Band'] == band]
        ann_corr = corrdat['SkyCorrection'][corrdat['Band'] == band]

        circ_corr_fluxes = circ_fluxes * ap_corr
        circ_corr_fluxerrs = circ_fluxerrs * ap_corr
        if subtract_sky:
            ann_corr_fluxes = ann_fluxes * ann_corr
            ann_corr_fluxerrs = ann_fluxerrs * ann_corr
            
    else:
        circ_corr_fluxes = circ_fluxes.copy()
        circ_corr_fluxerrs = circ_fluxerrs.copy()
        if subtract_sky:
            ann_corr_fluxes = ann_fluxes.copy()
            ann_corr_fluxerrs = ann_fluxerrs.copy()
    
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


def get_snr(im_snr, im_diff, im_mask, x, y, band, ps_grid=.03, r1=1):  
    im_var = (im_diff/im_snr)**2  #N = S / (S/N)
    
    mask_1 = np.abs(im_diff/np.sqrt(im_var)) > .1
    mask_2 = np.abs(im_diff/np.sqrt(im_var)) > .1
    mask2 = im_mask | mask_1 | mask_2
    
    
    im_s = im_snr.copy()
    im_s[im_mask] = 0.
    im_s[np.isnan(im_s)] = 0.
    im_var[im_mask] = 0.
    im_var[np.isnan(im_var)] = 0.
        
      
    circ_ap = CircularAperture(zip(x, y), r=r1/ps_grid)
    circ_npx = circ_ap.area
    
    stat1 = ApertureStats(im_snr/np.sqrt(im_var), circ_ap, mask=im_mask, sum_method='exact') #S/var
    stat2 = ApertureStats(1/im_var, circ_ap, mask=im_mask, sum_method='exact')

    fhat = stat1.sum / stat2.sum
    var_fhat = 1./ stat2.sum

    ap_snr = fhat / np.sqrt(var_fhat)
    return ap_snr


def get_multidiff_snr(im_snr, im_sfftdiff, im_mask, 
                      im_rawdiff, im_rawdiff_var,
                      x, y, band, ps, 
                      fname_apcorr, r1=1, r2=2,
                      sum_px=True, subtract_sky=True):

    # apcorr_dat = Table.read(fname_apcorr, format='ascii.commented_header')
    
    sfft_flux = im_sfftdiff.copy()
    sfft_var = (im_sfftdiff/im_snr)**2  #N = S / (S/N)
    rawdiff_flux = im_rawdiff.copy()
    rawdiff_var = im_rawdiff_var.copy()
        
    sfft_flux[im_mask] = 0.
    sfft_var[im_mask] = 0.
    rawdiff_flux[im_mask] = 0.
    rawdiff_var[im_mask] = 0.

    sfft_flux[np.isnan(sfft_flux)] = 0.
    sfft_var[np.isnan(sfft_var)] = 0.
    rawdiff_flux[np.isnan(rawdiff_flux)] = 0.
    rawdiff_var[np.isnan(rawdiff_var)] = 0.
    
    # #############################################################################################################################
    # #Mask unphysical values
    # print('Masking unphysical values')
    
    # mask_1 = np.abs(flux_1_tot/np.sqrt(var_1_tot)) > .1
    # mask_2 = np.abs(flux_2_tot/np.sqrt(var_2_tot)) > .1
    
    # ##################################################URGENTURGENTURGENT
    # ##################################################URGENTURGENTURGENT
    # #CHECK TO MAKE SURE THIS IS OK
    # mask_good = mask_1 & mask_2
    # flux_1_tot[~mask_1] = np.nan
    # flux_2_tot[~mask_2] = np.nan
    # var_1_tot[~mask_1] = np.nan
    # var_2_tot[~mask_2] = np.nan
    # ##################################################URGENTURGENTURGENT
    # ##################################################URGENTURGENTURGENT

    
    ############################################################################################################################
    #Get multi-image SNR

    if sum_px:
        circ_ap = CircularAperture(zip(x, y), r=r1/ps)
        
        im1 = sfft_flux/sfft_var
        im1[np.isnan(im1)] = 0.
        im2 = 1./sfft_var
        im2[np.isnan(im2)] = 0.
        stat1_sfft = ApertureStats(im1, circ_ap, mask=im_mask, sum_method='exact') #S/var
        stat2_sfft = ApertureStats(im2, circ_ap, mask=im_mask, sum_method='exact')
        
        im1 = rawdiff_flux/rawdiff_var
        im1[np.isnan(im1)] = 0.
        im2 = 1./rawdiff_var
        im2[np.isnan(im2)] = 0.
        stat1_raw = ApertureStats(im1, circ_ap, mask=im_mask, sum_method='exact') #S/var
        stat2_raw = ApertureStats(im2, circ_ap, mask=im_mask, sum_method='exact')
        
    
        fhat = (stat1_sfft.sum + stat1_raw.sum) / (stat2_sfft.sum + stat2_raw.sum)
        var_fhat = 1. / (stat2_sfft.sum + stat2_raw.sum)
        
    else:
        apflux_sfft, apfluxerr_sfft = get_fluxes(sfft_flux, sfft_var, im_mask, x, y, fname_apcorr, band, ps_grid=ps, neg=False, r1=r1, r2=r2, subtract_sky=subtract_sky)
        apflux_raw, apfluxerr_raw = get_fluxes(rawdiff_flux, rawdiff_var, im_mask, x, y, fname_apcorr, band, ps_grid=ps, neg=False, r1=r1, r2=r2, subtract_sky=subtract_sky)
        
        fhat = np.nansum([apflux_sfft/(apfluxerr_sfft**2), apflux_raw/(apfluxerr_raw**2)], axis=0) / np.nansum([1/(apfluxerr_sfft**2), 1/(apfluxerr_raw**2)], axis=0)
        var_fhat = 1. / np.nansum([1/(apfluxerr_sfft**2), 1/(apfluxerr_raw**2)], axis=0)
        

    snr_multi = fhat / np.sqrt(var_fhat)

    return snr_multi

##########################################################################################################################################
##########################################################################################################################################
# Get sources in individual DIFF images

def get_sources_indiv_image(fname_diff, fname_cutout,
                            fname_r, fname_s,
                            fname_mask_r, fname_mask_s, 
                            fname_psf_r, fname_psf_s,
                            band='F115W', neg=False, ps_grid=.03, matching_radius=.3, difftype='sfft'):
    
    #####################################################################
    #Get images
    
    cutout_dat = Table.read(fname_cutout, format='ascii')
    
    with warnings.catch_warnings(action="ignore"):
        if difftype == 'sfft':
            with fits.open(fname_diff) as hdul:
                im_diff_sfft = hdul[0].data
                hdr_diff = hdul[0].header
                wcs_diff = WCS(hdr_diff)
            
        with fits.open(fname_r) as hdul:
            im_diff_r = hdul[0].data
            hdr_r = hdul[0].header
            wcs_r = WCS(hdr_r)
        with fits.open(fname_s) as hdul:
            im_diff_s = hdul[0].data

        if difftype == 'sfft':
            im_diff = im_diff_sfft.copy()
        elif difftype == 'sub':
            im_diff = im_diff_r - im_diff_s
            wcs_diff = wcs_r

        if neg:
            im_diff *= -1

        with fits.open(fname_mask_r) as hdul:
            im_mask_r = hdul[0].data.astype(bool)
        with fits.open(fname_mask_s) as hdul:
            im_mask_s = hdul[0].data.astype(bool)

        im_mask = im_mask_r | im_mask_s



        with fits.open(fname_psf_r) as hdul:
            hdr_psf_r = hdul[0].header
        with fits.open(fname_psf_s) as hdul:
            hdr_psf_s = hdul[0].header

        im_diff[im_mask] = np.nan

    #####################################################################
    #Setup DAOStarFinder
    if 'FWHM_major' in hdr_psf_r:
        fwhm1 = (hdr_psf_r['FWHM_major'] + hdr_psf_r['FWHM_minor']) / 2.
    elif 'FWHM' in hdr_psf_r:
        fwhm1 = hdr_psf_r['FWHM']
    
    if 'FWHM_major' in hdr_psf_s:
        fwhm2 = (hdr_psf_s['FWHM_major'] + hdr_psf_s['FWHM_minor']) / 2.
    elif 'FWHM' in hdr_psf_s:
        fwhm2 = hdr_psf_s['FWHM']
    
    fwhm = np.max([ fwhm1/ps_grid, fwhm2/ps_grid ])
    fwhm = np.ceil(fwhm).astype(int)
    # print('\t\t Using FWHM: {} px'.format(fwhm))
    
    # #Split up the image into a grid of smaller images for better background estimation and star finding
    # xsplit = np.arange(0, im_diff.shape[1], 500)
    # ysplit = np.arange(0, im_diff.shape[0], 500)

    # stardat_all = []    
    # for i in range(len(cutout_dat)):
    #     xc = cutout_dat['COL_INDEX'][i]
    #     yc = cutout_dat['ROW_INDEX'][i]
        
    #     dx = cutout_dat['N1'][i]
    #     dy = cutout_dat['N0'][i]
        
    #     im_diff_i = Cutout2D(im_diff, position=(xc, yc), size=(dx, dy), wcs=wcs_diff)
    #     im_mask_i = Cutout2D(im_mask, position=(xc, yc), size=(dx, dy), wcs=wcs_diff)
        
    #     if np.all(im_diff_i.data == 0.) or np.all(np.isnan(im_diff_i.data)) or np.all(im_mask_i.data):
    #         continue

    with warnings.catch_warnings(action="ignore"):
        bkg = sigma_clipped_stats(im_diff, mask=im_mask, sigma=3.0, maxiters=None)
    
    if band in ['F090W', 'F115W', 'F150W', 'F200W', 'F277W']:
        thresh = 4*bkg[-1]
    else:
        thresh = 3*bkg[-1]
        

    # try:
    sf = DAOStarFinder(thresh, fwhm, roundlo=-.6, roundhi=.6, exclude_border=True, min_separation=matching_radius/ps_grid)
    stardat = sf.find_stars(im_diff)
    # except:
    #     continue

    
    ##############################################################
    #Transform from pixel to RA/DEC
    
    if stardat is not None:
        ra, dec = wcs_diff.wcs_pix2world(stardat['xcentroid'].data, stardat['ycentroid'].data, 0)
        stardat['RA'] = ra
        stardat['DEC'] = dec
        
        # stardat_all.append(stardat)
                
    # stardat = vstack(stardat, join_type='exact')

    return stardat



def combine_sources(stardat, stardat_n, matching_radius=.6):

    cols_in = ['xcentroid', 'ycentroid', 'sharpness', 'roundness1', 'roundness2', 'npix', 'peak', 'flux', 'RA', 'DEC']#, 'AperFlux', 'AperFluxErr', 'SNR']
    cols_out_p = ['X', 'Y', 'Sharpness', 'Roundness1', 'Roundness2', 'Npix', 'Peak', 'Flux', 'RA', 'DEC']#, 'AperFlux', 'AperFluxErr', 'SNR']
    cols_out_n = [col+'_n' for col in cols_out_p]

    cols_in_float = ['xcentroid', 'ycentroid', 'sharpness', 'roundness1', 'roundness2', 'peak', 'flux', 'RA', 'DEC']#, 'AperFlux', 'AperFluxErr', 'SNR']
    cols_in_int = ['npix']

    cols_out = []
    for i in range(len(cols_out_p)):    
        cols_out.append(cols_out_p[i])

    for i in range(len(cols_out_n)):
        cols_out.append(cols_out_n[i])
        
    cols_out_final = ['Index'] + cols_out
    
    
    
    
    if (len(stardat) > 0) and (len(stardat_n) > 0):
        coords = SkyCoord(ra=stardat['RA']*u.deg, dec=stardat['DEC']*u.deg)
        coords_n = SkyCoord(ra=stardat_n['RA']*u.deg, dec=stardat_n['DEC']*u.deg)

        idx, d2d, _ = coords.match_to_catalog_sky(coords_n)
        mask = d2d.arcsec < matching_radius

    else:
        mask = np.array([], dtype=bool)
        idx = np.array([], dtype=int)

    
    # print('\t\t Combining POS+NEG: {} matched'.format(mask.sum()))
    
    #In POS+NEG
    tab1 = Table()
    for c1, c2 in zip(cols_in, cols_out_p):
        tab1[c2] = stardat[c1][mask]
    for c1, c2 in zip(cols_in, cols_out_n):
        tab1[c2] = stardat_n[c1][idx[mask]]
        
    # print('\t\t Adding unmatched sources from POS: {}'.format((~mask).sum()))
        
    #In POS only
    
    if len(stardat) > 0:
        tab2 = Table()
        for c1, c2 in zip(cols_in, cols_out_p):
            tab2[c2] = stardat[c1][~mask]
        for c1, c2 in zip(cols_in, cols_out_n):
            if c1 in cols_in_float:
                tab2[c2] = np.full( (~mask).sum(), np.nan)
            elif c1 in cols_in_int:
                tab2[c2] = np.full( (~mask).sum(), -1)        
    
    else:
        tab2 = Table(names=tab1.colnames)

    #In NEG only
    # print('\t\t Finding unmatched sources in NEG')
    unmatched_mask = np.ones(len(stardat_n), dtype=bool)
    unmatched_mask[idx[mask]] = False

    # print('\t\t Adding unmatched sources from NEG: {}'.format(unmatched_mask.sum()))

    if len(stardat_n) > 0:
        tab3 = Table()
        for c1, c2 in zip(cols_in, cols_out_n):
            tab3[c2] = stardat_n[c1][unmatched_mask]
        for c1, c2 in zip(cols_in, cols_out_p):
            if c1 in cols_in_float:
                tab3[c2] = np.full( unmatched_mask.sum(), np.nan)
            elif c1 in cols_in_int:
                tab3[c2] = np.full( unmatched_mask.sum(), -1)
            
    else:
        tab3 = Table(names=tab1.colnames)


    #Combine  
    tab1 = tab1[cols_out].copy()
    tab2 = tab2[cols_out].copy()
    tab3 = tab3[cols_out].copy()

    # print('\t\t Combining tables')
    tab_all = vstack([tab1, tab2, tab3])
    tab_all['Index'] = np.arange(len(tab_all))

    tab_all = tab_all[cols_out_final].copy()
    # print('\t\t Found {} sources'.format(len(tab_all)))

    return tab_all

##########################################################################################################################################
##########################################################################################################################################
# Combine catalogs across bands

def combine_sources_bands(maindir, bands, matching_radius=.6):
    
    cols_in_float = ['X', 'Y', 'Sharpness', 'Roundness1', 'Roundness2', 'Peak', 'Flux', 'RA', 'DEC']#, 'AperFlux', 'AperFluxErr', 'SNR']
    cols_in_int = ['Npix', 'Cutout']

    cols_in_float_n = [col + '_n' for col in cols_in_float]
    cols_in_int_n = ['Npix_n']

    wl = []
    for band in bands:
        wl.append(int(band[1:-1]))

    sort_ind = np.argsort(wl)
    bands_ordered = np.array(bands)[sort_ind]

    ################################################################
    # Combine Band1+Band2

    tab1 = Table.read(maindir + 'diff_sources_{}.fits.gz'.format(bands_ordered[0], matching_radius))
    tab2 = Table.read(maindir + 'diff_sources_{}.fits.gz'.format(bands_ordered[1], matching_radius))


    cols_in = tab1.colnames
    cols_indiv = []
    for c in cols_in:
         if 'Index' in c:
             continue

         cols_indiv.append(c)


    mask_pos = ~np.isnan(tab1['RA'].data)
    ra1 = np.zeros(len(tab1), dtype=float)
    ra1[mask_pos] = tab1['RA'][mask_pos].data
    ra1[~mask_pos] = tab1['RA_n'][~mask_pos].data
    dec1 = np.zeros(len(tab1), dtype=float)
    dec1[mask_pos] = tab1['DEC'][mask_pos].data
    dec1[~mask_pos] = tab1['DEC_n'][~mask_pos].data

    mask_pos = ~np.isnan(tab2['RA'].data)
    ra2 = np.zeros(len(tab2), dtype=float)
    ra2[mask_pos] = tab2['RA'][mask_pos].data
    ra2[~mask_pos] = tab2['RA_n'][~mask_pos].data
    dec2 = np.zeros(len(tab2), dtype=float)
    dec2[mask_pos] = tab2['DEC'][mask_pos].data
    dec2[~mask_pos] = tab2['DEC_n'][~mask_pos].data


    # ra1 = np.vstack([tab1['RA'], tab1['RA_n']])
    # ra1 = np.nanmedian(ra1, axis=0)
    # dec1 = np.vstack([tab1['DEC'], tab1['DEC_n']])
    # dec1 = np.nanmedian(dec1, axis=0)

    # ra2 = np.vstack([tab2['RA'], tab2['RA_n']])
    # ra2 = np.nanmedian(ra2, axis=0)
    # dec2 = np.vstack([tab2['DEC'], tab2['DEC_n']])
    # dec2 = np.nanmedian(dec2, axis=0)

    coords1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
    coords2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)



    idx, d2d, _ = coords1.match_to_catalog_sky(coords2)
    mask = d2d.arcsec < matching_radius

    print('\t Combining {}+{}: {} matched'.format(bands_ordered[0], bands_ordered[1], mask.sum()))

    #In 1+2
    tab1_new = Table()
    for c in cols_indiv:
        tab1_new[c + '_' + bands_ordered[0]] = tab1[c][mask]
        tab1_new[c + '_' + bands_ordered[1]] = tab2[c][idx[mask]]

    print( '\t Adding unmatched sources from {}: {}'.format(bands_ordered[0], (~mask).sum()) )

    #In 2 only
    tab2_new = Table()
    for c in cols_indiv:
        #print(c, c in cols_in_float, c in cols_in_float_n)
        
        tab2_new[c + '_' + bands_ordered[0]] = tab1[c][~mask]

        if (c in cols_in_float) or (c in cols_in_float_n):
            tab2_new[c + '_' + bands_ordered[1]] = np.full( (~mask).sum(), np.nan)
        elif (c in cols_in_int) or (c in cols_in_int_n):
            tab2_new[c + '_' + bands_ordered[1]] = np.full( (~mask).sum(), -1)


    #In 1 only
    unmatched_mask = np.ones(len(tab2), dtype=bool)
    unmatched_mask[idx[mask]] = False

    print('\t Adding unmatched sources from {}: {}'.format(bands_ordered[1], unmatched_mask.sum()))

    tab3_new = Table()
    for c in cols_indiv:
        tab3_new[c + '_' + bands_ordered[1]] = tab2[c][unmatched_mask]

        if (c in cols_in_float) or (c in cols_in_float_n):
            tab3_new[c + '_' + bands_ordered[0]] = np.full( unmatched_mask.sum(), np.nan)
        elif (c in cols_in_int) or (c in cols_in_int_n):
            tab3_new[c + '_' + bands_ordered[0]] = np.full( unmatched_mask.sum(), -1)
    


            
    #Combine
    tab_tot = vstack([tab1_new, tab2_new, tab3_new], join_type='exact')    
    tab_tot['Index'] = np.arange(len(tab_tot))

    print('\t Found {} sources'.format(len(tab_tot)))

    return tab_tot

##########################################################################################################################################
##########################################################################################################################################
# Multiprocessing

def get_stardat_table_full_image(mdir, fname_cutout, band, ps_grid, radius, difftype='sub'):
    
    fname_r = mdir + 'input/nexus_wide01_{}.fits'.format(band)
    fname_s = mdir + 'input/nexus_deep01_{}.fits'.format(band)

    fname_mask_r = mdir + 'input/nexus_wide01_{}.maskin.fits'.format(band)
    fname_mask_s = mdir + 'input/nexus_deep01_{}.maskin.fits'.format(band)

    fname_psf_r = mdir + 'psf/nexus_wide01_{}.psf.fits'.format(band)
    fname_psf_s = mdir + 'psf/nexus_deep01_{}.psf.fits'.format(band)
    
    if difftype == 'sfft':
        fname_diff = mdir + 'output/nexus_deep01_{}.sfftdiff.fits'.format(band)
    else:
        fname_diff = ''

    #Get sources in pos/neg DIFF images
    stardat = get_sources_indiv_image(fname_diff, fname_cutout, fname_r, fname_s, fname_mask_r, fname_mask_s, fname_psf_r, fname_psf_s,
                                    band=band, neg=False, ps_grid=ps_grid, difftype=difftype, matching_radius=radius)

    stardat_n = get_sources_indiv_image(fname_diff, fname_cutout, fname_r, fname_s, fname_mask_r, fname_mask_s, fname_psf_r, fname_psf_s,
                                        band=band, neg=True, ps_grid=ps_grid, difftype=difftype, matching_radius=radius)
    
    if (stardat is None) and (stardat_n is None):
        return None
    if (stardat is None) and (stardat_n is not None):
        stardat = Table(names=stardat_n.colnames)
    if (stardat is not None) and (stardat_n is None):
        stardat_n = Table(names=stardat.colnames)

    #Combine
    tab_all = combine_sources(stardat, stardat_n, matching_radius=radius)
    
    return tab_all

def get_stardat_table_indiv_cutout(label, fname_cutout, mdir, band, ps_grid, radius, difftype='sub'):
    
    fname_r = mdir + 'output_{}/input/nexus_wide01_{}.skysub.fits'.format(label, band)
    fname_s = mdir + 'output_{}/input/nexus_deep01_{}.skysub.fits'.format(label, band)
    
    fname_mask_r = mdir + 'output_{}/input/nexus_wide01_{}.maskin.fits'.format(label, band)
    fname_mask_s = mdir + 'output_{}/input/nexus_deep01_{}.maskin.fits'.format(label, band)

    fname_psf_r = mdir + 'output_{}/psf/nexus_wide01_{}.psf.fits'.format(label, band)
    fname_psf_s = mdir + 'output_{}/psf/nexus_deep01_{}.psf.fits'.format(label, band)
    
    fname_diff = mdir + 'output_{}/output/nexus_deep01_{}.sfftdiff.fits'.format(label, band)

    #Get sources in pos/neg DIFF images
    stardat = get_sources_indiv_image(fname_diff, fname_cutout, fname_r, fname_s, fname_mask_r, fname_mask_s, fname_psf_r, fname_psf_s,
                                    band=band, neg=False, ps_grid=ps_grid, difftype=difftype, matching_radius=radius)
    stardat_n = get_sources_indiv_image(fname_diff, fname_cutout, fname_r, fname_s, fname_mask_r, fname_mask_s, fname_psf_r, fname_psf_s,
                                        band=band, neg=True, ps_grid=ps_grid, difftype=difftype, matching_radius=radius)
    
    if (stardat is None) and (stardat_n is None):
        return None
    if (stardat is None) and (stardat_n is not None):
        stardat = Table(names=stardat_n.colnames)
    if (stardat is not None) and (stardat_n is None):
        stardat_n = Table(names=stardat.colnames)

    #Combine
    # print('\t Combining sources')
    tab_all = combine_sources(stardat, stardat_n, matching_radius=radius)
    tab_all['Cutout'] = label

    return tab_all


##########################################################################################################################################
##########################################################################################################################################
# Multiprocessing


def get_candidates(maindir, bands, fname_nexus_cat, apcorr_dir, cc1=False, cc2=False,
                   matching_radius=.3, r1_vals=[1.,.5,.25], r2_vals=[2.,1.,.5],
                   difftype='sub'):

    if cc1 and cc2:
        outdir_suffix = '_CC'
    elif cc1:
        outdir_suffix = '_CC1'
    elif cc2:
        outdir_suffix = '_CC2'
    else:
        outdir_suffix = ''
        
    if cc1:
        suff1 = '_CC'
    else:
        suff1 = ''
        
    if cc2:
        suff2 = '_CC'
    else:
        suff2 = ''
    
    outdir = maindir + 'source_sfft_nexus_wide01_deep01_match{:.2f}{}/'.format(matching_radius, outdir_suffix)
    os.makedirs(outdir, exist_ok=True)

    ################################################################
    ################################################################
    #Get POS/NEG sources in each band for each cutout

    print('Getting sources in individual cutouts')
    for b in bands:
        if b == 'F444W':
            ps = .06
            suff = suff2
        elif b == 'F200W':
            ps = .03
            suff = suff1
        
        mdir = maindir + 'sfft_nexus_wide01_deep01_{}{}/'.format(b, suff)
        fname_out = outdir + 'diff_sources_{}.fits.gz'.format(b, matching_radius)
        
        fname_cutout = mdir + 'cutout_info.txt'
        cutout_dat = Table.read(fname_cutout, format='ascii')
        labels = cutout_dat['LABEL'].data.astype(int)
        
        if os.path.exists(fname_out):
            continue
        
        print(b)
        
        if difftype == 'sfft':
            func = partial(get_stardat_table_indiv_cutout, fname_cutout=fname_cutout, mdir=mdir, band=b, ps_grid=ps, radius=matching_radius, difftype=difftype)
            
            output = p_map(func, labels)
            stardats = []
            for i in range(len(output)):
                if output[i] is not None:
                    stardats.append(output[i])

            tab_allcutout = vstack(stardats, join_type='exact')
            
        else:
            tab_allcutout = get_stardat_table_full_image(mdir, fname_cutout, b, ps, matching_radius, difftype=difftype)
            
        print('\t Found {} sources'.format(len(tab_allcutout)))
        print('\t Writing to file')
        tab_allcutout.write(fname_out, overwrite=True)

    ################################################################
    ################################################################
    #Combine across bands
    print('Combining across bands')

    tab_allband = combine_sources_bands(outdir, bands, matching_radius=matching_radius)
    
    tab_allband_simple = Table()
    tab_allband_simple['Index'] = tab_allband['Index']
    
    if difftype == 'sfft':
        tab_allband_simple['Cutout_{}'.format(bands[0])] = tab_allband['Cutout_{}'.format(bands[0])]
        tab_allband_simple['Cutout_{}'.format(bands[1])] = tab_allband['Cutout_{}'.format(bands[1])]
    else:
        tab_allband_simple['Cutout_{}'.format(bands[0])] = np.full(len(tab_allband), -1)
        tab_allband_simple['Cutout_{}'.format(bands[1])] = np.full(len(tab_allband), -1)


    ra_cols = [col for col in tab_allband.colnames if 'RA' in col]
    dec_cols = [col for col in tab_allband.colnames if 'DEC' in col]

    ra_tot = np.vstack([tab_allband[col].data.data for col in ra_cols])
    ra_tot = np.nanmedian(ra_tot, axis=0)
    dec_tot = np.vstack([tab_allband[col].data.data for col in dec_cols])
    dec_tot = np.nanmedian(dec_tot, axis=0)

    tab_allband_simple['RA'] = ra_tot
    tab_allband_simple['DEC'] = dec_tot
    tab_allband_simple['DETECT_{}'.format(bands[0])] = ~np.isnan(tab_allband['RA_{}'.format(bands[0])].data.data)
    tab_allband_simple['DETECT_{}'.format(bands[1])] = ~np.isnan(tab_allband['RA_{}'.format(bands[1])].data.data)
    tab_allband_simple['DETECT_{}_n'.format(bands[0])] = ~np.isnan(tab_allband['RA_n_{}'.format(bands[0])].data.data)
    tab_allband_simple['DETECT_{}_n'.format(bands[1])] = ~np.isnan(tab_allband['RA_n_{}'.format(bands[1])].data.data)


    #Get cutout number for sources not detected in F444W
    fname_cutout = '/data6/stone28/nexus/sfft_nexus_wide01_deep01_{}{}/cutout_info.txt'.format(bands[1], suff2)
    cutout_dat = Table.read(fname_cutout, format='ascii')
    dx = dy = cutout_dat['N0'][0]

    fname_r = '/data6/stone28/nexus/sfft_nexus_wide01_deep01_{}{}/input/nexus_wide01_{}.fits'.format(bands[1], suff2, bands[1])
    with fits.open(fname_r) as hdul:
        hdr_r = hdul[0].header
        wcs_r = WCS(hdr_r)
        
    xs, ys = wcs_r.all_world2pix(tab_allband_simple['RA'].data, tab_allband_simple['DEC'].data, 0)
    mask1 = (tab_allband_simple['Cutout_{}'.format(bands[1])] == -1)
    for i in tqdm( range(len(cutout_dat)) ):
        yc = cutout_dat['ROW_INDEX'][i]
        xc = cutout_dat['COL_INDEX'][i]
        y1 = yc - dx/2
        y2 = yc + dx/2
        x1 = xc - dy/2
        x2 = xc + dy/2
        
        mask2 = (ys > y1) & (ys < y2) & (xs > x1) & (xs < x2)
        tab_allband_simple['Cutout_{}'.format(bands[1])][mask1 & mask2] = cutout_dat['LABEL'][i]        


    #Get cutout number for sources not detected in F200W
    fname_cutout = '/data6/stone28/nexus/sfft_nexus_wide01_deep01_{}{}/cutout_info.txt'.format(bands[0], suff1)
    cutout_dat = Table.read(fname_cutout, format='ascii')
    dx = dy = cutout_dat['N0'][0]

    fname_r = '/data6/stone28/nexus/sfft_nexus_wide01_deep01_{}{}/input/nexus_wide01_{}.fits'.format(bands[0], suff1, bands[0])
    with fits.open(fname_r) as hdul:
        hdr_r = hdul[0].header
        wcs_r = WCS(hdr_r)

    xs, ys = wcs_r.all_world2pix(tab_allband_simple['RA'].data, tab_allband_simple['DEC'].data, 0)
    mask1 = (tab_allband_simple['Cutout_{}'.format(bands[0])] == -1)
    for i in tqdm( range(len(cutout_dat)) ):
        yc = cutout_dat['ROW_INDEX'][i]
        xc = cutout_dat['COL_INDEX'][i]
        y1 = yc - dx/2
        y2 = yc + dx/2
        x1 = xc - dy/2
        x2 = xc + dy/2
        
        mask2 = (ys > y1) & (ys < y2) & (xs > x1) & (xs < x2)
        tab_allband_simple['Cutout_{}'.format(bands[0])][mask1 & mask2] = cutout_dat['LABEL'][i]       


    tab_allband_simple.write(outdir + 'diff_sources_allband.fits.gz', overwrite=True)
    
    
    ################################################################
    ################################################################
    #Match candidates to NEXUS catalog
    #Take the closest match to each NEXUS source within 0.5", discard the rest
    #We only want nuclear variability, so we don't want multiple matches to the same source
    print('Matching candidates to NEXUS catalog')

    tab_allband_simple = Table.read(outdir + 'diff_sources_allband.fits.gz')

    nexus_catdat = Table.read(fname_nexus_cat)
    coord_nex = SkyCoord(ra=nexus_catdat['RA'].data*u.deg, dec=nexus_catdat['DEC'].data*u.deg)
    coord_all = SkyCoord(ra=tab_allband_simple['RA'].data*u.deg, dec=tab_allband_simple['DEC'].data*u.deg)

    idx, d2d, _ = coord_nex.match_to_catalog_sky(coord_all)
    mask = d2d.arcsec < .5


    tab_match = tab_allband_simple[idx[mask]].copy()
    tab_match['NEXUS_ID'] = nexus_catdat['ID'][mask]
    tab_match['NEXUS_Z'] = nexus_catdat['z'][mask]
    tab_match['NEXUS_RA'] = nexus_catdat['RA'][mask]
    tab_match['NEXUS_DEC'] = nexus_catdat['DEC'][mask]
    tab_match['NEXUS_HostSep'] = d2d.arcsec[mask]
    tab_match.write(outdir + 'diff_sources_allband_match.fits.gz', overwrite=True)
    
    print('\t NEXUS catalog contains {} sources'.format(len(nexus_catdat)))
    print('\t Found {} matched sources'.format(len(tab_match)))
    
    ################################################################
    ################################################################
    #Get fluxes and SNRs
    print('Getting fluxes and SNRs')
    skysub_vals = [False, True]
    catcoord_vals = [False, True]

    tab_allband_simple = Table.read(outdir + 'diff_sources_allband_match.fits.gz')

    for b in bands:
        if b == 'F444W':
            ps_grid = .06
            suff = suff2
        elif b == 'F200W':
            ps_grid = .03
            suff = suff1
            
            
        mdir = '/data6/stone28/nexus/sfft_nexus_wide01_deep01_{}{}/'.format(b, suff)
        cutout_vals = np.unique(tab_allband_simple['Cutout_{}'.format(b)].data)
        
        print(b)
        
        for skysub in skysub_vals:
            print('\t Sky Subtraction: {}'.format(skysub))
            
                        #Nsource, Nradius, [ref, sci, rawdiff, diff], [nexus_coord, my_coord]
            f_vals = np.full((len(tab_allband_simple), len(r1_vals), 4, 2), np.nan)
            ferr_vals = np.full((len(tab_allband_simple), len(r1_vals), 4, 2), np.nan)
            snr_vals = np.full((len(tab_allband_simple), len(r1_vals), 4, 2), np.nan)
            
            if difftype == 'sub':
                fname_r = mdir + 'input/nexus_wide01_{}.fits'.format(b)
                fname_s = mdir + 'input/nexus_deep01_{}.fits'.format(b)       
                
                fname_mask_r = mdir + 'input/nexus_wide01_{}.maskin.fits'.format(b)
                fname_mask_s = mdir + 'input/nexus_deep01_{}.maskin.fits'.format(b)
                
                fname_err_r = mdir + 'noise/nexus_wide01_{}.noise.fits'.format(b)
                fname_err_s = mdir + 'noise/nexus_deep01_{}.noise.fits'.format(b)
                
                with warnings.catch_warnings(action="ignore"):
                    with fits.open(fname_r) as hdul:
                        im_r = hdul[0].data
                        hdr_r = hdul[0].header
                        wcs_r = WCS(hdr_r)
                    with fits.open(fname_s) as hdul:
                        im_s = hdul[0].data
                        hdr_s = hdul[0].header
                        wcs_s = WCS(hdr_s)
                        
                    im_rawdiff = im_r - im_s
                
                    with fits.open(fname_err_r) as hdul:
                        im_err_r = hdul[0].data
                    with fits.open(fname_err_s) as hdul:
                        im_err_s = hdul[0].data
                        
                    im_var_r = im_err_r**2
                    im_var_s = im_err_s**2
                    im_var_rawdiff = im_var_r + im_var_s

                    with fits.open(fname_mask_r) as hdul:
                        im_mask_r = hdul[0].data.astype(bool)
                    with fits.open(fname_mask_s) as hdul:
                        im_mask_s = hdul[0].data.astype(bool)
                        
                    im_mask = im_mask_r | im_mask_s
                    
                    for i, (im, imv, wcs, label) in enumerate(zip([im_r, im_s, im_rawdiff], [im_var_r, im_var_s, im_var_rawdiff], [wcs_r, wcs_s, wcs_r], ['REF', 'SCI', 'RAWDIFF'])):
                        for j, (r1, r2) in enumerate( zip(r1_vals, r2_vals) ):
                            
                            if label == 'RAWDIFF':
                                fname_apcorr = None
                            else:
                                fname_apcorr = apcorr_dir + 'aperture_corrections_r{:.2f}.txt'.format(r1)
                            
                            for k, catcoord in enumerate(catcoord_vals):
                                if catcoord:
                                    ra = tab_allband_simple['NEXUS_RA'].data
                                    dec = tab_allband_simple['NEXUS_DEC'].data
                                else:
                                    ra = tab_allband_simple['RA'].data
                                    dec = tab_allband_simple['DEC'].data
                                
                                x, y = wcs.wcs_world2pix(ra, dec, 0)
                                f, ferr = get_fluxes(im, imv, im_mask, x, y, fname_apcorr, b, ps_grid=ps_grid, neg=False, r1=r1, r2=r2, subtract_sky=skysub)
                                
                                with warnings.catch_warnings(action="ignore"):
                                    snr = f / ferr
                        
                                f_vals[:, j, i, k] = f
                                ferr_vals[:, j, i, k] = ferr
                                snr_vals[:, j, i, k] = snr


            for n in tqdm( cutout_vals ):
                if (n == -1) and (difftype == 'sfft'):
                    continue
                
                if difftype == 'sfft':
                    fname_r = mdir + 'output_{}/input/nexus_wide01_{}.skysub.fits'.format(n, b)
                    fname_s = mdir + 'output_{}/input/nexus_deep01_{}.skysub.fits'.format(n, b)
                    
                    fname_err_r = mdir + 'output_{}/noise/nexus_wide01_{}.noise.fits'.format(n, b)
                    fname_err_s = mdir + 'output_{}/noise/nexus_deep01_{}.noise.fits'.format(n, b)
                    
                    fname_mask_r = mdir + 'output_{}/input/nexus_wide01_{}.maskin.fits'.format(n, b)
                    fname_mask_s = mdir + 'output_{}/input/nexus_deep01_{}.maskin.fits'.format(n, b)    
                    
                    
                elif difftype == 'sub':
                    fname_r = mdir + 'input/nexus_wide01_{}.fits'.format(b)
                    fname_s = mdir + 'input/nexus_deep01_{}.fits'.format(b)       
                    
                    fname_mask_r = mdir + 'input/nexus_wide01_{}.maskin.fits'.format(b)
                    fname_mask_s = mdir + 'input/nexus_deep01_{}.maskin.fits'.format(b)
                    
                    fname_err_r = mdir + 'noise/nexus_wide01_{}.noise.fits'.format(b)
                    fname_err_s = mdir + 'noise/nexus_deep01_{}.noise.fits'.format(b)
                         
                
                fname_diff = mdir + 'output_{}/output/nexus_deep01_{}.sfftdiff.decorr.fits'.format(n, b)
                fname_diff_snr = mdir + 'output_{}/output/nexus_deep01_{}.sfftdiff.decorr.snr.fits'.format(n, b)

                with warnings.catch_warnings(action="ignore"):
                    if difftype == 'sfft':
                        with fits.open(fname_r) as hdul:
                            im_r = hdul[0].data
                            hdr_r = hdul[0].header
                            wcs_r = WCS(hdr_r)
                        with fits.open(fname_s) as hdul:
                            im_s = hdul[0].data
                            hdr_s = hdul[0].header
                            wcs_s = WCS(hdr_s)
                            
                        im_rawdiff = im_r - im_s
                            
                        with fits.open(fname_err_r) as hdul:
                            im_err_r = hdul[0].data
                        with fits.open(fname_err_s) as hdul:
                            im_err_s = hdul[0].data
                            
                        im_var_r = im_err_r**2
                        im_var_s = im_err_s**2
                        im_var_rawdiff = im_var_r + im_var_s
                        
                        with fits.open(fname_mask_r) as hdul:
                            im_mask_r = hdul[0].data.astype(bool)
                        with fits.open(fname_mask_s) as hdul:
                            im_mask_s = hdul[0].data.astype(bool)
                            
                        im_mask = im_mask_r | im_mask_s
                        
                    
                    if (n == -1):
                        im_diff = 1.
                        wcs_diff = None
                        im_snr = 1.
                    else:                        
                        with fits.open(fname_diff) as hdul:
                            im_diff = hdul[0].data
                            hdr_diff = hdul[0].header
                            wcs_diff = WCS(hdr_diff)
                        with fits.open(fname_diff_snr) as hdul:
                            im_snr = hdul[0].data
                        
                    im_var = (im_diff / im_snr)**2
                
                
                cutout_mask = (tab_allband_simple['Cutout_{}'.format(b)].data == n)

                if difftype == 'sfft':
                    for i, (im, imv, wcs, label) in enumerate(  zip([im_r, im_s, im_diff, im_rawdiff], [im_var_r, im_var_s, im_var, im_var_rawdiff], [wcs_r, wcs_s, wcs_diff, wcs_r], ['REF', 'SCI', 'RAWDIFF', 'DIFF'])  ):

                        if wcs is None:
                            continue

                        if (label == 'DIFF') and (difftype == 'sub'):
                            im_mask_i = (im_diff == 0.) | np.isnan(im_diff)
                        else:
                            im_mask_i = im_mask.copy()


                        for j, (r1, r2) in enumerate( zip(r1_vals, r2_vals) ):
                            
                            if label in ['RAWDIFF', 'DIFF']:
                                fname_apcorr = None
                            else:
                                fname_apcorr = apcorr_dir + 'aperture_corrections_r{:.2f}.txt'.format(r1)

                            for k, catcoord in enumerate(catcoord_vals):
                                if catcoord:
                                    ra = tab_allband_simple['NEXUS_RA'][cutout_mask].data
                                    dec = tab_allband_simple['NEXUS_DEC'][cutout_mask].data
                                else:
                                    ra = tab_allband_simple['RA'][cutout_mask].data
                                    dec = tab_allband_simple['DEC'][cutout_mask].data
                                
                                x, y = wcs.wcs_world2pix(ra, dec, 0)
                                f, ferr = get_fluxes(im, imv, im_mask_i, x, y, fname_apcorr, b, ps_grid=ps_grid, neg=False, r1=r1, r2=r2, subtract_sky=skysub)
                                
                                with warnings.catch_warnings(action="ignore"):
                                    snr = f / ferr
                        
                                f_vals[cutout_mask, j, i, k] = f
                                ferr_vals[cutout_mask, j, i, k] = ferr
                                snr_vals[cutout_mask, j, i, k] = snr
                                
                else:
                    i = 3
                    im = im_diff
                    imv = im_var
                    wcs = wcs_diff
                    label = 'DIFF'
                    if wcs is None:
                        pass
                    else:
                        if (label == 'DIFF') and (difftype == 'sub'):
                            im_mask_i = (im_diff == 0.) | np.isnan(im_diff)
                        else:
                            im_mask_i = im_mask.copy()


                        for j, (r1, r2) in enumerate( zip(r1_vals, r2_vals) ):
                            
                            if label in ['RAWDIFF', 'DIFF']:
                                fname_apcorr = None
                            else:
                                fname_apcorr = apcorr_dir + 'aperture_corrections_r{:.2f}.txt'.format(r1)
                            
                            for k, catcoord in enumerate(catcoord_vals):
                                if catcoord:
                                    ra = tab_allband_simple['NEXUS_RA'][cutout_mask].data
                                    dec = tab_allband_simple['NEXUS_DEC'][cutout_mask].data
                                else:
                                    ra = tab_allband_simple['RA'][cutout_mask].data
                                    dec = tab_allband_simple['DEC'][cutout_mask].data
                                
                                x, y = wcs.wcs_world2pix(ra, dec, 0)
                                f, ferr = get_fluxes(im, imv, im_mask_i, x, y, fname_apcorr, b, ps_grid=ps_grid, neg=False, r1=r1, r2=r2, subtract_sky=skysub)
                                
                                with warnings.catch_warnings(action="ignore"):
                                    snr = f / ferr
                        
                                f_vals[cutout_mask, j, i, k] = f
                                ferr_vals[cutout_mask, j, i, k] = ferr
                                snr_vals[cutout_mask, j, i, k] = snr


            for i, label in enumerate(['REF', 'SCI', 'RAWDIFF', 'DIFF']):
                for j, (catcoord, label2) in enumerate( zip(catcoord_vals, ['', '_catcoord']) ):
                    if skysub:
                        tab_allband_simple['Flux_{}_{}_bkgsub{}'.format(b, label, label2)] = f_vals[:,:,i, j]
                        tab_allband_simple['FluxErr_{}_{}_bkgsub{}'.format(b, label, label2)] = ferr_vals[:,:,i, j]
                        tab_allband_simple['SNR_{}_{}_bkgsub{}'.format(b, label, label2)] = snr_vals[:,:,i, j]
                    else:
                        tab_allband_simple['Flux_{}_{}{}'.format(b, label, label2)] = f_vals[:,:,i, j]
                        tab_allband_simple['FluxErr_{}_{}{}'.format(b, label, label2)] = ferr_vals[:,:,i, j]
                        tab_allband_simple['SNR_{}_{}{}'.format(b, label, label2)] = snr_vals[:,:,i, j]


    # print('Getting complex SNRs')
    # for b in bands:
    #     if b == 'F444W':
    #         ps_grid = .06
    #         suff = suff2
    #     elif b == 'F200W':
    #         ps_grid = .03
    #         suff = suff1
        
            
            
    #     mdir = '/data6/stone28/nexus/sfft_nexus_wide01_deep01_{}{}/'.format(b, suff)
    #     cutout_vals = np.unique(tab_allband_simple['Cutout_{}'.format(b)].data)
        
    #     snr_vals_sfft = np.full((len(tab_allband_simple), len(r1_vals), 2), np.nan)
    #     snr_vals_multidiff = np.full((len(tab_allband_simple), len(r1_vals), 2, 2), np.nan)

    #     for n in tqdm( cutout_vals ):
    #         if n == -1:
    #             continue
            
    #         fname_r = mdir + 'output_{}/input/nexus_wide01_{}.skysub.fits'.format(n, b)
    #         fname_s = mdir + 'output_{}/input/nexus_deep01_{}.skysub.fits'.format(n, b)
            
    #         fname_noise_r = mdir + 'output_{}/noise/nexus_wide01_{}.noise.fits'.format(n, b)
    #         fname_noise_s = mdir + 'output_{}/noise/nexus_deep01_{}.noise.fits'.format(n, b)
            
    #         fname_mask_r = mdir + 'output_{}/input/nexus_wide01_{}.maskin.fits'.format(n, b)
    #         fname_mask_s = mdir + 'output_{}/input/nexus_deep01_{}.maskin.fits'.format(n, b)        
            
    #         fname_diff = mdir + 'output_{}/output/nexus_deep01_{}.sfftdiff.decorr.fits'.format(n, b)    
    #         fname_diff_snr = mdir + 'output_{}/output/nexus_deep01_{}.sfftdiff.decorr.snr.fits'.format(n, b)

    #         with warnings.catch_warnings(action="ignore"):
    #             with fits.open(fname_diff_snr) as hdul:
    #                 im_snr = hdul[0].data
    #                 hdr_snr = hdul[0].header
    #                 wcs_snr = WCS(hdr_snr)
    #             with fits.open(fname_diff) as hdul:
    #                 im_diff = hdul[0].data
    #                 hdr_diff = hdul[0].header
    #                 wcs_diff = WCS(hdr_diff)
                    
    #             with fits.open(fname_r) as hdul:
    #                 im_r = hdul[0].data
    #                 hdr_r = hdul[0].header
    #                 wcs_r = WCS(hdr_r)
    #             with fits.open(fname_s) as hdul:
    #                 im_s = hdul[0].data
    #                 hdr_s = hdul[0].header
    #                 wcs_s = WCS(hdr_s)
                    
    #             with fits.open(fname_noise_r) as hdul:
    #                 im_err_r = hdul[0].data
    #             with fits.open(fname_noise_s) as hdul:
    #                 im_err_s = hdul[0].data
                    
    #             with fits.open(fname_mask_r) as hdul:
    #                 im_mask_r = hdul[0].data.astype(bool)
    #             with fits.open(fname_mask_s) as hdul:
    #                 im_mask_s = hdul[0].data.astype(bool)
                    
    #             im_mask = im_mask_r | im_mask_s
                
    #             im_rawdiff = im_s - im_r
    #             im_var_rawdiff = im_err_s**2 + im_err_r**2
            
            
    #         cutout_mask = (tab_allband_simple['Cutout_{}'.format(b)].data == n)

    #         for k, catcoord in enumerate(catcoord_vals):
    #             if catcoord:
    #                 ra = tab_allband_simple['NEXUS_RA'][cutout_mask].data
    #                 dec = tab_allband_simple['NEXUS_DEC'][cutout_mask].data
    #             else:
    #                 ra = tab_allband_simple['RA'][cutout_mask].data
    #                 dec = tab_allband_simple['DEC'][cutout_mask].data

    #             x, y = wcs_snr.wcs_world2pix(ra, dec, 0)

    #             for j, (r1, r2) in enumerate( zip(r1_vals, r2_vals) ):
    #                 fname_apcorr = apcorr_dir + 'aperture_corrections_r{:.2f}.txt'.format(r1)
                    
    #                 snr = get_snr(im_snr, im_diff, im_mask, x, y, b, ps_grid=ps_grid, r1=r1)
    #                 snr_vals_sfft[cutout_mask, j, k] = snr
                    
    #                 for l, skysub in enumerate(skysub_vals):
    #                     snr = get_multidiff_snr(im_snr, im_diff, im_mask, 
    #                                             im_rawdiff, im_var_rawdiff,
    #                                             x, y, b, ps, 
    #                                             fname_apcorr, r1=r1, r2=r2,
    #                                             sum_px=False, subtract_sky=skysub)
                        
    #                     snr_vals_multidiff[cutout_mask, j, k, l] = snr

    #     tab_allband_simple['SFFT_SNR_{}'.format(b)] = snr_vals_sfft[:,:,0]
    #     tab_allband_simple['SFFT_SNR_{}_catcoord'.format(b)] = snr_vals_sfft[:,:,1]
    #     tab_allband_simple['MultiDiff_SNR_{}'.format(b)] = snr_vals_multidiff[:,:,0,0]
    #     tab_allband_simple['MultiDiff_SNR_{}_catcoord'.format(b)] = snr_vals_multidiff[:,:,0,1]
    #     tab_allband_simple['MultiDiff_SNR_{}_bkgsub'.format(b)] = snr_vals_multidiff[:,:,1,0]
    #     tab_allband_simple['MultiDiff_SNR_{}_bkgsub_catcoord'.format(b)] = snr_vals_multidiff[:,:,1,1]


    print('Calculating differential magnitudes')
    for b in bands:
        for skysub in skysub_vals:
            for catcoord in catcoord_vals:
                if catcoord:
                    label2 = '_catcoord'
                else:
                    label2 = ''

                for label in ['DIFF', 'RAWDIFF']:
                    if skysub:
                        fd = tab_allband_simple['Flux_{}_{}_bkgsub{}'.format(b, label, label2)].data
                        ferrd = tab_allband_simple['FluxErr_{}_{}_bkgsub{}'.format(b, label, label2)].data
                        
                        fr = tab_allband_simple['Flux_{}_{}_bkgsub{}'.format(b, 'REF', label2)].data
                        ferrr = tab_allband_simple['FluxErr_{}_{}_bkgsub{}'.format(b, 'REF', label2)].data
                        
                        dm = -2.5 * np.log10(np.abs(fd/fr) + 1 ) * np.sign(fd)
                        dm_err = 2.5 * np.log10(1 + (ferrd/fd)**2 + (ferrr/fr)**2)
                        tab_allband_simple['DMag_{}_{}_bkgsub{}'.format(label, b, label2)] = dm
                        tab_allband_simple['DMagErr_{}_{}_bkgsub{}'.format(label, b, label2)] = dm_err

                    else:
                        fd = tab_allband_simple['Flux_{}_{}{}'.format(b, label, label2)].data
                        ferrd = tab_allband_simple['FluxErr_{}_{}{}'.format(b, label, label2)].data
                        
                        fr = tab_allband_simple['Flux_{}_{}{}'.format(b, 'REF', label2)].data
                        ferrr = tab_allband_simple['FluxErr_{}_{}{}'.format(b, 'REF', label2)].data
                        
                        dm = -2.5 * np.log10(np.abs(fd/fr) + 1 ) * np.sign(fd)
                        dm_err = 2.5 * np.log10(1 + (ferrd/fd)**2 + (ferrr/fr)**2)
                        tab_allband_simple['DMag_{}_{}{}'.format(label, b, label2)] = dm
                        tab_allband_simple['DMagErr_{}_{}{}'.format(label, b, label2)] = dm_err


    tab_allband_simple.write(outdir + 'diff_sources_allband_flux.fits.gz', overwrite=True)

    return