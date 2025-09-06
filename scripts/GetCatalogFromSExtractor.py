import glob
import warnings
from functools import partial
from tqdm import tqdm
from p_tqdm import p_map
from mpire import WorkerPool

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, splev


from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats, sigma_clip

from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats


warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

################################################################################################################
#FUNCTIONS FOR COMBINING SEXTRACTOR RESULTS  ###################################################################
################################################################################################################

def match_nexus_catalogs(i, cat_r, cat_s):
    if cat_s['ID'][i] in cat_r['ID'].data:
        ind2 = i
        ind1 = np.argwhere(cat_r['ID'] == cat_s['ID'][i]).flatten()[0]
        return ind1, ind2
        
    else:
        return -1, -1


def combine_pos_neg(dat, dat_n, matching_radius=.3):
    coords_p = SkyCoord(dat['ALPHA_J2000'], dat['DELTA_J2000'], unit=(u.deg, u.deg))
    coords_n = SkyCoord(dat_n['ALPHA_J2000'], dat_n['DELTA_J2000'], unit=(u.deg, u.deg))
    
    idx, d2d, _ = coords_p.match_to_catalog_sky(coords_n)
    mask = d2d < matching_radius * u.arcsec
    
    pos_match = dat[mask].copy()
    neg_match = dat_n[idx[mask]].copy()
    
    pos_unmatch = dat[~mask].copy()
    
    unmatch_mask = np.ones(len(dat_n), dtype=bool)
    unmatch_mask[idx[mask]] = False
    neg_unmatch = dat_n[unmatch_mask].copy()    
    
    
    
    cols_out = ['ALPHA_J2000', 'DELTA_J2000', 'FLUX_AUTO', 'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER', 'ELONGATION', 'FLAGS', 'IMAFLAGS_ISO', 'CLASS_STAR', 'ISOAREA_IMAGE']#, 'ISOAREAF_IMAGE']
    cols_out_n = [col + '_n' for col in cols_out]
    cols_tot = cols_out + cols_out_n
    
    #Both
    tab1 = Table()
    for col in cols_out:
        tab1[col] = pos_match[col]
    for col in cols_out:
        if 'FLUX' in col:
            tab1[col + '_n'] = -neg_match[col]
        else:
            tab1[col + '_n'] = neg_match[col]
            
    #Pos only
    tab2 = Table()
    for col in cols_out:
        tab2[col] = pos_unmatch[col]
    for col in cols_out:
        if 'APER' in col:
            tab2[col + '_n'] = np.full( (len(pos_unmatch), 6), np.nan )
        elif ('FLAG' in col) or ('ISOAREA' in col):
            tab2[col + '_n'] = np.full(len(pos_unmatch), -1, dtype=int)  
        else:
            tab2[col + '_n'] = np.full(len(pos_unmatch), np.nan)
            

    #Neg only  
    tab3 = Table()
    for col in cols_out:
        tab3[col + '_n'] = neg_unmatch[col]
    for col in cols_out:
        if 'APER' in col:
            tab3[col] = np.full( (len(neg_unmatch), 6), np.nan )
        elif ('FLAG' in col) or ('ISOAREA' in col):
            tab3[col] = np.full(len(neg_unmatch), -1, dtype=int)  
        else:
            tab3[col] = np.full(len(neg_unmatch), np.nan)
    

    tab1 = tab1[cols_tot].copy()
    tab2 = tab2[cols_tot].copy()
    tab3 = tab3[cols_tot].copy()
    tab = vstack([tab1, tab2, tab3], join_type='exact')
    
    return tab    



def get_flux(tab_tot, auto=True, aper_ind=0):
    if auto:
        det_pos = ~np.isnan(tab_tot['FLUX_AUTO'].data)
        
        f = tab_tot['FLUX_AUTO'].data
        ferr = tab_tot['FLUXERR_AUTO'].data
        
        fn = tab_tot['FLUX_AUTO_n'].data
        ferrn = tab_tot['FLUXERR_AUTO_n'].data
        
    else:
        det_pos = ~np.isnan(tab_tot['FLUX_APER'].data[:, aper_ind])
        
        f = tab_tot['FLUX_APER'].data[:, aper_ind]
        ferr = tab_tot['FLUXERR_APER'].data[:, aper_ind]
        
        fn = tab_tot['FLUX_APER_n'].data[:, aper_ind]
        ferrn = tab_tot['FLUXERR_APER_n'].data[:, aper_ind]
        
    flux = np.zeros(len(tab_tot), dtype=float)
    flux[det_pos] = f[det_pos]
    flux[~det_pos] = fn[~det_pos]
    
    fluxerr = np.zeros(len(tab_tot), dtype=float)
    fluxerr[det_pos] = ferr[det_pos]
    fluxerr[~det_pos] = ferrn[~det_pos]
    
    return flux, fluxerr   


def get_radec(tab_tot):
    det_pos = ~np.isnan(tab_tot['ALPHA_J2000'].data)
    
    ra = tab_tot['ALPHA_J2000'].data
    dec = tab_tot['DELTA_J2000'].data
    
    ra_n = tab_tot['ALPHA_J2000_n'].data
    dec_n = tab_tot['DELTA_J2000_n'].data
    
    
    ra_out = np.zeros(len(tab_tot), dtype=float)
    ra_out[det_pos] = ra[det_pos]
    ra_out[~det_pos] = ra_n[~det_pos]
    
    dec_out = np.zeros(len(tab_tot), dtype=float)
    dec_out[det_pos] = dec[det_pos]
    dec_out[~det_pos] = dec_n[~det_pos]

    return ra_out, dec_out

def get_flux_all(dat_all, auto=True, aper_ind=0):
    
    if auto:
        det_pos = ~np.isnan(dat_all['FLUX_AUTO_Diff'].data)
        flux = np.full(len(dat_all), np.nan, dtype=float)
        fluxerr = np.full(len(dat_all), np.nan, dtype=float)
        flux[det_pos] = dat_all['FLUX_AUTO_Diff'].data[det_pos]
        fluxerr[det_pos] = dat_all['FLUXERR_AUTO_Diff'].data[det_pos]
        flux[~det_pos] = dat_all['FLUX_AUTO_n_Diff'].data[~det_pos]
        fluxerr[~det_pos] = dat_all['FLUXERR_AUTO_n_Diff'].data[~det_pos]
        
    else:
        det_pos = ~np.isnan(dat_all['FLUX_APER_Diff'].data[:, aper_ind])
        flux = np.full(len(dat_all), np.nan, dtype=float)
        fluxerr = np.full(len(dat_all), np.nan, dtype=float)
        flux[det_pos] = dat_all['FLUX_APER_Diff'].data[:, aper_ind][det_pos]
        fluxerr[det_pos] = dat_all['FLUXERR_APER_Diff'].data[:, aper_ind][det_pos]
        flux[~det_pos] = dat_all['FLUX_APER_n_Diff'].data[:, aper_ind][~det_pos]
        fluxerr[~det_pos] = dat_all['FLUXERR_APER_n_Diff'].data[:, aper_ind][~det_pos]

    return flux, fluxerr

def combine_refstack_nexus(diffdat_sub, diffdat_sfft, nexdat_edr, nexdat_refstack, nexdat_scistack, band='F200W', matching_radius=.3):
    ap_inds_nex = [0,1,3]
    ap_inds_diffstack = [0,1,2]
    
    #Match REFStack and SCIStack by ID
    match_func = partial(match_nexus_catalogs, cat_r=nexdat_refstack, cat_s=nexdat_scistack)
    inputs = list( range(len(nexdat_scistack)) )

    with WorkerPool(n_jobs=30) as pool:
        output = pool.map(match_func, inputs)

    inds1 = []
    inds2 = []
    for i in range(len(output)):
        if output[i] != (-1, -1):
            inds1.append(output[i][0])
            inds2.append(output[i][1])

    nexdat_refstack = nexdat_refstack[inds1].copy()
    nexdat_scistack = nexdat_scistack[inds2].copy()
    nexdat_refstack.sort('ID')
    nexdat_scistack.sort('ID')


    cols_in = ['ALPHA_J2000', 'DELTA_J2000', 'FLUX_AUTO', 'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER', 'ELONGATION', 'FLAGS', 'IMAFLAGS_ISO', 'CLASS_STAR', 'ISOAREA_IMAGE']
    cols_out = ['RA', 'DEC', 'FLUX_AUTO', 'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER', 'ELONGATION', 'FLAGS', 'IMAFLAGS_ISO', 'CLASS_STAR', 'ISOAREA_IMAGE'] 
    cols_out_diffsub = [col + '_DiffStackSub' for col in cols_out]
    cols_out_diffsfft = [col + '_DiffStackSFFT' for col in cols_out]

    # cols_in_nex_wide = []
    # cols_out_nex_wide = []
    # for col in nexdat_wide.colnames:
    #     if ('TOTAL' in col):
    #         continue
    #     if ('F' in col) and ('W' in col) and (not (band in col)):
    #         continue

    #     if ('F' in col) and ('W' in col) and (band in col):
    #         if ('_1' in col) or ('_2' in col) or ('_3' in col) or ('_4' in col) or ('_5' in col):
    #             continue
   
    #         cols_in_nex_wide.append(col)
    #         cols_out_nex_wide.append(col.replace('_{}'.format(band), '_Wide'))
    #     else:
    #         cols_in_nex_wide.append(col)
    #         cols_out_nex_wide.append(col + '_Wide')

    cols_in_nex_refstack = ['ALPHA_J2000', 'DELTA_J2000', 'FLUX_AUTO', 'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER', 'ELONGATION', 'FLAGS', 'IMAFLAGS_ISO', 'CLASS_STAR', 'ISOAREA_IMAGE', 'ID']
    cols_in_nex_scistack = cols_in_nex_refstack.copy()
    cols_out_nex_refstack = [x + '_REFStack' for x in ['RA', 'DEC', 'FLUX_AUTO', 'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER', 'ELONGATION', 'FLAGS', 'IMAFLAGS_ISO', 'CLASS_STAR', 'ISOAREA_IMAGE', 'ID']]
    cols_out_nex_scistack = [ x + '_SCIStack' for x in ['RA', 'DEC', 'FLUX_AUTO', 'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER', 'ELONGATION', 'FLAGS', 'IMAFLAGS_ISO', 'CLASS_STAR', 'ISOAREA_IMAGE', 'ID']]

    cols_tot = cols_out_diffsub + cols_out_diffsfft + cols_out_nex_refstack + cols_out_nex_scistack #+ cols_out_nex_wide
    
    
    #Combine the apeture fluxes into a single array for all apertures
    # flux_aper_w = np.full((len(nexdat_wide), 5), np.nan)
    # fluxerr_aper_w = np.full((len(nexdat_wide), 5), np.nan)
    flux_aper_ws = np.full((len(nexdat_refstack), 3), np.nan)
    fluxerr_aper_ws = np.full((len(nexdat_refstack), 3), np.nan)
    flux_aper_ds = np.full((len(nexdat_scistack), 3), np.nan)
    fluxerr_aper_ds = np.full((len(nexdat_scistack), 3), np.nan)
    for i, ind in enumerate(ap_inds_nex):
        # flux_aper_w[:, i] = nexdat_wide['FLUX_APER_{}_{}'.format(i + 1, band)].data
        # fluxerr_aper_w[:, i] = nexdat_wide['FLUXERR_APER_{}_{}'.format(i + 1, band)].data  
        
        if ind == 0:
            suffix = ''
        else:
            suffix = '_{}'.format(ind)
        
        flux_aper_ws[:, i] = nexdat_refstack['FLUX_APER' + suffix].data
        fluxerr_aper_ws[:, i] = nexdat_refstack['FLUXERR_APER' + suffix].data
        flux_aper_ds[:, i] = nexdat_scistack['FLUX_APER' + suffix].data
        fluxerr_aper_ds[:, i] = nexdat_scistack['FLUXERR_APER' + suffix].data

    #Add the fluxes to the nexus catalogs
    # nexdat_wide['FLUX_APER_{}'.format(band)] = flux_aper_w
    # nexdat_wide['FLUXERR_APER_{}'.format(band)] = fluxerr_aper_w
    nexdat_refstack['FLUX_APER'] = flux_aper_ws
    nexdat_refstack['FLUXERR_APER'] = fluxerr_aper_ws
    nexdat_scistack['FLUX_APER'] = flux_aper_ds
    nexdat_scistack['FLUXERR_APER'] = fluxerr_aper_ds



    #Combine the diffdat_sub and diffdat_sfft
    tab_diff = Table()
    for cin, cout in zip(cols_in, cols_out):
        tab_diff[cout + '_DiffStackSub'] = diffdat_sub[cin]
        tab_diff[cout + '_DiffStackSFFT'] = diffdat_sfft[cin]
        
    #Combine widestack and deepstack
    tab_stack = Table()
    for cin, cout in zip(cols_in_nex_refstack, cols_out_nex_refstack):
        tab_stack[cout] = nexdat_refstack[cin]
    for cin, cout in zip(cols_in_nex_scistack, cols_out_nex_scistack):
        tab_stack[cout] = nexdat_scistack[cin]
            

    #Match nexus EDR catalog to stacked catalogs (Only keep the matched sources)
    # coords_w = SkyCoord(nexdat_wide['RA'].data, nexdat_wide['DEC'].data, unit=(u.deg, u.deg))
    # coords_d = SkyCoord(tab_stack['RA_WideStack'].data, tab_stack['DEC_WideStack'].data, unit=(u.deg, u.deg))
    
    # idx, d2d, _ = coords_w.match_to_catalog_sky(coords_d)
    # mask = d2d.arcsec < matching_radius
    
    # matched_w = nexdat_wide[mask].copy()
    # matched_stack = tab_stack[idx[mask]].copy()
    
    # tab_nex = Table()
    # for cin, cout in zip(cols_in_nex_wide, cols_out_nex_wide):
    #     tab_nex[cout] = matched_w[cin]
    # for c in matched_stack.colnames:
    #     tab_nex[c] = matched_stack[c]


    #Match the diff table to the nexus catalogs
    coords_diff = SkyCoord(tab_diff['RA_DiffStackSub'].data, tab_diff['DEC_DiffStackSub'].data, unit=(u.deg, u.deg))
    coords_nex = SkyCoord(tab_stack['RA_REFStack'].data, tab_stack['DEC_REFStack'].data, unit=(u.deg, u.deg))
    
    idx, d2d, _ = coords_nex.match_to_catalog_sky(coords_diff)
    mask = d2d.arcsec < matching_radius
    
    matched_ind1 = np.argwhere(mask).flatten()
    matched_ind2 = idx[mask].flatten()
    matched_d2d = d2d[mask].arcsec.flatten()
    
    #Get rid of duplicate matches
    matched_ind1_nodup = []
    matched_ind2_nodup = []
    matched_d2d_nodup = []
    
    bad_ind = []
    for i in range(len(matched_ind1)):
        if i in bad_ind:
            continue
        
        repeat_ind = np.argwhere(matched_ind2 == matched_ind2[i]).flatten()
        n = len(repeat_ind)

        if n > 1:
            #Only take match with smallest matching distance
            d2d_arr = []
            for j in repeat_ind:
                d2d_arr.append(matched_d2d[j])
                
            min_ind = np.argmin(d2d_arr)
            matched_ind1_nodup.append(matched_ind1[repeat_ind[min_ind]])
            matched_ind2_nodup.append(matched_ind2[repeat_ind[min_ind]])
            matched_d2d_nodup.append(matched_d2d[repeat_ind[min_ind]])
            
            for j in repeat_ind:
                if j != min_ind:
                    bad_ind.append(j)

        else:
            matched_ind1_nodup.append(matched_ind1[i])
            matched_ind2_nodup.append(matched_ind2[i])
            matched_d2d_nodup.append(matched_d2d[i])

        
    print('\t Found {} duplicate matches'.format(len(bad_ind)))    
            
    matched_ind1 = np.array(matched_ind1_nodup)
    matched_ind2 = np.array(matched_ind2_nodup)
    matched_d2d = np.array(matched_d2d_nodup)


    matched_diff = tab_diff[matched_ind2].copy()
    matched_nex = tab_stack[matched_ind1].copy()

    tab_out = Table()
    for col in cols_out:
        tab_out[col + '_DiffStackSub'] = matched_diff[col + '_DiffStackSub']
        tab_out[col + '_DiffStackSFFT'] = matched_diff[col + '_DiffStackSFFT']
    # for col in cols_out_nex_wide:
    #     tab_out[col] = matched_nex[col]
    for col in cols_out_nex_refstack:
        tab_out[col] = matched_nex[col]
    for col in cols_out_nex_scistack:
        tab_out[col] = matched_nex[col]
    
    tab_out = tab_out[cols_tot].copy()
    
    #Turn fluxes into NaNs
    for col in tab_out.colnames:
        if 'FLUX' in col:
            mask = tab_out[col].data == -99.
            tab_out[col].data[mask] = np.nan
    
    return tab_out
    
    
    


def combine_stack_diffonly(diffdat, stackdat, matching_radius=.5):
    ra_diff, dec_diff = get_radec(diffdat)
    
    coords_diff = SkyCoord(ra_diff, dec_diff, unit=(u.deg, u.deg))
    coords_stack = SkyCoord(stackdat['RA_WideStack'].data, stackdat['DEC_WideStack'].data, unit=(u.deg, u.deg))
    
    idx, d2d, _ = coords_stack.match_to_catalog_sky(coords_diff)
    mask = d2d.arcsec < matching_radius
    
    matched_diff = diffdat[idx[mask]].copy()
    matched_stack = stackdat[mask].copy()
    unmatched_stack = stackdat[~mask].copy()
    
    
    cols_out = ['ALPHA_J2000', 'DELTA_J2000', 'FLUX_AUTO', 'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER', 'ELONGATION', 'FLAGS', 'IMAFLAGS_ISO', 'CLASS_STAR', 'ISOAREA_IMAGE']#, 'ISOAREAF_IMAGE']
    cols_out_stack = stackdat.colnames
    cols_out_diff = [col + '_Diff' for col in diffdat.colnames]
    cols_tot = cols_out_stack + cols_out_diff
    
    #Both
    tab1 = Table()
    for col in diffdat.colnames:
        tab1[col + '_Diff'] = matched_diff[col]
    for col in cols_out_stack:
        tab1[col] = matched_stack[col]
            
    tab1 = tab1[cols_tot].copy()
    
    
    #Stacked only
    tab2 = Table()
    for col in cols_out_stack:
        tab2[col] = unmatched_stack[col]
    for col in cols_out_diff:
        if 'APER' in col:
            tab2[col] = np.full((len(unmatched_stack), 6), np.nan)
        elif ('FLAG' in col) or ('ISOAREA' in col):
            tab2[col] = np.full(len(unmatched_stack), -1, dtype=int)  
        else:
            tab2[col] = np.full(len(unmatched_stack), np.nan)      


    tab1 = tab1[cols_tot].copy()
    tab2 = tab2[cols_tot].copy() 
    tab_tot = vstack([tab1, tab2], join_type='exact')

    return tab_tot


############################################################################################################
#FUNCTIONS FOR ALTERING CATALOG VALUES  ####################################################################
############################################################################################################

def check_if_in_mask(ra, dec, im_diff, wcs_diff, dx=3):
    coords = SkyCoord(ra, dec, unit=(u.deg, u.deg))
    
    dat = Cutout2D(im_diff, coords, wcs=wcs_diff, size=(dx * u.arcsec, dx * u.arcsec)).data
    
    if np.nansum(dat) == 0:
        return False
    else:
        return True    
    
def nexstack_to_ujy(f, zp):
    f_ujy = f * 10**( (23.9 - zp)/2.5 )
    return f_ujy    

def nexstack_to_ujy_err(ferr, zp):
    f_ujy_err = ferr * 10**( (23.9 - zp)/2.5 )
    return f_ujy_err

def get_dm(fdiff, fwide):
    return -2.5 * np.log10(1. + fdiff/fwide)

def get_dm_err(fdiff, fwide, ferr_diff, ferr_wide):
    return (5/2/np.log(10)/(fwide + fdiff)) * np.sqrt( (ferr_wide*fdiff/fwide)**2 + ferr_diff**2 )



def bkgsub_fluxes(im_diff, im_err, im_mask, x, y, fname_apcorr, band, r1=1, subtract_sky=True):
    
    r2 = 2*r1

    if band == 'F200W':
        ps_grid = .031
    elif band == 'F444W':
        ps_grid = .063
    
    
    circ_ap = CircularAperture(zip(x, y), r=r1/ps_grid)
    if subtract_sky:
        ann_ap = CircularAnnulus(zip(x, y), r_in=r1/ps_grid, r_out=r2/ps_grid)

    circ_npx = circ_ap.area
    if subtract_sky:
        ann_npx = ann_ap.area

    
    circ_stats = ApertureStats(im_diff, circ_ap, error=im_err, mask=im_mask, sum_method='exact')
    circ_fluxes = circ_stats.sum
    circ_fluxerrs = circ_stats.sum_err
    if subtract_sky:
        ann_stats = ApertureStats(im_diff, ann_ap, error=im_err, mask=im_mask, sum_method='exact')
        ann_fluxes = ann_stats.sum
        ann_fluxerrs = ann_stats.sum_err
        
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
        
        circ_skyfluxes = np.zeros_like(circ_fluxes_skysub)
        circ_skyfluxerrs = np.zeros_like(circ_fluxerrs_skysub)
    
    return circ_fluxes_skysub, circ_fluxerrs_skysub, circ_skyfluxes, circ_skyfluxerrs

################################################################################################################
#DEFINITIONS  ##################################################################################################
################################################################################################################

zp_f200w = 28.0865 #Wide01/Deep01 stacked catalogs

matching_radius = 0.1 #arcsec

names = ['wide', 'deep']
epochs = ['01', '01']
bands = ['F200W', 'F444W']

name_prefix = '{}{}_{}{}'.format(names[0], epochs[0], names[1], epochs[1])
prefix1 = '{}{}'.format(names[0], epochs[0])
prefix2 = '{}{}'.format(names[1], epochs[1])

maindir = '/data6/stone28/nexus/'

################################################################################################################################################################################################################################
################################################################################################################################################################################################################################
#READ IN DATA  #################################################################################################################################################################################################################
################################################################################################################################################################################################################################
################################################################################################################################################################################################################################

print('Reading in data')

#SFFT
# dat_f200w_sfft_p = Table.read('/data6/stone28/nexus/sfftsource_combined_nexus_wide01_deep01/output_pos/nexus_F200W_sfftdiff.cat')
# dat_f444w_sfft_p = Table.read('/data6/stone28/nexus/sfftsource_combined_nexus_wide01_deep01/output_pos/nexus_F444W_sfftdiff.cat')
# dat_f200w_sfft_n = Table.read('/data6/stone28/nexus/sfftsource_combined_nexus_wide01_deep01/output_neg/nexus_F200W_sfftdiff.cat')
# dat_f444w_sfft_n = Table.read('/data6/stone28/nexus/sfftsource_combined_nexus_wide01_deep01/output_neg/nexus_F444W_sfftdiff.cat')
# dat_f200w_sfft = combine_pos_neg(dat_f200w_sfft_p, dat_f200w_sfft_n, matching_radius=.1)
# dat_f444w_sfft = combine_pos_neg(dat_f444w_sfft_p, dat_f444w_sfft_n, matching_radius=.1)

dat_stack_f200w_sfft = Table.read(maindir + 'sfftsource_combined_nexus_{}/output_pos/nexus_{}_sfftdiff_stack.cat'.format(name_prefix, bands[0]))
dat_stack_f444w_sfft = Table.read(maindir + 'sfftsource_combined_nexus_{}/output_pos/nexus_{}_sfftdiff_stack.cat'.format(name_prefix, bands[1]))


#SUB
# dat_f200w_sub_p = Table.read('/data6/stone28/nexus/subsource_combined_nexus_wide01_deep01/output_pos/nexus_F200W_subdiff.cat')
# dat_f444w_sub_p = Table.read('/data6/stone28/nexus/subsource_combined_nexus_wide01_deep01/output_pos/nexus_F444W_subdiff.cat')
# dat_f200w_sub_n = Table.read('/data6/stone28/nexus/subsource_combined_nexus_wide01_deep01/output_neg/nexus_F200W_subdiff.cat')
# dat_f444w_sub_n = Table.read('/data6/stone28/nexus/subsource_combined_nexus_wide01_deep01/output_neg/nexus_F444W_subdiff.cat')
# dat_f200w_sub = combine_pos_neg(dat_f200w_sub_p, dat_f200w_sub_n, matching_radius=.1)
# dat_f444w_sub = combine_pos_neg(dat_f444w_sub_p, dat_f444w_sub_n, matching_radius=.1)

dat_stack_f200w_sub = Table.read(maindir + 'subsource_combined_nexus_{}/output_pos/nexus_{}_subdiff_stack.cat'.format(name_prefix, bands[0]))
dat_stack_f444w_sub = Table.read(maindir + 'subsource_combined_nexus_{}/output_pos/nexus_{}_subdiff_stack.cat'.format(name_prefix, bands[1]))


assert (dat_stack_f200w_sfft['ALPHA_J2000'].data == dat_stack_f200w_sub['ALPHA_J2000'].data).all()
assert (dat_stack_f444w_sfft['ALPHA_J2000'].data == dat_stack_f444w_sub['ALPHA_J2000'].data).all()

#NEXUS catalog
cat_nexus_wide = Table.read('/data3/web/nexus/edr/nircam/catalog/wide_ep1_01_src_catalog_v1.fits')

indir = '/data4/jwst/nexus/reduced_data/SEXtractor/catalogs/'
cat_nexus_widestack_f200w = Table.read(indir + 'nexus_full_{}_{}_catalog.fits'.format(prefix1, bands[0].lower()))
cat_nexus_widestack_f444w = Table.read(indir + 'nexus_full_{}_{}_catalog.fits'.format(prefix1, bands[1].lower()))
cat_nexus_deepstack_f200w = Table.read(indir + 'nexus_full_{}_{}_catalog.fits'.format(prefix2, bands[0].lower()))
cat_nexus_deepstack_f444w = Table.read(indir + 'nexus_full_{}_{}_catalog.fits'.format(prefix2, bands[1].lower()))

#Saturated sources
satdat_f200w = Table.read(maindir + 'sfft_nexus_{}_{}/output/nexus_{}_{}.saturated_sources.combined.cat'.format(name_prefix, bands[0], prefix2, bands[0]))
satdat_f444w = Table.read(maindir + 'sfft_nexus_{}_{}/output/nexus_{}_{}.saturated_sources.combined.cat'.format(name_prefix, bands[1], prefix2, bands[1]))

#Empty cutouts/segmaps
with fits.open(maindir + 'sfft_nexus_{}_{}/output/nexus_{}_{}.cutout_segmap.fits'.format(name_prefix, bands[0], prefix2, bands[0])) as hdul:
    segmap_f200w = hdul[0].data
    wcs_segmap_f200w = WCS(hdul[0].header)
with fits.open(maindir + 'sfft_nexus_{}_{}/output/nexus_{}_{}.cutout_segmap.fits'.format(name_prefix, bands[1], prefix2, bands[1])) as hdul:
    segmap_f444w = hdul[0].data
    wcs_segmap_f444w = WCS(hdul[0].header)

emptydat_f200w = Table.read(maindir + 'sfft_nexus_{}_{}/output/empty_cutouts.txt'.format(name_prefix, bands[0]), format='ascii')
emptydat_f444w = Table.read(maindir + 'sfft_nexus_{}_{}/output/empty_cutouts.txt'.format(name_prefix, bands[1]), format='ascii')

################################################################################################################
#COMBINE CATALOGS AND FIND SATURATED MASKED SOURCES  ###########################################################
################################################################################################################

print('Combining catalogs with NEXUS catalogs')

#Combine stacked with NEXUS
dat_f200w_stacked = combine_refstack_nexus(dat_stack_f200w_sub, dat_stack_f200w_sfft, cat_nexus_wide, cat_nexus_widestack_f200w, cat_nexus_deepstack_f200w, band=bands[0], matching_radius=matching_radius)
dat_f444w_stacked = combine_refstack_nexus(dat_stack_f444w_sub, dat_stack_f444w_sfft, cat_nexus_wide, cat_nexus_widestack_f444w, cat_nexus_deepstack_f444w, band=bands[1], matching_radius=matching_radius)

#Combine stacked+diffonly
# dat_f200w_sub_all = combine_stack_diffonly(dat_f200w_sub, dat_f200w_stacked, matching_radius=.5)
# dat_f444w_sub_all = combine_stack_diffonly(dat_f444w_sub, dat_f444w_stacked, matching_radius=.5)
# dat_f200w_sfft_all = combine_stack_diffonly(dat_f200w_sfft, dat_f200w_stacked, matching_radius=.5)
# dat_f444w_sfft_all = combine_stack_diffonly(dat_f444w_sfft, dat_f444w_stacked, matching_radius=.5)


print('Matching saturated sources to stacked catalogs')

#Match saturated sources
coords_sat = SkyCoord(satdat_f200w['RA'].data, satdat_f200w['DEC'].data, unit=(u.deg, u.deg))
coords_stack = SkyCoord(dat_f200w_stacked['RA_REFStack'].data, dat_f200w_stacked['DEC_REFStack'].data, unit=(u.deg, u.deg))

idx, d2d, _ = coords_stack.match_to_catalog_sky(coords_sat)
mask = d2d.arcsec < matching_radius

match_mask = np.zeros(len(dat_f200w_stacked), dtype=bool)
match_mask[mask] = True
dat_f200w_stacked['SATURATED'] = match_mask






coords_sat = SkyCoord(satdat_f444w['RA'].data, satdat_f444w['DEC'].data, unit=(u.deg, u.deg))
coords_stack = SkyCoord(dat_f444w_stacked['RA_REFStack'].data, dat_f444w_stacked['DEC_REFStack'].data, unit=(u.deg, u.deg))

idx, d2d, _ = coords_stack.match_to_catalog_sky(coords_sat)
mask = d2d.arcsec < matching_radius

match_mask = np.zeros(len(dat_f444w_stacked), dtype=bool)
match_mask[mask] = True
dat_f444w_stacked['SATURATED'] = match_mask
    
################################################################################################################
#REMOVE SOURCES WITHIN IMAGE MASKS  ############################################################################
################################################################################################################
print('Removing sources within image masks')

fname_submask_f200w = maindir + 'zogy_nexus_{}_{}/output/nexus_{}_{}.mask.fits'.format(name_prefix, bands[0], prefix2, bands[0])
fname_submask_f444w = maindir + 'zogy_nexus_{}_{}/output/nexus_{}_{}.mask.fits'.format(name_prefix, bands[1], prefix2, bands[1])

fname_sfftmask_f200w = maindir + 'sfft_nexus_{}_{}/output/nexus_{}_{}.sfftdiff.decorr.mask.combined.fits'.format(name_prefix, bands[0], prefix2, bands[0])
fname_sfftmask_f444w = maindir + 'sfft_nexus_{}_{}/output/nexus_{}_{}.sfftdiff.decorr.mask.combined.fits'.format(name_prefix, bands[1], prefix2, bands[1])

fname_sfftdiff_f200w = maindir + 'sfft_nexus_{}_{}/output/nexus_{}_{}.sfftdiff.decorr.combined.fits'.format(name_prefix, bands[0], prefix2, bands[0])
fname_sfftdiff_f444w = maindir + 'sfft_nexus_{}_{}/output/nexus_{}_{}.sfftdiff.decorr.combined.fits'.format(name_prefix, bands[1], prefix2, bands[1])
fname_subdiff_f200w = maindir + 'zogy_nexus_{}_{}/output/nexus_{}_{}.subdiff.fits'.format(name_prefix, bands[0], prefix2, bands[0])
fname_subdiff_f444w = maindir + 'zogy_nexus_{}_{}/output/nexus_{}_{}.subdiff.fits'.format(name_prefix, bands[1], prefix2, bands[1])

empty_mask_f200w = np.zeros(len(dat_f200w_stacked), dtype=bool)
empty_mask_f444w = np.zeros(len(dat_f444w_stacked), dtype=bool)

with fits.open(fname_sfftdiff_f200w) as hdul:
    im_sfft_f200w = hdul[0].data
    wcs_f200w = WCS(hdul[0].header)
with fits.open(fname_sfftdiff_f444w) as hdul:
    im_sfft_f444w = hdul[0].data
    wcs_f444w = WCS(hdul[0].header)
with fits.open(fname_subdiff_f200w) as hdul:
    im_sub_f200w = hdul[0].data
with fits.open(fname_subdiff_f444w) as hdul:
    im_sub_f444w = hdul[0].data
    
with fits.open(fname_submask_f200w) as hdul:
    submask_f200w = hdul[0].data.astype(bool)
with fits.open(fname_submask_f444w) as hdul:
    submask_f444w = hdul[0].data.astype(bool)
with fits.open(fname_sfftmask_f200w) as hdul:
    sfftmask_f200w = hdul[0].data.astype(bool)
with fits.open(fname_sfftmask_f444w) as hdul:
    sfftmask_f444w = hdul[0].data.astype(bool)

xvals, yvals = wcs_f200w.wcs_world2pix(dat_f200w_stacked['RA_REFStack'].data, dat_f200w_stacked['DEC_REFStack'].data, 0)
xvals = xvals.astype(int)
yvals = yvals.astype(int)

#Outside image bounds
xvals[yvals > sfftmask_f200w.shape[0] - 1] = -1
yvals[xvals > sfftmask_f200w.shape[1] - 1] = -1
xvals[yvals < 0] = -1
yvals[xvals < 0] = -1

#Find sources in empty cutouts (ones that failed)
cutout_labels = segmap_f200w[yvals, xvals]
empty_mask_f200w = np.array([ x in emptydat_f200w['LABEL'].data for x in cutout_labels ])

#In mask
bad_mask_f200w_1 = np.zeros(len(dat_f200w_stacked), dtype=bool)
bad_mask_f200w_1[~empty_mask_f200w] = sfftmask_f200w[yvals, xvals][~empty_mask_f200w] #For non-empty cutouts, use sfft mask
bad_mask_f200w_1[empty_mask_f200w] = submask_f200w[yvals, xvals][empty_mask_f200w] #For empty cutouts, use sub mask
bad_mask_f200w_1[ xvals<0 ] = True
bad_mask_f200w_1[ yvals<0 ] = True



xvals, yvals = wcs_f444w.wcs_world2pix(dat_f444w_stacked['RA_REFStack'].data, dat_f444w_stacked['DEC_REFStack'].data, 0)
xvals = xvals.astype(int)
yvals = yvals.astype(int)

#Outside image bounds
xvals[yvals > sfftmask_f444w.shape[0] - 1] = -1
yvals[xvals > sfftmask_f444w.shape[1] - 1] = -1
xvals[yvals < 0] = -1
yvals[xvals < 0] = -1

#Find sources in empty cutouts (ones that failed)
cutout_labels = segmap_f444w[yvals, xvals]
empty_mask_f444w = np.array([ x in emptydat_f444w['LABEL'].data for x in cutout_labels ])

#In mask
bad_mask_f444w_1 = np.zeros(len(dat_f444w_stacked), dtype=bool)
bad_mask_f444w_1[~empty_mask_f444w] = sfftmask_f444w[yvals, xvals][~empty_mask_f444w] #For non-empty cutouts, use sfft mask
bad_mask_f444w_1[empty_mask_f444w] = submask_f444w[yvals, xvals][empty_mask_f444w] #For empty cutouts, use sub mask
bad_mask_f444w_1[ yvals<0 ] = True
bad_mask_f444w_1[ xvals<0 ] = True






dat_f444w_stacked = dat_f444w_stacked[~bad_mask_f444w_1].copy()
dat_f200w_stacked = dat_f200w_stacked[~bad_mask_f200w_1].copy()


# func_f200w = partial(check_if_in_mask, im_diff=im_f200w, wcs_diff=wcs_f200w, dx=3)
# func_f444w = partial(check_if_in_mask, im_diff=im_f444w, wcs_diff=wcs_f444w, dx=3)
# bad_mask_f200w_2 = p_map(func_f200w, dat_f200w_stacked['RA_WideStack'].data, dat_f200w_stacked['DEC_WideStack'].data, num_cpus=64)
# bad_mask_f444w_2 = p_map(func_f444w, dat_f444w_stacked['RA_WideStack'].data, dat_f444w_stacked['DEC_WideStack'].data, num_cpus=64)



# dat_f200w_stacked = dat_f200w_stacked[~bad_mask_f200w_2].copy()
# dat_f444w_stacked = dat_f444w_stacked[~bad_mask_f444w_2].copy()

dat_all = [dat_f200w_stacked, dat_f444w_stacked]

################################################################################################################
#FIX FLUX UNITS (W/ ZEROPOINT)  ################################################################################
################################################################################################################

print('Converting Wide/Deep stacked fluxes to uJy')

cols = ['FLUX_AUTO', 'FLUX_APER']
cols_err = ['FLUXERR_AUTO', 'FLUXERR_APER']
suffixes = ['_REFStack', '_SCIStack']

for n in range(len(dat_all)):
    for c, ce in zip(cols, cols_err):
        for suffix in suffixes:
            dat_all[n][c + suffix] = nexstack_to_ujy(dat_all[n][c + suffix].data, zp=zp_f200w)
            dat_all[n][ce + suffix] = nexstack_to_ujy_err(dat_all[n][ce + suffix].data, zp=zp_f200w)

################################################################################################################
#GET BACKGROUND-SUBTRACTED FLUXES  #############################################################################
################################################################################################################

# print('Getting background-subtracted aperture fluxes')

# bands = ['F200W', 'F444W']
# difftypes = ['sfft', 'sub']
# im_names = ['wide', 'deep']
# apers = [0.2, 0.3, 0.5] #arcsec

# #Wide/Deep images
# for n, b in enumerate(bands):
    
#     dat_all[n]['FLUX_APER_BKGSUB_WideStack'] = np.full((len(dat_all[n]), len(apers)), np.nan)
#     dat_all[n]['FLUXERR_APER_BKGSUB_WideStack'] = np.full((len(dat_all[n]), len(apers)), np.nan)
#     dat_all[n]['FLUX_APER_BKGSUB_DeepStack'] = np.full((len(dat_all[n]), len(apers)), np.nan)
#     dat_all[n]['FLUXERR_APER_BKGSUB_DeepStack'] = np.full((len(dat_all[n]), len(apers)), np.nan)
    
#     ra = dat_all[n]['RA_WideStack'].data
#     dec = dat_all[n]['DEC_WideStack'].data  
    
#     for i, name in enumerate(im_names):
#         print('\t {} {}'.format(b, name))

#         fname = '/data6/stone28/nexus/sfft_nexus_wide01_deep01_{}/input/nexus_{}01_{}.fits'.format(b, name, b)
#         fname_err = '/data6/stone28/nexus/sfft_nexus_wide01_deep01_{}/noise/nexus_{}01_{}.noise.fits'.format(b, name, b)
#         fname_mask1 = '/data6/stone28/nexus/sfft_nexus_wide01_deep01_{}/input/nexus_wide01_{}.maskin.fits'.format(b, b)
#         fname_mask2 = '/data6/stone28/nexus/sfft_nexus_wide01_deep01_{}/input/nexus_deep01_{}.maskin.fits'.format(b, b)


#         with fits.open(fname) as hdul:
#             im = hdul[0].data
#             wcs = WCS(hdul[0].header)
#         with fits.open(fname_err) as hdul:
#             im_err = hdul[0].data
            
#         with fits.open(fname_mask1) as hdul:
#             im_mask1 = hdul[0].data.astype(bool)
#         with fits.open(fname_mask2) as hdul:
#             im_mask2 = hdul[0].data.astype(bool)
            
#         im_mask = im_mask1 | im_mask2 | np.isnan(im) | (im == 0.)

            
#         xvals, yvals = wcs.all_world2pix(ra, dec, 0)
            
#         for j, r1 in enumerate(apers):
#             fname_apcorr = 'aperture_corrections_deep01_r{:.2f}.txt'.format(r1)
            
#             f, ferr, skyflux, skufluxerr = bkgsub_fluxes(im, im_err, im_mask, xvals, yvals, None, b, r1=r1, subtract_sky=True)
#             #Only use sky value
            
#             corrdat = Table.read(fname_apcorr, format='ascii.commented_header')
#             ap_corr = corrdat['ApertureCorrection'][corrdat['Band'] == b]
#             ann_corr = corrdat['SkyCorrection'][corrdat['Band'] == b]
            
#             ap_corr = 1.
#             ann_corr = 1.
            
#             if i == 0:
#                 dat_all[n]['FLUX_APER_BKGSUB_WideStack'][:,j] = dat_all[n]['FLUX_APER_WideStack'][:,j].data*ap_corr - skyflux*ann_corr
#                 dat_all[n]['FLUXERR_APER_BKGSUB_WideStack'][:,j] = np.sqrt(dat_all[n]['FLUXERR_APER_WideStack'][:,j].data**2 * ap_corr**2 + skufluxerr**2 * ann_corr**2)
#             else:
#                 dat_all[n]['FLUX_APER_BKGSUB_DeepStack'][:,j] = dat_all[n]['FLUX_APER_DeepStack'][:,j].data*ap_corr - skyflux*ann_corr
#                 dat_all[n]['FLUXERR_APER_BKGSUB_DeepStack'][:,j] = np.sqrt(dat_all[n]['FLUXERR_APER_DeepStack'][:,j].data**2 * ap_corr**2 + skufluxerr**2 * ann_corr**2)


# # Diff images
# for n, b in enumerate(bands):
#     dat_all[n]['FLUX_APER_BKGSUB_DiffStackSFFT'] = np.full((len(dat_all[n]), len(apers)), np.nan)
#     dat_all[n]['FLUXERR_APER_BKGSUB_DiffStackSFFT'] = np.full((len(dat_all[n]), len(apers)), np.nan)
#     dat_all[n]['FLUX_APER_BKGSUB_DiffStackSub'] = np.full((len(dat_all[n]), len(apers)), np.nan)
#     dat_all[n]['FLUXERR_APER_BKGSUB_DiffStackSub'] = np.full((len(dat_all[n]), len(apers)), np.nan)
    
#     dat_all[n]['FLUX_APER_PHOT_BKGSUB_DiffStackSFFT'] = np.full((len(dat_all[n]), len(apers)), np.nan)
#     dat_all[n]['FLUXERR_APER_PHOT_BKGSUB_DiffStackSFFT'] = np.full((len(dat_all[n]), len(apers)), np.nan)
#     dat_all[n]['FLUX_APER_PHOT_BKGSUB_DiffStackSub'] = np.full((len(dat_all[n]), len(apers)), np.nan)
#     dat_all[n]['FLUXERR_APER_PHOT_BKGSUB_DiffStackSub'] = np.full((len(dat_all[n]), len(apers)), np.nan)
    
#     ra = dat_all[n]['RA_DiffStackSub'].data
#     dec = dat_all[n]['DEC_DiffStackSub'].data  
    
#     for i, dt in enumerate(difftypes):
#         print('\t {} {}'.format(b, dt))
        
#         if dt == 'sfft':
#             prefix = 'sfft'
#             prefix2 = '.sfftdiff'
            
#             # suffix = '_CC'
#             suffix2 = '.decorr'
#             suffix3 = '.combined'
#         elif dt == 'sub':
#             prefix = 'zogy'
#             prefix2 = ''
            
#             # suffix = ''
#             suffix2 = ''
#             suffix3 = ''

#         fname_diff = '/data6/stone28/nexus/{}_nexus_wide01_deep01_{}/output/nexus_deep01_{}.{}diff{}{}.fits'.format(prefix, b, b, dt, suffix2, suffix3)
#         fname_differr = '/data6/stone28/nexus/{}_nexus_wide01_deep01_{}/output/nexus_deep01_{}{}{}.noise{}.fits'.format(prefix, b, b, prefix2, suffix2, suffix3)
#         fname_diffmask = '/data6/stone28/nexus/{}_nexus_wide01_deep01_{}/output/nexus_deep01_{}{}{}.mask{}.fits'.format(prefix, b, b, prefix2, suffix2, suffix3)
        
        
#         with fits.open(fname_diff) as hdul:
#             im_diff = hdul[0].data
#             wcs = WCS(hdul[0].header)
#         with fits.open(fname_differr) as hdul:
#             im_err = hdul[0].data
#         with fits.open(fname_diffmask) as hdul:
#             im_mask = hdul[0].data.astype(bool)
            
#         xvals, yvals = wcs.all_world2pix(ra, dec, 0)
            
#         for j, r1 in enumerate(apers):
#             fname_apcorr = 'aperture_corrections_deep01_r{:.2f}.txt'.format(r1)
            
#             f, ferr, skyflux, skyfluxerr = bkgsub_fluxes(im_diff, im_err, im_mask, xvals, yvals, None, b, r1=r1, subtract_sky=True)
#             #Only use sky value
    
#             # corrdat = Table.read(fname_apcorr, format='ascii.commented_header')
#             # ap_corr = corrdat['ApertureCorrection'][corrdat['Band'] == b]
#             # ann_corr = corrdat['SkyCorrection'][corrdat['Band'] == b]
            
#             ap_corr = 1.
#             ann_corr = 1.
            
#             if i == 0:
#                 dat_all[n]['FLUX_APER_BKGSUB_DiffStackSFFT'][:,j] = dat_all[n]['FLUX_APER_DiffStackSFFT'][:,j].data*ap_corr - skyflux*ann_corr
#                 dat_all[n]['FLUXERR_APER_BKGSUB_DiffStackSFFT'][:,j] = np.sqrt(dat_all[n]['FLUXERR_APER_DiffStackSFFT'][:,j].data**2 * ap_corr**2 + skyfluxerr**2 * ann_corr**2)
                
#                 dat_all[n]['FLUX_APER_PHOT_BKGSUB_DiffStackSFFT'][:,j] = f
#                 dat_all[n]['FLUXERR_APER_PHOT_BKGSUB_DiffStackSFFT'][:,j] = ferr
                
#             else:
#                 dat_all[n]['FLUX_APER_BKGSUB_DiffStackSub'][:,j] = dat_all[n]['FLUX_APER_DiffStackSub'][:,j].data*ap_corr - skyflux*ann_corr
#                 dat_all[n]['FLUXERR_APER_BKGSUB_DiffStackSub'][:,j] = np.sqrt(dat_all[n]['FLUXERR_APER_DiffStackSub'][:,j].data**2 * ap_corr**2 + skyfluxerr**2 * ann_corr**2)
                
#                 dat_all[n]['FLUX_APER_PHOT_BKGSUB_DiffStackSub'][:,j] = f
#                 dat_all[n]['FLUXERR_APER_PHOT_BKGSUB_DiffStackSub'][:,j] = ferr


################################################################################################################
#ADD DELTA_M  ##################################################################################################
################################################################################################################
#Add Delta m columns

print('Getting DM')

cols = ['FLUX_AUTO', 'FLUX_APER']
cols_err = ['FLUXERR_AUTO', 'FLUXERR_APER']
suffixes = ['_DiffStackSub', '_DiffStackSFFT']# '_BKGSUB_DiffStackSub', '_BKGSUB_DiffStackSFFT', '_PHOT_BKGSUB_DiffStackSub', '_PHOT_BKGSUB_DiffStackSFFT']

for n, b in enumerate(bands):


    for c, ce in zip(cols, cols_err):
        
        for suffix in suffixes:
            if ('AUTO' in c) and ('BKGSUB' in suffix):
                continue
            
            # if 'BKGSUB' in suffix:
            #     ap_inds1 = [0,1,2]
            #     ap_inds2 = [0,1,2]
            # else:
            #     ap_inds1 = ap_inds_nex
            #     ap_inds2 = ap_inds_diff
            
            ap_inds1 = ap_inds2 = [0,1,2]
            
        
            #Wide/Deep
            if 'AUTO' in c:
                fr = dat_all[n][c + '_REFStack'].data.copy()
                ferr_r = dat_all[n][ce + '_REFStack'].data.copy()
                # fs = dat_all[n][c + '_SCIStack'].data.copy()
                # ferr_s = dat_all[n][ce + '_SCIStack'].data.copy()
            elif 'APER' in c:        
                fr = dat_all[n][c + '_REFStack'].data[:, ap_inds1].copy()
                ferr_r = dat_all[n][ce + '_REFStack'].data[:, ap_inds1].copy()
                # fs = dat_all[n][c + '_SCIStack'].data[:, ap_inds1].copy()
                # ferr_s = dat_all[n][ce + '_SCIStack'].data[:, ap_inds1].copy()

            
            #Diff
            if 'AUTO' in c:
                fdiff = dat_all[n][c + suffix].data.copy()
                ferr_diff = dat_all[n][ce + suffix].data.copy()
                col_out = c.replace('FLUX', 'DM') + suffix
                col_err_out = ce.replace('FLUXERR', 'DMERR') + suffix
            elif 'APER' in c:
                fdiff = dat_all[n][c + suffix].data[:, ap_inds2].copy()
                ferr_diff = dat_all[n][ce + suffix].data[:, ap_inds2].copy()
                col_out = c.replace('FLUX', 'DM') + suffix
                col_err_out = ce.replace('FLUXERR', 'DMERR') + suffix
                
            
            dm = get_dm(fdiff, fr)
            dm_err = get_dm_err(fdiff, fr, ferr_diff, ferr_r)
            
            dat_all[n][col_out] = dm
            dat_all[n][col_err_out] = dm_err
            
            
            del fdiff, ferr_diff, dm, dm_err
            
        del fr, ferr_r#, fs, ferr_s


################################################################################################################
#GET "BEST" VALUES FROM DIFF IMAGES  ###########################################################################
################################################################################################################

print('Choosing "best" catalog values')

choice_f200w = np.zeros( (len(dat_all[0]), 4), dtype=int) #1=SFFT, 0=SUB
choice_f444w = np.zeros( (len(dat_all[1]), 4), dtype=int)
# choice_f200w_bkgsub = np.zeros( (len(dat_all[0]), 3), dtype=int)
# choice_f444w_bkgsub = np.zeros( (len(dat_all[1]), 3), dtype=int)
# choice_f200w_phot_bkgsub = np.zeros( (len(dat_all[0]), 3), dtype=int)
# choice_f444w_phot_bkgsub = np.zeros( (len(dat_all[1]), 3), dtype=int)

x_f200w, y_f200w = wcs_segmap_f200w.wcs_world2pix(dat_all[0]['RA_DiffStackSFFT'].data, dat_all[0]['DEC_DiffStackSFFT'], 0)
x_f444w, y_f444w = wcs_segmap_f444w.wcs_world2pix(dat_all[1]['RA_DiffStackSFFT'].data, dat_all[1]['DEC_DiffStackSFFT'], 0)

x_f200w = x_f200w.astype(int)
y_f200w = y_f200w.astype(int)
x_f444w = x_f444w.astype(int)
y_f444w = y_f444w.astype(int)

remove_mask_f200w = np.zeros(len(dat_all[0]), dtype=bool)
remove_mask_f444w = np.zeros(len(dat_all[1]), dtype=bool)
    
for i in range(len(dat_all[0])):
    #See which cutout it is in
    cutout_i = segmap_f200w[y_f200w[i], x_f200w[i]]
    
    dx = 3
    dy = 3
    # sfftmask_cutout = sfftmask_f200w[y_f200w[i]-dy:y_f200w[i]+dy+1, x_f200w[i]-dx:x_f200w[i]+dx+1]
    # submask_cutout = submask_f200w[y_f200w[i]-dy:y_f200w[i]+dy+1, x_f200w[i]-dx:x_f200w[i]+dx+1]
    sfft_cutout = im_sfft_f200w[y_f200w[i]-dy:y_f200w[i]+dy+1, x_f200w[i]-dx:x_f200w[i]+dx+1]
    sub_cutout = im_sub_f200w[y_f200w[i]-dy:y_f200w[i]+dy+1, x_f200w[i]-dx:x_f200w[i]+dx+1]
    
    nbad = ( np.isnan(sfft_cutout) | (sfft_cutout == 0) | np.isnan(sub_cutout) | (sub_cutout == 0)).sum()
    nbad_sfft = ( np.isnan(sfft_cutout) | (sfft_cutout == 0)).sum()
    nbad_sub = ( np.isnan(sub_cutout) | (sub_cutout == 0)).sum()

    badpx_sfft = np.any(np.isnan(sfft_cutout)) or np.any(sfft_cutout == 0)
    badpx_sub = np.any(np.isnan(sub_cutout)) or np.any(sub_cutout == 0)
    
    remove_mask_f200w[i] = badpx_sfft and badpx_sub
    
    #If cutout is empty, set to SUB
    if cutout_i in emptydat_f200w['LABEL'].data:
        choice_f200w[i,:] = 0
        # choice_f200w_bkgsub[i,:] = 0
        # choice_f200w_phot_bkgsub[i,:] = 0
    
    elif badpx_sfft and (not badpx_sub): #If SFFT cutout has bad pixels, but SUB doesn't, choose SUB
        choice_f200w[i,:] = 0
        # choice_f200w_bkgsub[i,:] = 0
        # choice_f200w_phot_bkgsub[i,:] = 0
    
    else:

        if np.abs(dat_all[0]['DM_AUTO_DiffStackSub'][i]) > np.abs(dat_all[0]['DM_AUTO_DiffStackSFFT'][i]):
            choice_f200w[i, 0] = 1
            
        for j in range(3):
            if np.abs(dat_all[0]['DM_APER_DiffStackSub'][i,j]) > np.abs(dat_all[0]['DM_APER_DiffStackSFFT'][i,j]):
                choice_f200w[i, j+1] = 1

            # if np.abs(dat_all[0]['FLUX_APER_BKGSUB_DiffStackSub'][i,j]) > np.abs(dat_all[0]['FLUX_APER_BKGSUB_DiffStackSFFT'][i,j]):
            #     choice_f200w_bkgsub[i, j] = 1

            # if np.abs(dat_all[0]['FLUX_APER_PHOT_BKGSUB_DiffStackSub'][i,j]) > np.abs(dat_all[0]['FLUX_APER_PHOT_BKGSUB_DiffStackSFFT'][i,j]):
            #     choice_f200w_phot_bkgsub[i, j] = 1


for i in range(len(dat_all[1])):
    #See which cutout it is in
    cutout_i = segmap_f444w[y_f444w[i], x_f444w[i]]
    
    dx = 2
    dy = 2
    # sfftmask_cutout = sfftmask_f444w[y_f444w[i]-dy:y_f444w[i]+dy+1, x_f444w[i]-dx:x_f444w[i]+dx+1]
    # submask_cutout = submask_f444w[y_f444w[i]-dy:y_f444w[i]+dy+1, x_f444w[i]-dx:x_f444w[i]+dx+1]
    

    sfft_cutout = im_sfft_f444w[y_f444w[i]-dy:y_f444w[i]+dy+1, x_f444w[i]-dx:x_f444w[i]+dx+1]
    sub_cutout = im_sub_f444w[y_f444w[i]-dy:y_f444w[i]+dy+1, x_f444w[i]-dx:x_f444w[i]+dx+1]

    nbad = ( np.isnan(sfft_cutout) | (sfft_cutout == 0) | np.isnan(sub_cutout) | (sub_cutout == 0)).sum()
    nbad_sfft = ( np.isnan(sfft_cutout) | (sfft_cutout == 0)).sum()
    nbad_sub = ( np.isnan(sub_cutout) | (sub_cutout == 0)).sum()

    badpx_sfft = np.any(np.isnan(sfft_cutout)) or np.any(sfft_cutout == 0)
    badpx_sub = np.any(np.isnan(sub_cutout)) or np.any(sub_cutout == 0)


    remove_mask_f444w[i] = badpx_sfft and badpx_sub

    #If cutout is empty, set to SUB
    if cutout_i in emptydat_f444w['LABEL'].data:
        choice_f444w[i,:] = 0
        # choice_f444w_bkgsub[i,:] = 0
        # choice_f444w_phot_bkgsub[i,:] = 0

    elif badpx_sfft and (not badpx_sub): #If SFFT cutout has bad pixels, but SUB doesn't, choose SUB
        choice_f444w[i,:] = 0
        # choice_f444w_bkgsub[i,:] = 0
        # choice_f444w_phot_bkgsub[i,:] = 0

    else:
    
        if np.abs(dat_all[1]['DM_AUTO_DiffStackSub'][i]) > np.abs(dat_all[1]['DM_AUTO_DiffStackSFFT'][i]):
            choice_f444w[i, 0] = 1
            
        for j in range(3):
            if np.abs(dat_all[1]['DM_APER_DiffStackSub'][i,j]) > np.abs(dat_all[1]['DM_APER_DiffStackSFFT'][i,j]):
                choice_f444w[i, j+1] = 1
                
            # if np.abs(dat_f444w_stacked['FLUX_APER_BKGSUB_DiffStackSub'][i,j]) > np.abs(dat_f444w_stacked['FLUX_APER_BKGSUB_DiffStackSFFT'][i,j]):
            #     choice_f444w_bkgsub[i, j] = 1
                
            # if np.abs(dat_f444w_stacked['FLUX_APER_PHOT_BKGSUB_DiffStackSub'][i,j]) > np.abs(dat_f444w_stacked['FLUX_APER_PHOT_BKGSUB_DiffStackSFFT'][i,j]):
            #     choice_f444w_phot_bkgsub[i, j] = 1
    
    
choice_f200w = choice_f200w.astype(bool)
choice_f444w = choice_f444w.astype(bool)
# choice_f200w_bkgsub = choice_f200w_bkgsub.astype(bool)
# choice_f444w_bkgsub = choice_f444w_bkgsub.astype(bool)
# choice_f200w_phot_bkgsub = choice_f200w_phot_bkgsub.astype(bool)
# choice_f444w_phot_bkgsub = choice_f444w_phot_bkgsub.astype(bool)
    

for i in range(len(dat_all)):
    dat_all[i]['DM_FINAL_TYPE'] = np.full( (len(dat_all[i]), 4), 'SUB', dtype='U4')
    # dat_all[i]['DM_BKGSUB_FINAL_TYPE'] = np.full( (len(dat_all[i]), 3), 'SUB', dtype='U4')
    # dat_all[i]['DM_PHOT_BKGSUB_FINAL_TYPE'] = np.full( (len(dat_all[i]), 3), 'SUB', dtype='U4')

for n in range(len(dat_all)):
    if n == 0:
        a1 = choice_f200w.copy()
        # a2 = choice_f200w_bkgsub.copy()
        # a3 = choice_f200w_phot_bkgsub.copy()
    elif n == 1:
        a1 = choice_f444w.copy()
        # a2 = choice_f444w_bkgsub.copy()
        # a3 = choice_f444w_phot_bkgsub.copy()
    
    for i in range(4):
        dat_all[n]['DM_FINAL_TYPE'][:,i][a1[:,i]] = 'SFFT'
    # for i in range(3):
    #     dat_all[n]['DM_BKGSUB_FINAL_TYPE'][:,i][a2[:,i]] = 'SFFT'
    #     dat_all[n]['DM_PHOT_BKGSUB_FINAL_TYPE'][:,i][a3[:,i]] = 'SFFT'

################################################################################################################
#ASSIGN "BEST" DM,mag,flux FROM DIFF IMAGES  ####################################################################
################################################################################################################

print('Assigning "best" DM, mag, flux from diff images')

cols_new = ['DM_FINAL', 'DIFF_FLUX_FINAL', 'DIFF_MAG_FINAL']
cols_new_err = ['DMERR_FINAL', 'DIFF_FLUXERR_FINAL']
# cols_new_bkgsub = ['DM_BKGSUB_FINAL', 'DIFF_FLUX_BKGSUB_FINAL', 'DIFF_MAG_BKGSUB_FINAL']
# cols_new_bkgsub_err = ['DMERR_BKGSUB_FINAL', 'DIFF_FLUXERR_BKGSUB_FINAL']
# cols_new_phot_bkgsub = ['DM_PHOT_BKGSUB_FINAL', 'DIFF_FLUX_PHOT_BKGSUB_FINAL', 'DIFF_MAG_PHOT_BKGSUB_FINAL']
# cols_new_phot_bkgsub_err = ['DMERR_PHOT_BKGSUB_FINAL', 'DIFF_FLUXERR_PHOT_BKGSUB_FINAL']

for n in range(len(dat_all)):

    #[AUTO, APER1, APER2, APER3]
    for c in cols_new:
        dat_all[n][c] = np.full( (len(dat_all[n]), 4), np.nan)
    for c in cols_new_err:
        dat_all[n][c] = np.full( (len(dat_all[n]), 4), np.nan)
        
    #[APER1, APER2, APER3]
    # for c in cols_new_bkgsub:
    #     dat_all[n][c] = np.full( (len(dat_all[n]), 3), np.nan)
    # for c in cols_new_bkgsub_err:
    #     dat_all[n][c] = np.full( (len(dat_all[n]), 3), np.nan)
    # for c in cols_new_phot_bkgsub:
    #     dat_all[n][c] = np.full( (len(dat_all[n]), 3), np.nan)
    # for c in cols_new_phot_bkgsub_err:
    #     dat_all[n][c] = np.full( (len(dat_all[n]), 3), np.nan)

dtypes = ['']#, '_BKGSUB', '_PHOT_BKGSUB']
for n in range(len(dat_all)):    
    
    for i in range(4):
        
        #APER
        if i > 0:
            for dt in dtypes:
                
                if 'BKGSUB' in dt:
                    suffix = dt
                    ind_in = i-1
                    ind_out = i-1
                    
                else:
                    suffix = ''
                    ind_in = i-1
                    ind_out = i
                    
                if dt == '':
                    cols1 = cols_new
                    cols2 = cols_new_err
                # elif dt == '_BKGSUB':
                #     cols1 = cols_new_bkgsub
                #     cols2 = cols_new_bkgsub_err
                # elif dt == '_PHOT_BKGSUB':
                #     cols1 = cols_new_phot_bkgsub
                #     cols2 = cols_new_phot_bkgsub_err
                    

                choice_arr = (dat_all[n]['DM{}_FINAL_TYPE'.format(suffix)][:,ind_out] == 'SFFT')
                    
                    
                for dt in ['SFFT', 'Sub']:
                    if dt == 'SFFT':
                        mask = choice_arr.copy()
                    elif dt == 'Sub':
                        mask = (~choice_arr).copy()
                        
                    suffix2 = '_DiffStack{}'.format(dt)

                    dat_all[n]['DM{}_FINAL'.format(suffix)][:,ind_out][mask] = dat_all[n]['DM_APER{}{}'.format(suffix,suffix2)][:,ind_in][mask]
                    dat_all[n]['DMERR{}_FINAL'.format(suffix)][:,ind_out][mask] = dat_all[n]['DMERR_APER{}{}'.format(suffix,suffix2)][:,ind_in][mask]
                    dat_all[n]['DIFF_FLUX{}_FINAL'.format(suffix)][:,ind_out][mask] = dat_all[n]['FLUX_APER{}{}'.format(suffix,suffix2)][:,ind_in][mask]
                    dat_all[n]['DIFF_FLUXERR{}_FINAL'.format(suffix)][:,ind_out][mask] = dat_all[n]['FLUXERR_APER{}{}'.format(suffix,suffix2)][:,ind_in][mask]
                    dat_all[n]['DIFF_MAG{}_FINAL'.format(suffix)][:,ind_out][mask] = -2.5 * np.log10(np.abs(dat_all[n]['DIFF_FLUX{}_FINAL'.format(suffix)][:,ind_out][mask])) + 23.9


        #AUTO
        else:
            choice_arr = (dat_all[n]['DM_FINAL_TYPE'][:,i] == 'SFFT')
            
            for dt in ['SFFT', 'Sub']:
                if dt == 'SFFT':
                    mask = choice_arr.copy()
                elif dt == 'Sub':
                    mask = (~choice_arr).copy()
                    
                suffix2 = '_DiffStack{}'.format(dt)
                    
                dat_all[n]['DM_FINAL'][:,i][mask] = dat_all[n]['DM_AUTO{}'.format(suffix2)][mask]
                dat_all[n]['DMERR_FINAL'][:,i][mask] = dat_all[n]['DMERR_AUTO{}'.format(suffix2)][mask]
                dat_all[n]['DIFF_FLUX_FINAL'][:,i][mask] = dat_all[n]['FLUX_AUTO{}'.format(suffix2)][mask]
                dat_all[n]['DIFF_FLUXERR_FINAL'][:,i][mask] = dat_all[n]['FLUXERR_AUTO{}'.format(suffix2)][mask]
                dat_all[n]['DIFF_MAG_FINAL'][:,i][mask] = -2.5 * np.log10(np.abs(dat_all[n]['DIFF_FLUX_FINAL'][:,i][mask])) + 23.9


################################################################################################################
#SAVE CATALOGS  ################################################################################################
################################################################################################################

print('Removing sources right next to bad pixels')
print('\t F200W: {}'.format(np.sum(remove_mask_f200w)))
print('\t F444W: {}'.format(np.sum(remove_mask_f444w)))

dat_all[0] = dat_all[0][~remove_mask_f200w].copy()
dat_all[1] = dat_all[1][~remove_mask_f444w].copy()

for n, b in enumerate(bands):
    dat_all[n].write(maindir + 'nexus_{}_stacked_sources_{}.fits'.format(name_prefix, b), overwrite=True)