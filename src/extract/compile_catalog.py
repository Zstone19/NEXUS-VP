import numpy as np
from tqdm import tqdm
import os
import warnings
import numba as nb

from astropy.table import Table, vstack
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord

#david wuz here haha 

def flux_to_mag(f):
    f = np.abs(f)
    
    return -2.5*np.log10(f) + 8.9

def flux_to_mag_w_err(f, ferr):
    f = np.abs(f)
    ferr = np.abs(ferr)
    
    m = flux_to_mag(f)
    dm = 2.5 * ferr / (f * np.log(10))
    return m, dm

############################################################################################################
############################################################################################################

#@nb.njit(fastmath=True)
def get_match_inds(coord1, coord2, thresh=.15, remove_close=False):
    
    inds_match1 = []
    inds_match2 = []
    bad_inds = []
    for i in range(len(coord1)):       
        # dists = np.sqrt((ra2 - ra1[i])**2 + (dec2 - dec1[i])**2) * 3600 #arcsec
        dists = coord2.separation(coord1[i]).to(u.arcsec).value
        mask_i = dists < thresh
        
        inds2_match_i = np.argwhere(mask_i).flatten()
        
        if mask_i.sum() == 1:
            if (inds2_match_i[0] in bad_inds) and remove_close:
                continue
            
            inds_match1.append(i)
            inds_match2.append(inds2_match_i[0])
            
        elif mask_i.sum() > 1:
            dists_match_i = dists[inds2_match_i]
            min_ind = np.argmin(dists_match_i)
            
            if (min_ind in bad_inds) and remove_close:
                continue
            
            inds_match1.append(i)
            inds_match2.append(inds2_match_i[min_ind])
            
            for j in range(len(inds2_match_i)):
                if j != min_ind:
                    bad_inds.append(inds2_match_i[j])
                    
    return inds_match1, inds_match2, bad_inds


def match_catalogs(cat1, cat2, ra1, ra2, dec1, dec2, method='astropy', thresh=.15, remove_close=False):
    if (len(cat1) == 0 or (len(cat2) == 0)):
        matched_cat1 = Table(names=cat1.colnames, dtype=cat1.dtype)
        matched_cat2 = Table(names=cat2.colnames, dtype=cat2.dtype)
        
        unmatched_cat1 = cat1.copy()
        unmatched_cat2 = cat2.copy()
        
        inds_match1 = np.array([], dtype=int)
        inds_match2 = np.array([], dtype=int)
    
    
    elif method == 'astropy':
        coords1 = SkyCoord(ra1, dec1, unit=(u.deg, u.deg))
        coords2 = SkyCoord(ra2, dec2, unit=(u.deg, u.deg))
        
        
        idx, d2d, _ = coords1.match_to_catalog_sky(coords2)
        mask = d2d.arcsec < thresh
        
        matched_cat1 = cat1[mask]
        matched_cat2 = cat2[idx[mask]]
        
        unmatched_cat1 = cat1[~mask]
        
        nomatch_mask = np.ones(len(cat2), dtype=bool)
        for i in range(len(cat2)):
            if i in idx[mask]:
                nomatch_mask[i] = False
            else:
                nomatch_mask[i] = True
                
        unmatched_cat2 = cat2[nomatch_mask]
        
        inds_match1 = np.argwhere(mask).flatten()
        inds_match2 = idx[mask].copy()


        
    elif method == 'custom':   
        coords1 = SkyCoord(ra1, dec1, unit=(u.deg, u.deg))
        coords2 = SkyCoord(ra2, dec2, unit=(u.deg, u.deg))

        inds_match1, inds_match2, bad_inds = get_match_inds(coords1, coords2, thresh=thresh, remove_close=remove_close)
        inds_match1 = np.array(inds_match1, dtype=int)
        inds_match2 = np.array(inds_match2, dtype=int)
        bad_inds = np.array(bad_inds, dtype=int)
             
        matched_cat1 = cat1[inds_match1].copy()
        matched_cat2 = cat2[inds_match2].copy()
        
        nomatch_mask1 = np.ones(len(cat1), dtype=bool)
        for i in range(len(cat1)):
            if i in inds_match1:
                nomatch_mask1[i] = False
            else:
                nomatch_mask1[i] = True
                
        unmatched_cat1 = cat1[nomatch_mask1].copy()
        
        nomatch_mask2 = np.ones(len(cat2), dtype=bool)
        for i in range(len(cat2)):
            if i in inds_match2:
                nomatch_mask2[i] = False
            else:
                nomatch_mask2[i] = True
                
        unmatched_cat2 = cat2[nomatch_mask2].copy()
        

    return matched_cat1, matched_cat2, unmatched_cat1, unmatched_cat2, inds_match1, inds_match2             
                
                
            
            
        
    
    
    


def combine_2_catalogs(colnames_in, colnames_out, 
                       match1, match2, 
                       unmatch1, unmatch2, 
                       suffix1, suffix2):
    nflux = match1['FLUX_APER'].shape[1]
    
    tab_new1 = Table()  #Both
    tab_new2 = Table()  #Only in catdat1
    tab_new3 = Table()  #Only in catdat2
    
    for c1, c2 in zip(colnames_in, colnames_out):
        for suffix, dat in zip([suffix1, suffix2], [match1, match2]):
            tab_new1[c2 + '_' + suffix] = dat[c1]
        
    for c1, c2 in zip(colnames_in, colnames_out):
        tab_new2[c2 + '_' + suffix1] = unmatch1[c1]
        
        if c2 in ['FLUX_APER', 'FLUXERR_APER']:
            tab_new2[c2 + '_' + suffix2] = np.full((len(unmatch1),nflux), np.nan)
        elif c2 == 'FLAGS':
            tab_new2[c2 + '_' + suffix2] = np.zeros(len(unmatch1), dtype=int)
        else:
            tab_new2[c2 + '_' + suffix2] = np.full(len(unmatch1), np.nan)
            
    for c1, c2 in zip(colnames_in, colnames_out):
        tab_new3[c2 + '_' + suffix2] = unmatch2[c1]
        
        if c2 in ['FLUX_APER', 'FLUXERR_APER']:
            tab_new3[c2 + '_' + suffix1] = np.full((len(unmatch2),nflux), np.nan)
        elif c2 == 'FLAGS':
            tab_new3[c2 + '_' + suffix1] = np.zeros(len(unmatch2), dtype=int)
        else:
            tab_new3[c2 + '_' + suffix1] = np.full(len(unmatch2), np.nan)
            
    tab_new_all = vstack([tab_new1, tab_new2, tab_new3], join_type='exact')
    return tab_new_all


def combine_2_catalogs_pubcat(cols1, cols2, cols_out,
                              match1, match2,
                              unmatch1, unmatch2,
                              suffix1, suffix2):
    
    tab_new1 = Table()  #Both
    tab_new2 = Table()  #Only in catdat1
    tab_new3 = Table()  #Only in catdat2
    
    for c1, c2, c3 in zip(cols1, cols2, cols_out):
        tab_new1[c3 + '_' + suffix1] = match1[c1]
        tab_new1[c3 + '_' + suffix2] = match2[c2]
        
    for c1, c2, c3 in zip(cols1, cols2, cols_out):
        tab_new2[c3 + '_' + suffix1] = unmatch1[c1]
        if ('FLAGS' in c2):
            tab_new2[c3 + '_' + suffix2] = np.zeros(len(unmatch1), dtype=int)
        elif 'ID' in c2:
            tab_new2[c3 + '_' + suffix2] = np.full(len(unmatch1), -1, dtype=int)
        else:
            tab_new2[c3 + '_' + suffix2] = np.full(len(unmatch1), np.nan)

    for c1, c2, c3 in zip(cols1, cols2, cols_out):
        tab_new3[c3 + '_' + suffix2] = unmatch2[c2]
        if ('FLAGS' in c1):
            tab_new3[c3 + '_' + suffix1] = np.zeros(len(unmatch2), dtype=int)
        elif 'ID' in c1:
            tab_new3[c3 + '_' + suffix1] = np.full(len(unmatch2), -1, dtype=int)
        else:
            tab_new3[c3 + '_' + suffix1] = np.full(len(unmatch2), np.nan)

    tab_new_all = vstack([tab_new1, tab_new2, tab_new3], join_type='exact')
    return tab_new_all



def add_additional_catalog(colnames_in, colnames_out, catdat_main, catdat_addtl, 
                           inds_match_main, inds_match_addtl, suffix_addtl):
    
    inds_nomatch_addtl = []
    for i in range(len(catdat_addtl)):
        if i not in inds_match_addtl:
            inds_nomatch_addtl.append(i)
    inds_nomatch_addtl = np.array(inds_nomatch_addtl, dtype=int)
    
    nflux = catdat_main['FLUX_APER_REF_INPUT'].shape[1]

    tab_old = catdat_main.copy()
    old_colnames = tab_old.colnames
    
    #Add columns
    for c1, c2 in zip(colnames_in, colnames_out):
        if c2 in ['FLUX_APER', 'FLUXERR_APER']:
            tab_old[c2 + '_' + suffix_addtl] = np.full((len(catdat_main),nflux), np.nan)
        elif c2 == 'FLAGS':
            tab_old[c2 + '_' + suffix_addtl] = 0
        else:
            tab_old[c2 + '_' + suffix_addtl] = np.nan

    #Fill columns
    for c1, c2 in zip(colnames_in, colnames_out):        
        tab_old[c2 + '_' + suffix_addtl][inds_match_main] = catdat_addtl[c1][inds_match_addtl]


    nnew = len(inds_nomatch_addtl)
    tab_new = Table()
    for col in old_colnames:
        if ('FLUX_APER' in col) or ('FLUXERR_APER' in col):
            tab_new[col] = np.full((nnew,nflux), np.nan)
        elif 'FLAGS' in col:
            tab_new[col] = 0
        else:
            tab_new[col] = np.nan
            

    for c1, c2 in zip(colnames_in, colnames_out):
        tab_new[c2 + '_' + suffix_addtl] = catdat_addtl[c1][inds_nomatch_addtl]

    tab_combined = vstack([tab_old, tab_new], join_type='exact')
    return tab_combined



def add_additional_catalog_pubcat(colnames_main, colnames_addtl, catdat_main, catdat_addtl, 
                                  inds_match_main, inds_match_addtl, suffix_addtl):
    
    inds_nomatch_addtl = []
    for i in range(len(catdat_addtl)):
        if i not in inds_match_addtl:
            inds_nomatch_addtl.append(i)
    inds_nomatch_addtl = np.array(inds_nomatch_addtl, dtype=int)


    tab_old = catdat_main.copy()
    old_colnames = tab_old.colnames
    
    #Add columns
    for c2 in colnames_main:
        if c2 == 'ID':
            continue
        
        if c2 == 'FLAGS':
            tab_old[c2 + '_' + suffix_addtl] = 0
        else:
            tab_old[c2 + '_' + suffix_addtl] = np.nan

    #Fill columns
    for c1, c2 in zip(colnames_main, colnames_addtl):        
        if c1 == 'ID':
            continue

        tab_old[c1 + '_' + suffix_addtl][inds_match_main] = catdat_addtl[c2][inds_match_addtl]
        

    nnew = len(inds_nomatch_addtl)
    tab_new = Table()
    for col in old_colnames:
        if 'FLAGS' in col:
            tab_new[col] = np.zeros(nnew, dtype=int)
        else:
            tab_new[col] = np.full(nnew, np.nan)


    for c1, c2 in zip(colnames_main, colnames_addtl):
        if c1 == 'ID':
            continue        
        tab_new[c1 + '_' + suffix_addtl] = catdat_addtl[c2][inds_nomatch_addtl]

    assert np.all( np.array(tab_old.colnames) == np.array(tab_new.colnames) )

    tab_combined = vstack([tab_old, tab_new], join_type='exact')
    return tab_combined


def get_total_catalog(maindir, thresh=1, remove_flags=False, sig=3, 
                      match_method='astropy', remove_close_matches=False,
                      use_zs=True, fname_jwstcat=None, fname_euccat=None, euc_band='J', jwst_band='F115W'):
    
    suffix_n = '.neg'
    suffix = '' 
        
    if sig == 3:
        diff_suffix = '1'
    if sig == 5:
        diff_suffix = '2'
        
    if use_zs:
        colnames1 = np.array(['ALPHA_J2000', 'DELTA_J2000', 
                            'X_IMAGE', 'Y_IMAGE', 'A_IMAGE', 'B_IMAGE', 'THETA_IMAGE', 'CXX_IMAGE', 'CXY_IMAGE', 'CYY_IMAGE',
                            'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'KRON_RADIUS',
                            'MAG_ISO', 'MAGERR_ISO', 'FLUX_ISO', 'FLUXERR_ISO',
                            'MAG_ISOCOR', 'MAGERR_ISOCOR', 'FLUX_ISOCOR', 'FLUXERR_ISOCOR',
                            'FLUX_APER', 'FLUXERR_APER', 
                            'FLAGS', 'SNR_WIN'], dtype=str)

        colnames2 = colnames1.copy()
        colnames2[0] = 'RA'
        colnames2[1] = 'DEC'
        colnames2[2] = 'X'
        colnames2[3] = 'Y'
        colnames2[4] = 'A'
        colnames2[5] = 'B'
        colnames2[6] = 'THETA'
        colnames2[7] = 'CXX'
        colnames2[8] = 'CXY'
        colnames2[9] = 'CYY'
        colnames2[-1] = 'SNR'
    else:
        #NEXUS
        colnames2 = ['ID', 'RA', 'DEC', 'FLUX_AUTO_{}'.format(jwst_band), 'FLUXERR_AUTO_{}'.format(jwst_band), 'FLAGS_{}'.format(jwst_band)]
        #Euclid
        colnames1 = ['OBJECT_ID', 'RIGHT_ASCENSION', 'DECLINATION', 'FLUX_{}_SERSIC'.format(euc_band), 'FLUXERR_{}_SERSIC'.format(euc_band), 'FLAG_{}'.format(euc_band)]
        #DIFF
        colnames3 = ['ID', 'ALPHA_J2000', 'DELTA_J2000', 'FLUX_AUTO', 'FLUXERR_AUTO', 'FLAGS']
        
        colnames_out = ['ID', 'RA', 'DEC', 'FLUX', 'FLUXERR', 'FLAGS']



    if use_zs:
        catdat_r = Table.read(maindir + 'REF.cat', hdu=2)
        catdat_s = Table.read(maindir + 'SCI.cat', hdu=2)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            catdat_r = Table.read(fname_euccat, hdu=1)
            catdat_s = Table.read(fname_jwstcat, hdu=1)
            
            # catdat_r2 = Table.read(maindir + 'REF.cat', hdu=2)
            # catdat_s2 = Table.read(maindir + 'SCI.cat', hdu=2)
    
    # catdat_sr = Table.read(maindir + 'SCI_r.cat', hdu=2)
    # catdat_rs = Table.read(maindir + 'REF_s.cat', hdu=2)
    # catdat_diff_r = Table.read(maindir + 'DIFF_r{}.cat'.format(suffix), hdu=2)
    # catdat_diff_s = Table.read(maindir + 'DIFF_s{}.cat'.format(suffix), hdu=2)
    catdat_diff = Table.read(maindir + 'DIFF_raw{}{}.cat'.format(diff_suffix, suffix), hdu=2)
    catdat_diff_n = Table.read(maindir + 'DIFF_raw{}{}.cat'.format(diff_suffix, suffix_n), hdu=2)
    
    if use_zs:
        nflux = catdat_r['FLUX_APER'].shape[1]    
    
    ########################################################################################################################
    #Objects in both REF and SCI public catalogs
        
    ra1 = catdat_r['RIGHT_ASCENSION'].data.astype(float)
    ra2 = catdat_s['RA'].data.astype(float)
    dec1 = catdat_r['DECLINATION'].data.astype(float)
    dec2 = catdat_s['DEC'].data.astype(float)

        
    matched_r, matched_s, unmatched_r, unmatched_s, _, _ = match_catalogs(catdat_r, catdat_s, ra1, ra2, dec1, dec2, method=match_method, thresh=thresh, remove_close=remove_close_matches)
    
    if use_zs:
        total_catdat = combine_2_catalogs(colnames1, colnames2, matched_r, matched_s, unmatched_r, unmatched_s, 'REF_INPUT', 'SCI_INPUT')        
    else:
        total_catdat = combine_2_catalogs_pubcat(colnames1, colnames2, colnames_out, 
                                                 matched_r, matched_s, 
                                                 unmatched_r, unmatched_s, 
                                                 'EUCLID', 'NEXUS')
        
    # ###########################################################################################################################
    # #Combine with REF
    
    # ra_tot = np.vstack([total_catdat['RA_EUCLID'], total_catdat['RA_NEXUS']])
    # ra_tot = np.nanmedian(ra_tot, axis=0)
    # dec_tot = np.vstack([total_catdat['DEC_EUCLID'], total_catdat['DEC_NEXUS']])
    # dec_tot = np.nanmedian(dec_tot, axis=0)
    
    # assert np.isnan(ra_tot).sum() == 0
    # assert np.isnan(dec_tot).sum() == 0
    # assert len(ra_tot) == len(dec_tot) == len(total_catdat)
    
    # ra_r = catdat_r2['ALPHA_J2000'].data.astype(np.float64)
    # dec_r = catdat_r2['DELTA_J2000'].data.astype(np.float64)
    
    # matched_r, matched_s, unmatched_r, unmatched_s, inds_match1, inds_match2 = match_catalogs(total_catdat, catdat_r2, ra_tot, ra_r, dec_tot, dec_r, method=match_method, thresh=thresh, remove_close=remove_close_matches)

    # total_catdat = add_additional_catalog_pubcat(colnames_out, colnames3, total_catdat, catdat_r2, 
    #                                              inds_match1, inds_match2, 'REF_INPUT')
    
    # ###########################################################################################################################
    # #Combine with SCI
    
    # ra_tot = np.vstack([total_catdat['RA_EUCLID'], total_catdat['RA_NEXUS'], total_catdat['RA_REF_INPUT']])
    # ra_tot = np.nanmedian(ra_tot, axis=0)
    # dec_tot = np.vstack([total_catdat['DEC_EUCLID'], total_catdat['DEC_NEXUS'], total_catdat['DEC_REF_INPUT']])
    # dec_tot = np.nanmedian(dec_tot, axis=0)
    
    # assert np.isnan(ra_tot).sum() == 0
    # assert np.isnan(dec_tot).sum() == 0
    # assert len(ra_tot) == len(dec_tot) == len(total_catdat)
    
    # ra_s = catdat_s2['ALPHA_J2000'].data.astype(np.float64)
    # dec_s = catdat_s2['DELTA_J2000'].data.astype(np.float64)
    
    # matched_r, matched_s, unmatched_r, unmatched_s, inds_match1, inds_match2 = match_catalogs(total_catdat, catdat_s2, ra_tot, ra_s, dec_tot, dec_s, method=match_method, thresh=thresh, remove_close=remove_close_matches)
  
    # total_catdat = add_additional_catalog_pubcat(colnames_out, colnames3, total_catdat, catdat_s2, 
    #                                              inds_match1, inds_match2, 'SCI_INPUT')
            
    ###########################################################################################################################
    #Combine with REF_s 
    
    # ra_tot = np.vstack([total_catdat['RA_REF_INPUT'], total_catdat['RA_SCI_INPUT']])
    # ra_tot = np.nanmedian(ra_tot, axis=0)
    # dec_tot = np.vstack([total_catdat['DEC_REF_INPUT'], total_catdat['DEC_SCI_INPUT']])
    # dec_tot = np.nanmedian(dec_tot, axis=0)
    
    # assert np.isnan(ra_tot).sum() == 0
    # assert np.isnan(dec_tot).sum() == 0
    # assert len(ra_tot) == len(dec_tot) == len(total_catdat)
    
    # catdat_total = SkyCoord(ra_tot, dec_tot, unit=(u.deg, u.deg))
    # coords_rs = SkyCoord(catdat_rs['ALPHA_J2000'], catdat_rs['DELTA_J2000'], unit=(u.deg, u.deg))
    
    # #Match
    # if (len(catdat_total) > 0) and (len(coords_rs) > 0):
    #     idx, d2d, _ = coords_rs.match_to_catalog_sky(catdat_total)
    #     mask = d2d.arcsec < thresh

    #     matched_total = total_catdat[idx[mask]]
    #     matched_rs = catdat_rs[mask]
    # else:
    #     matched_total = Table(names=total_catdat.colnames, dtype=total_catdat.dtype)
    #     matched_rs = Table(names=catdat_rs.colnames, dtype=catdat_rs.dtype)
    #     mask = np.full(len(total_catdat), False)
    #     idx = np.array([], dtype=int)
        
    # #Unmatched
    # unmatched_rs = catdat_rs[~mask]
    
    # #Combine
    # total_catdat = add_additional_catalog(colnames1, colnames2, 
    #                                       total_catdat, catdat_rs, 
    #                                       idx, mask, 
    #                                       'REF_SCIAPER')

    ###########################################################################################################################
    #Combine with SCI_r
    
    # ra_tot = np.vstack([total_catdat['RA_REF_INPUT'], total_catdat['RA_SCI_INPUT'], total_catdat['RA_REF_SCIAPER']])
    # ra_tot = np.nanmedian(ra_tot, axis=0)
    # dec_tot = np.vstack([total_catdat['DEC_REF_INPUT'], total_catdat['DEC_SCI_INPUT'], total_catdat['DEC_REF_SCIAPER']])
    # dec_tot = np.nanmedian(dec_tot, axis=0)
    
    # assert np.isnan(ra_tot).sum() == 0
    # assert np.isnan(dec_tot).sum() == 0
    # assert len(ra_tot) == len(dec_tot) == len(total_catdat)
    
    # catdat_total = SkyCoord(ra_tot, dec_tot, unit=(u.deg, u.deg))
    # coords_sr = SkyCoord(catdat_sr['ALPHA_J2000'], catdat_sr['DELTA_J2000'], unit=(u.deg, u.deg))
    
    # #Match
    # if (len(catdat_total) > 0) and (len(coords_sr) > 0):
    #     idx, d2d, _ = coords_sr.match_to_catalog_sky(catdat_total)
    #     mask = d2d.arcsec < thresh

    #     matched_total = total_catdat[idx[mask]]
    #     matched_sr = catdat_sr[mask]
    # else:
    #     matched_total = Table(names=total_catdat.colnames, dtype=total_catdat.dtype)
    #     matched_sr = Table(names=catdat_sr.colnames, dtype=catdat_sr.dtype)
    #     mask = np.full(len(total_catdat), False)
    #     idx = np.array([], dtype=int)
        
    # #Unmatched    
    # unmatched_sr = catdat_sr[~mask]
    
    # #Combine
    # total_catdat = add_additional_catalog(colnames1, colnames2, 
    #                                       total_catdat, catdat_sr, 
    #                                       idx, mask, 'SCI_REFAPER')
    
    ###########################################################################################################################
    #Combine with DIFF_r
    
    # ra_tot = np.vstack([total_catdat['RA_REF_INPUT'], total_catdat['RA_SCI_INPUT'], total_catdat['RA_REF_SCIAPER'], total_catdat['RA_SCI_REFAPER']])
    # ra_tot = np.nanmedian(ra_tot, axis=0)
    # dec_tot = np.vstack([total_catdat['DEC_REF_INPUT'], total_catdat['DEC_SCI_INPUT'], total_catdat['DEC_REF_SCIAPER'], total_catdat['DEC_SCI_REFAPER']])
    # dec_tot = np.nanmedian(dec_tot, axis=0)
    
    # assert np.isnan(ra_tot).sum() == 0
    # assert np.isnan(dec_tot).sum() == 0
    # assert len(ra_tot) == len(dec_tot) == len(total_catdat)

    # catdat_total = SkyCoord(ra_tot, dec_tot, unit=(u.deg, u.deg))
    # coords_diff_r = SkyCoord(catdat_diff_r['ALPHA_J2000'], catdat_diff_r['DELTA_J2000'], unit=(u.deg, u.deg))
    
    # #Match
    # if (len(catdat_total) > 0) and (len(coords_diff_r) > 0):
    #     idx, d2d, _ = coords_diff_r.match_to_catalog_sky(catdat_total)
    #     mask = d2d.arcsec < thresh

    #     matched_total = total_catdat[idx[mask]]
    #     matched_diff_r = catdat_diff_r[mask]
    # else:
    #     matched_total = Table(names=total_catdat.colnames, dtype=total_catdat.dtype)
    #     matched_diff_r = Table(names=catdat_diff_r.colnames, dtype=catdat_diff_r.dtype)
    #     mask = np.full(len(total_catdat), False)
    #     idx = np.array([], dtype=int)
        
    # #Unmatched
    # unmatched_diff_r = catdat_diff_r[~mask]

    # #Combine
    # total_catdat = add_additional_catalog(colnames1, colnames2, 
    #                                       total_catdat, catdat_diff_r, 
    #                                       idx, mask, 'REF')

    ###########################################################################################################################
    #Combine with DIFF_s
    
    # ra_tot = np.vstack([total_catdat['RA_REF_INPUT'], total_catdat['RA_SCI_INPUT'], total_catdat['RA_REF_SCIAPER'], total_catdat['RA_SCI_REFAPER'], total_catdat['RA_REF']])
    # ra_tot = np.nanmedian(ra_tot, axis=0)
    # dec_tot = np.vstack([total_catdat['DEC_REF_INPUT'], total_catdat['DEC_SCI_INPUT'], total_catdat['DEC_REF_SCIAPER'], total_catdat['DEC_SCI_REFAPER'], total_catdat['DEC_REF']])
    # dec_tot = np.nanmedian(dec_tot, axis=0)
    
    # assert np.isnan(ra_tot).sum() == 0
    # assert np.isnan(dec_tot).sum() == 0
    # assert len(ra_tot) == len(dec_tot) == len(total_catdat)
    
    # catdat_total = SkyCoord(ra_tot, dec_tot, unit=(u.deg, u.deg))
    # coords_diff_s = SkyCoord(catdat_diff_s['ALPHA_J2000'], catdat_diff_s['DELTA_J2000'], unit=(u.deg, u.deg))
    
    # #Match
    # if (len(catdat_total) > 0) and (len(coords_diff_s) > 0):
    #     idx, d2d, _ = coords_diff_s.match_to_catalog_sky(catdat_total)
    #     mask = d2d.arcsec < thresh

    #     matched_total = total_catdat[idx[mask]]
    #     matched_diff_s = catdat_diff_s[mask]
    # else:
    #     matched_total = Table(names=total_catdat.colnames, dtype=total_catdat.dtype)
    #     matched_diff_s = Table(names=catdat_diff_s.colnames, dtype=catdat_diff_s.dtype)
    #     mask = np.full(len(total_catdat), False)
    #     idx = np.array([], dtype=int)
        
    # #Unmatched
    # unmatched_diff_s = catdat_diff_s[~mask]
    
    # #Combine
    # total_catdat = add_additional_catalog(colnames1, colnames2, 
    #                                       total_catdat, catdat_diff_s, 
    #                                       idx, mask, 'SCI')        
    
    ###########################################################################################################################
    #Combine DIFF and DIFF_n
    
    ra1 = catdat_diff['ALPHA_J2000'].data.astype(np.float64)
    ra2 = catdat_diff_n['ALPHA_J2000'].data.astype(np.float64)
    dec1 = catdat_diff['DELTA_J2000'].data.astype(np.float64)
    dec2 = catdat_diff_n['DELTA_J2000'].data.astype(np.float64)
    matched_r, matched_s, unmatched_r, unmatched_s, inds_match1, inds_match2 = match_catalogs(catdat_diff, catdat_diff_n, ra1, ra2, dec1, dec2, method=match_method, thresh=thresh, remove_close=remove_close_matches)
    
    cols_in = ['ALPHA_J2000', 'DELTA_J2000', 'FLUX_AUTO', 'FLUXERR_AUTO', 'FLAGS']
    cols_out = ['RA', 'DEC', 'FLUX', 'FLUXERR', 'FLAGS']
    difftot = combine_2_catalogs(cols_in, cols_out, matched_r, matched_s, unmatched_r, unmatched_s, 'DIFF', 'DIFF_n')
    
    ###########################################################################################################################
    
    if remove_flags:
        if use_zs:
            m = (total_catdat['FLAGS_REF_INPUT'] == 0) & (total_catdat['FLAGS_SCI_INPUT'] == 0)
        else:
            m = (total_catdat['FLAGS_EUCLID'] == 0) & (total_catdat['FLAGS_NEXUS'] == 0)

        total_catdat = total_catdat[m].copy()
        
        if use_zs:
            m = (difftot['FLAGS'] == 0) & (difftot['FLAGS_n'] == 0)
        else:
            m = (difftot['FLAGS_DIFF'] == 0) & (difftot['FLAGS_DIFF_n'] == 0)
            
        difftot = difftot[m].copy()

    return total_catdat, difftot

    
############################################################################################################
############################################################################################################
# Combine all catalogs

def make_full_catalog(maindir, thresh=1, sig=5, match_method='astropy', remove_close_matches=False):
    colnames1 = np.array(['ALPHA_J2000', 'DELTA_J2000', 
                          'X_IMAGE', 'Y_IMAGE', 'A_IMAGE', 'B_IMAGE', 'THETA_IMAGE', 'CXX_IMAGE', 'CXY_IMAGE', 'CYY_IMAGE',
                          'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'KRON_RADIUS',
                          'MAG_ISO', 'MAGERR_ISO', 'FLUX_ISO', 'FLUXERR_ISO',
                          'MAG_ISOCOR', 'MAGERR_ISOCOR', 'FLUX_ISOCOR', 'FLUXERR_ISOCOR',
                          'FLUX_APER', 'FLUXERR_APER', 
                          'FLAGS', 'SNR_WIN'], dtype=str)

    colnames2 = colnames1.copy()
    colnames2[0] = 'RA'
    colnames2[1] = 'DEC'
    colnames2[2] = 'X'
    colnames2[3] = 'Y'
    colnames2[4] = 'A'
    colnames2[5] = 'B'
    colnames2[6] = 'THETA'
    colnames2[7] = 'CXX'
    colnames2[8] = 'CXY'
    colnames2[9] = 'CYY'
    colnames2[-1] = 'SNR'
    
    
    
    #Get individual catalogs
    alldat = get_total_catalog(maindir, thresh=thresh, sig=sig,
                                match_method=match_method, remove_close_matches=remove_close_matches)
    alldat_n = get_total_catalog(maindir, thresh=thresh, neg=True, sig=sig,
                                match_method=match_method, remove_close_matches=remove_close_matches)        
    
    nflux = alldat['FLUX_APER_REF_INPUT'].shape[1]
    

    catnames1 = ['REF', 'SCI', 'REF_INPUT', 'SCI_INPUT', 'REF_SCIAPER', 'SCI_REFAPER', 'DIFF']    
    names2 = ['DIFF']
    names3 = ['REF', 'SCI', 'REF_INPUT', 'SCI_INPUT', 'DIFF']
    names4 = ['DIFF']
        
    names1 = ['REF_SCIAPER', 'SCI_REFAPER']
    
    
    colnames_out = []
    for c2 in colnames2:
        for name in catnames1:

            if (c2 in ['RA', 'DEC', 'X', 'Y', 'A', 'B', 'THETA', 'CXX', 'CXY', 'CYY']) and (name in names1):
                continue

            colnames_out.append(c2 + '_' + name)
        
            if (c2 in ['RA', 'DEC', 'X', 'Y', 'A', 'B', 'THETA', 'CXX', 'CXY', 'CYY']) and (name in names2):
                colnames_out.append(c2 + '_' + name + '_n')

            if (c2 in ['MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'KRON_RADIUS', 'MAG_ISO', 'MAGERR_ISO', 'FLUX_ISO', 'FLUXERR_ISO', 'MAG_ISOCOR', 'MAGERR_ISOCOR', 'FLUX_ISOCOR', 'FLUXERR_ISOCOR', 'FLUX_APER', 'FLUXERR_APER']) and (name in names3):
                colnames_out.append(c2 + '_' + name + '_n')
                
            if (c2 in ['FLAGS', 'SNR']) and (name in names4):
                colnames_out.append(c2 + '_' + name + '_n')

    colnames_out.append('DIFF_DETECTED')
    colnames_out.append('DIFF_DETECTED_n')
        

    ra_tot = np.vstack([alldat['RA_REF_INPUT'], alldat['RA_SCI_INPUT'], alldat['RA_REF_SCIAPER'], alldat['RA_SCI_REFAPER'], alldat['RA_REF'], alldat['RA_SCI'], alldat['RA_DIFF']])
    ra_tot = np.nanmedian(ra_tot, axis=0)
    dec_tot = np.vstack([alldat['DEC_REF_INPUT'], alldat['DEC_SCI_INPUT'], alldat['DEC_REF_SCIAPER'], alldat['DEC_SCI_REFAPER'], alldat['DEC_REF'], alldat['DEC_SCI'], alldat['DEC_DIFF']])
    dec_tot = np.nanmedian(dec_tot, axis=0)
    
    assert np.isnan(ra_tot).sum() == 0
    assert np.isnan(dec_tot).sum() == 0
    assert len(ra_tot) == len(dec_tot) == len(alldat)
    
    ra_tot_n = np.vstack([alldat_n['RA_REF_INPUT'], alldat_n['RA_SCI_INPUT'], alldat_n['RA_REF_SCIAPER'], alldat_n['RA_SCI_REFAPER'], alldat_n['RA_REF'], alldat_n['RA_SCI'], alldat_n['RA_DIFF']])
    ra_tot_n = np.nanmedian(ra_tot_n, axis=0)
    dec_tot_n = np.vstack([alldat_n['DEC_REF_INPUT'], alldat_n['DEC_SCI_INPUT'], alldat_n['DEC_REF_SCIAPER'], alldat_n['DEC_SCI_REFAPER'], alldat_n['DEC_REF'], alldat_n['DEC_SCI'], alldat_n['DEC_DIFF']])
    dec_tot_n = np.nanmedian(dec_tot_n, axis=0)
    
    assert np.isnan(ra_tot_n).sum() == 0
    assert np.isnan(dec_tot_n).sum() == 0
    assert len(ra_tot_n) == len(dec_tot_n) == len(alldat_n)

    
    #Match
    matched_pos, matched_neg, unmatched_pos, unmatched_neg, inds_match1, inds_match2 = match_catalogs(alldat, alldat_n, ra_tot, ra_tot_n, dec_tot, dec_tot_n, 
                                                                                                      method=match_method, thresh=thresh, remove_close=remove_close_matches)

    if (len(coords_pos) > 0) and (len(coords_neg) > 0):        
        # if np.abs(mask.sum() - len(alldat)) > 0:
        #     print('WARNING: Some objects removed from DIFF catalog!')
        # if np.abs(len(reftot_n) - len(idx[mask])) > 0:
        #     print('WARNING: Some objects removed from -DIFF catalog!')

        #Combine matched
        alldat_out1 = Table()
        for c1, c2 in zip(colnames1, colnames2):
            for name in catnames1:
                
                if (c2 in ['RA', 'DEC', 'X', 'Y', 'A', 'B', 'THETA', 'CXX', 'CXY', 'CYY']) and (name in names1):
                    continue
                
                alldat_out1[c2 + '_' + name] = alldat[c2 + '_' + name][inds_match1]
                
                if (c2 in ['RA', 'DEC', 'X', 'Y', 'A', 'B', 'THETA', 'CXX', 'CXY', 'CYY']) and (name in names2):
                    alldat_out1[c2 + '_' + name + '_n'] = alldat_n[c2 + '_' + name][inds_match2]
                    
                if (c2 in ['MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'KRON_RADIUS', 'MAG_ISO', 'MAGERR_ISO', 'FLUX_ISO', 'FLUXERR_ISO', 'MAG_ISOCOR', 'MAGERR_ISOCOR', 'FLUX_ISOCOR', 'FLUXERR_ISOCOR', 'FLUX_APER', 'FLUXERR_APER']) and (name in names3):
                    alldat_out1[c2 + '_' + name + '_n'] = alldat_n[c2 + '_' + name][inds_match2]
                    
                if (c2 in ['FLAGS', 'SNR']) and (name in names4):
                    alldat_out1[c2 + '_' + name + '_n'] = alldat_n[c2 + '_' + name][inds_match2]    
    


        #In pos, but not neg  
        alldat_out2 = Table()    
        for c1, c2 in zip(colnames1, colnames2):
            for name in catnames1:
                
                if (c2 in ['RA', 'DEC', 'X', 'Y', 'A', 'B', 'THETA', 'CXX', 'CXY', 'CYY']) and (name in names1):
                    continue
                
                alldat_out2[c2 + '_' + name] = unmatched_pos[c2 + '_' + name]
                
                if (c2 in ['RA', 'DEC', 'X', 'Y', 'A', 'B', 'THETA', 'CXX', 'CXY', 'CYY']) and (name in names2):
                    alldat_out2[c2 + '_' + name + '_n'] = np.full(len(unmatched_pos), np.nan) 

                    
                if (c2 in ['MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'KRON_RADIUS', 'MAG_ISO', 'MAGERR_ISO', 'FLUX_ISO', 'FLUXERR_ISO', 'MAG_ISOCOR', 'MAGERR_ISOCOR', 'FLUX_ISOCOR', 'FLUXERR_ISOCOR', 'FLUX_APER', 'FLUXERR_APER']) and (name in names3):
                    if c2 in ['FLUX_APER', 'FLUXERR_APER']:
                        alldat_out2[c2 + '_' + name + '_n'] = np.full((len(unmatched_pos), nflux), np.nan)
                    else:
                        alldat_out2[c2 + '_' + name + '_n'] = np.full(len(unmatched_pos), np.nan)
                    
                if (c2 in ['FLAGS', 'SNR']) and (name in names4):
                    if c2 == 'FLAGS':
                        alldat_out2[c2 + '_' + name + '_n'] = np.zeros(len(unmatched_pos), dtype=int)
                    else:
                        alldat_out2[c2 + '_' + name + '_n'] = np.full(len(unmatched_pos), np.nan)
                        

        # In neg, but not pos
        alldat_out3 = Table()      
        for c1, c2 in zip(colnames1, colnames2):
            for name in catnames1:
                
                if (c2 in ['RA', 'DEC', 'X', 'Y', 'A', 'B', 'THETA', 'CXX', 'CXY', 'CYY']) and (name in names1):
                    continue
                
                if c2 in ['FLUX_APER', 'FLUXERR_APER']:
                    alldat_out3[c2 + '_' + name] = np.full((len(unmatched_neg), nflux), np.nan)
                elif c2 == 'FLAGS':
                    alldat_out3[c2 + '_' + name] = np.zeros(len(unmatched_neg), dtype=int)
                else:
                    alldat_out3[c2 + '_' + name] = np.full(len(unmatched_neg), np.nan)

                
                if (c2 in ['RA', 'DEC', 'X', 'Y', 'A', 'B', 'THETA', 'CXX', 'CXY', 'CYY']) and (name in names2):
                    alldat_out3[c2 + '_' + name + '_n'] = unmatched_neg[c2 + '_' + name]
                    
                if (c2 in ['MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'KRON_RADIUS', 'MAG_ISO', 'MAGERR_ISO', 'FLUX_ISO', 'FLUXERR_ISO', 'MAG_ISOCOR', 'MAGERR_ISOCOR', 'FLUX_ISOCOR', 'FLUXERR_ISOCOR', 'FLUX_APER', 'FLUXERR_APER']) and (name in names3):
                    alldat_out3[c2 + '_' + name + '_n'] = unmatched_neg[c2 + '_' + name]
                    
                if (c2 in ['FLAGS', 'SNR']) and (name in names4):
                    alldat_out3[c2 + '_' + name + '_n'] = unmatched_neg[c2 + '_' + name]  


        alldat_out = vstack([alldat_out1, alldat_out2, alldat_out3], join_type='exact')


        #DIFF Detected
        mask_diff_detect = ( ~np.isnan(alldat_out['FLUX_AUTO_DIFF']) ) & (alldat_out['FLUXERR_AUTO_DIFF'] > 0)
        alldat_out['DIFF_DETECTED'] = mask_diff_detect
        mask_diff_detect_n = ( ~np.isnan(alldat_out['FLUX_AUTO_DIFF_n']) ) & (alldat_out['FLUXERR_AUTO_DIFF_n'] > 0)
        alldat_out['DIFF_DETECTED_n'] = mask_diff_detect_n

        
        outdat = alldat_out.copy()        

        
    else:
        outdat = Table(names=colnames_out)
    
    return outdat



def make_full_catalog_pubcat(maindir, fname_jwstcat, fname_euccat, thresh=1, sig=5, match_method='astropy', remove_close_matches=False):
    colnames2 = ['ID', 'RA', 'DEC', 'FLUX', 'FLUXERR', 'FLAGS']
    
    #Get individual catalogs
    alldat, diffdat = get_total_catalog(maindir, thresh=thresh, sig=sig, 
                                        match_method=match_method, remove_close_matches=False,
                                        use_zs=False, fname_jwstcat=fname_jwstcat, fname_euccat=fname_euccat)       

    catnames1 = ['EUCLID', 'NEXUS', 'DIFF']# 'REF_INPUT', 'SCI_INPUT']    
    names3 = ['EUCLID', 'NEXUS', 'DIFF']# 'REF_INPUT', 'SCI_INPUT', 'DIFF']
    names4 = ['DIFF']#, 'REF_INPUT', 'SCI_INPUT']
    
    
    colnames_out = []
    for c2 in colnames2:
        for name in catnames1:
            
            if (c2 == 'ID') and (name in names4):
                continue

            colnames_out.append(c2 + '_' + name)
        
            if (c2 in ['RA', 'DEC']) and (name in ['DIFF']):
                colnames_out.append(c2 + '_' + name + '_n')

            if (c2 in ['FLUX', 'FLUXERR']) and (name in names4):
                colnames_out.append(c2 + '_' + name + '_n')
                
            if (c2 in ['FLAGS']) and (name in ['DIFF']):
                colnames_out.append(c2 + '_' + name + '_n')

    colnames_out.append('DIFF_DETECTED')
    colnames_out.append('DIFF_DETECTED_n')

    ra_tot = np.vstack([alldat['RA_EUCLID'], alldat['RA_NEXUS']])#, alldat['RA_REF_INPUT'], alldat['RA_SCI_INPUT']])
    ra_tot = np.nanmedian(ra_tot, axis=0)
    dec_tot = np.vstack([alldat['DEC_EUCLID'], alldat['DEC_NEXUS']])#, alldat['DEC_REF_INPUT'], alldat['DEC_SCI_INPUT']])
    dec_tot = np.nanmedian(dec_tot, axis=0)
    
    ra_tot_d = np.vstack([diffdat['RA_DIFF'], diffdat['RA_DIFF_n']])
    ra_tot_d = np.nanmedian(ra_tot_d, axis=0)
    dec_tot_d = np.vstack([diffdat['DEC_DIFF'], diffdat['DEC_DIFF_n']])
    dec_tot_d = np.nanmedian(dec_tot_d, axis=0)

    
    #Match
    coords_cat = SkyCoord(ra_tot, dec_tot, unit=(u.deg, u.deg))
    coords_diff = SkyCoord(ra_tot_d, dec_tot_d, unit=(u.deg, u.deg))

    if (len(coords_cat) > 0) and (len(coords_diff) > 0):
        # idx, d2d, _ = coords_cat.match_to_catalog_sky(coords_diff)
        # mask = d2d.arcsec < thresh 
        
        # alldat_matched = alldat[mask]
        # diffdat_matched = diffdat[idx[mask]]
        
        # alldat_unmatched = alldat[~mask]
        # nomatch_mask = np.ones(len(diffdat), dtype=bool)
        # for i in idx[mask]:
        #     nomatch_mask[i] = False
        # diffdat_unmatched = diffdat[nomatch_mask]
        
        alldat_matched, diffdat_matched, alldat_unmatched, diffdat_unmatched, inds_match_a, inds_match_d = match_catalogs(alldat, diffdat, ra_tot, ra_tot_d, dec_tot, dec_tot_d, method=match_method, thresh=thresh, remove_close=remove_close_matches)

        
        # if np.abs(mask.sum() - len(alldat)) > 0:
        #     print('WARNING: Some objects removed from DIFF catalog!')
        # if np.abs(len(reftot_n) - len(idx[mask])) > 0:
        #     print('WARNING: Some objects removed from -DIFF catalog!')

        #Combine matched
        alldat_out1 = Table()
        for c2 in colnames2:
            for name in catnames1:
                
                if name in ['EUCLID', 'NEXUS']:
                    alldat_out1[c2 + '_' + name] = alldat_matched[c2 + '_' + name]
                else:
                    if c2 == 'ID':
                        pass
                    else:
                        alldat_out1[c2 + '_' + name] = diffdat_matched[c2 + '_' + name]
                        
                        if name == 'DIFF':
                            alldat_out1[c2 + '_' + name + '_n'] = diffdat_matched[c2 + '_' + name + '_n']    


        #In cat, but not diff  
        alldat_out2 = Table()    
        for c2 in colnames2:
            for name in catnames1:
                
                if name in ['EUCLID', 'NEXUS', 'REF_INPUT', 'SCI_INPUT']:
                    if (c2 == 'ID') and (name in ['REF_INPUT', 'SCI_INPUT']):
                        pass
                    else:
                        alldat_out2[c2 + '_' + name] = alldat_unmatched[c2 + '_' + name]
                else:
                    if c2 == 'ID':
                        pass
                    elif c2 == 'FLAGS':
                        alldat_out2[c2 + '_' + name] = np.zeros(len(alldat_unmatched), dtype=int)
                        alldat_out2[c2 + '_' + name + '_n'] = np.zeros(len(alldat_unmatched), dtype=int)
                    else:
                        alldat_out2[c2 + '_' + name] = np.full( len(alldat_unmatched), np.nan)
                        alldat_out2[c2 + '_' + name + '_n'] = np.full( len(alldat_unmatched), np.nan)

        # In diff, but not cat
        alldat_out3 = Table()      
        for c2 in colnames2:
            for name in catnames1:
                
                if name in ['EUCLID', 'NEXUS', 'REF_INPUT', 'SCI_INPUT']:
                    if c2 == 'ID':
                        if name in ['REF_INPUT', 'SCI_INPUT']:
                            pass
                        
                        alldat_out3[c2 + '_' + name] = np.full( len(diffdat_unmatched), -1, dtype=int)
                    elif c2 == 'FLAGS':
                        alldat_out3[c2 + '_' + name] = np.zeros(len(diffdat_unmatched), dtype=int)
                    else:
                        alldat_out3[c2 + '_' + name] = np.full( len(diffdat_unmatched), np.nan)

                else:
                    if c2 == 'ID':
                        pass
                    elif c2 == 'FLAGS':
                        alldat_out3[c2 + '_' + name] = np.zeros(len(diffdat_unmatched), dtype=int)
                        alldat_out3[c2 + '_' + name + '_n'] = np.zeros(len(diffdat_unmatched), dtype=int)
                    else:
                        alldat_out3[c2 + '_' + name] = diffdat_unmatched[c2 + '_' + name]
                        alldat_out3[c2 + '_' + name + '_n'] = diffdat_unmatched[c2 + '_' + name + '_n']


        alldat_out = vstack([alldat_out1, alldat_out2, alldat_out3], join_type='exact')

        #DIFF Detected
        mask_diff_detect = ( ~np.isnan(alldat_out['FLUX_DIFF']) ) & (alldat_out['FLUXERR_DIFF'] > 0)
        alldat_out['DIFF_DETECTED'] = mask_diff_detect
        mask_diff_detect_n = ( ~np.isnan(alldat_out['FLUX_DIFF_n']) ) & (alldat_out['FLUXERR_DIFF_n'] > 0)
        alldat_out['DIFF_DETECTED_n'] = mask_diff_detect_n

        
        outdat = alldat_out.copy()        

        
    else:
        outdat = Table(names=colnames_out)
    
    return outdat


############################################################################################################
############################################################################################################
# Combine all catalogs from all cutouts

def get_catalog_allcutout(maindir_source, thresh=1, magtype='auto', neg=False, ncutout=63):
    
    assert magtype in ['auto', 'aper']    
    
    mvals_all_r = []
    dmvals_all_r = []
    merr_vals_all_r = []
    dmerr_vals_all_r = []

    mvals_all_r_nf = []
    dmvals_all_r_nf = []
    merr_vals_all_r_nf = []
    dmerr_vals_all_r_nf = []

    mvals_all_s = []
    dmvals_all_s = []
    merr_vals_all_s = []
    dmerr_vals_all_s = []

    mvals_all_s_nf = []
    dmvals_all_s_nf = []
    merr_vals_all_s_nf = []
    dmerr_vals_all_s_nf = []
    
    dmvals_all_d = []
    dmerr_vals_all_d = []
    dmvals_all_d_nf = []
    dmerr_vals_all_d_nf = []
    
    for i in tqdm(  range(ncutout)  ):
        maindir_diff = maindir_source + 'sources_{}/'.format(i)
        
        if not os.path.exists(maindir_diff):
            continue

        alldat = make_full_catalog(maindir_diff, thresh=thresh)

        if (len(alldat) == 0):
            continue
        

        if magtype == 'auto':
            colref = 'MAG_AUTO_REF'
            colsci = 'MAG_AUTO_SCI'
            coldiff = 'MAG_AUTO_DIFF'
            colref_in = 'MAG_AUTO_REF_INPUT'
            colsci_in = 'MAG_AUTO_SCI_INPUT'
            
            colref_err = 'MAGERR_AUTO_REF'
            colsci_err = 'MAGERR_AUTO_SCI'
            coldiff_err = 'MAGERR_AUTO_DIFF'
            colref_in_err = 'MAGERR_AUTO_REF_INPUT'
            colsci_in_err = 'MAGERR_AUTO_SCI_INPUT'

            
        if magtype == 'aper':
            colref = 'FLUX_APER_REF'
            colsci = 'FLUX_APER_SCI'
            coldiff = 'FLUX_APER_DIFF'
            colref_in = 'FLUX_APER_REF_INPUT'
            colsci_in = 'FLUX_APER_SCI_INPUT'
            
            colref_err = 'FLUXERR_APER_REF'
            colsci_err = 'FLUXERR_APER_SCI'
            coldiff_err = 'FLUXERR_APER_DIFF'
            colref_in_err = 'FLUXERR_APER_REF_INPUT'
            colsci_in_err = 'FLUXERR_APER_SCI_INPUT'
            
            
        colref_flag = 'FLAGS_REF'
        colsci_flag = 'FLAGS_SCI'
        coldiff_flag = 'FLAGS_DIFF'
        colref_in_flag = 'FLAGS_REF_INPUT'
        colsci_in_flag = 'FLAGS_SCI_INPUT'
        
            
        if neg:
            colref += '_n'
            colsci += '_n'
            coldiff += '_n'
            # colref_in += '_n'
            # colsci_in += '_n'
            colref_err += '_n'
            colsci_err += '_n'
            coldiff_err += '_n'
            # colref_in_err += '_n'
            # colsci_in_err += '_n'
            # colref_flag += '_n'
            # colsci_flag += '_n'
            # coldiff_flag += '_n'
            # colref_in_flag += '_n'
            # colsci_in_flag += '_n'

        
        #############################
        #DIFF

        if magtype == 'auto':
            mask_diff = (alldat[coldiff] < 99) & ~np.isnan(alldat[coldiff]) & (alldat[coldiff_err] < 99)
            mask_diff_nf = mask_diff & (alldat[coldiff_flag] == 0)
        if magtype == 'aper':
            mask_diff = ~np.isnan(alldat[coldiff][:,0])
            mask_diff_nf = mask_diff & (alldat[coldiff_flag] == 0)

        
        alldat_diff = alldat[mask_diff].copy()
        alldat_diff_nf = alldat[mask_diff_nf].copy()

        if magtype == 'auto':
            dm = alldat_diff[coldiff].data
            dmerr = alldat_diff[coldiff_err].data
            dm_nf = alldat_diff_nf[coldiff].data
            dmerr_nf = alldat_diff_nf[coldiff_err].data
            
        if magtype == 'aper':
            f = alldat_diff[coldiff].data
            ferr = alldat_diff[coldiff_err].data
            dm, dmerr = flux_to_mag_w_err(f, ferr)
            
            f_nf = alldat_diff_nf[coldiff].data
            ferr_nf = alldat_diff_nf[coldiff_err].data
            dm_nf, dmerr_nf = flux_to_mag_w_err(f_nf, ferr_nf)
            
        dmvals_all_d.append(dm)
        dmerr_vals_all_d.append(dmerr)
        dmvals_all_d_nf.append(dm_nf)
        dmerr_vals_all_d_nf.append(dmerr_nf)
        
        
        
        #############################
        #REF

        if magtype == 'auto':
            mask_ref = (alldat[colref] < 99) & (alldat[colref_in] < 99) & (alldat[colref_in_err] < 99) & ~np.isnan(alldat[colref]) & ~np.isnan(alldat[colref_in])
            mask_ref_nf = mask_diff & (alldat[colref_in_flag] == 0)
        if magtype == 'aper':
            mask_ref = ~np.isnan(alldat[colref][:,0]) & ~np.isnan(alldat[colref_in][:,0])
            mask_ref_nf = mask_diff & (alldat[colref_in_flag] == 0)

        
        alldat_ref = alldat[mask_ref].copy()
        alldat_ref_nf = alldat[mask_ref_nf].copy()

        if magtype == 'auto':
            m = alldat_ref[colref_in].data
            merr = alldat_ref[colref_in_err].data
            m_nf = alldat_ref_nf[colref_in].data
            merr_nf = alldat_ref_nf[colref_in_err].data
            
            dm = alldat_ref[colref].data
            dmerr = alldat_ref[colref_err].data
            dm_nf = alldat_ref_nf[colref].data
            dmerr_nf = alldat_ref_nf[colref_err].data
            

        if magtype == 'aper':
            f = alldat_ref[colref].data
            ferr = alldat_ref[colref_err].data
            dm, dmerr = flux_to_mag_w_err(f, ferr)
            
            f_nf = alldat_ref_nf[colref].data
            ferr_nf = alldat_ref_nf[colref_err].data
            dm_nf, dmerr_nf = flux_to_mag_w_err(f_nf, ferr_nf)
            
            f = alldat_ref[colref_in].data
            ferr = alldat_ref[colref_in_err].data
            m, merr = flux_to_mag_w_err(f, ferr)
            
            f_nf = alldat_ref_nf[colref_in].data
            ferr_nf = alldat_ref_nf[colref_in_err].data
            m_nf, merr_nf = flux_to_mag_w_err(f_nf, ferr_nf)
            
        assert len(m) == len(dm) == len(merr) == len(dmerr)
        assert len(m_nf) == len(dm_nf) == len(merr_nf) == len(dmerr_nf)
            
        mvals_all_r.append(m)
        merr_vals_all_r.append(merr)
        mvals_all_r_nf.append(m_nf)
        merr_vals_all_r_nf.append(merr_nf)
        
        dmvals_all_r.append(dm)
        dmerr_vals_all_r.append(dmerr)
        dmvals_all_r_nf.append(dm_nf)
        dmerr_vals_all_r_nf.append(dmerr_nf)

        #############################
        #SCI
        
        if magtype == 'auto':
            mask_sci = (alldat[colsci] < 99) & (alldat[colsci_in] < 99) & (alldat[colsci_in_err] < 99) & ~np.isnan(alldat[colsci]) & ~np.isnan(alldat[colsci_in])
            mask_sci_nf = mask_diff & (alldat[colsci_in_flag] == 0)
        if magtype == 'aper':
            mask_sci = ~np.isnan(alldat[colsci][:,0]) & ~np.isnan(alldat[colsci_in][:,0])
            mask_sci_nf = mask_diff & (alldat[colsci_in_flag] == 0)

        
        alldat_sci = alldat[mask_sci].copy()
        alldat_sci_nf = alldat[mask_sci_nf].copy()

        if magtype == 'auto':
            m = alldat_sci[colsci_in].data
            merr = alldat_sci[colsci_in_err].data
            m_nf = alldat_sci_nf[colsci_in].data
            merr_nf = alldat_sci_nf[colsci_in_err].data
            
            dm = alldat_sci[colsci].data
            dmerr = alldat_sci[colsci_err].data
            dm_nf = alldat_sci_nf[colsci].data
            dmerr_nf = alldat_sci_nf[colsci_err].data
            

        if magtype == 'aper':
            f = alldat_sci[colsci].data
            ferr = alldat_sci[colsci_err].data
            dm, dmerr = flux_to_mag_w_err(f, ferr)
            
            f_nf = alldat_sci_nf[colsci].data
            ferr_nf = alldat_sci_nf[colsci_err].data
            dm_nf, dmerr_nf = flux_to_mag_w_err(f_nf, ferr_nf)
            
            f = alldat_sci[colsci_in].data
            ferr = alldat_sci[colsci_in_err].data
            m, merr = flux_to_mag_w_err(f, ferr)
            
            f_nf = alldat_sci_nf[colsci_in].data
            ferr_nf = alldat_sci_nf[colsci_in_err].data
            m_nf, merr_nf = flux_to_mag_w_err(f_nf, ferr_nf)
            
        assert len(m) == len(dm) == len(merr) == len(dmerr)
        assert len(m_nf) == len(dm_nf) == len(merr_nf) == len(dmerr_nf)
            
        mvals_all_s.append(m)
        merr_vals_all_s.append(merr)
        mvals_all_s_nf.append(m_nf)
        merr_vals_all_s_nf.append(merr_nf)
        
        dmvals_all_s.append(dm)
        dmerr_vals_all_s.append(dmerr)
        dmvals_all_s_nf.append(dm_nf)
        dmerr_vals_all_s_nf.append(dmerr_nf)
        
    
    ##############################
    #Combine
        
    mvals_all_r = np.concatenate(mvals_all_r)
    dmvals_all_r = np.concatenate(dmvals_all_r)
    merr_vals_all_r = np.concatenate(merr_vals_all_r)
    dmerr_vals_all_r = np.concatenate(dmerr_vals_all_r)
    assert len(mvals_all_r) == len(dmvals_all_r) == len(merr_vals_all_r) == len(dmerr_vals_all_r)
    
    mvals_all_r_nf = np.concatenate(mvals_all_r_nf)
    dmvals_all_r_nf = np.concatenate(dmvals_all_r_nf)
    merr_vals_all_r_nf = np.concatenate(merr_vals_all_r_nf)
    dmerr_vals_all_r_nf = np.concatenate(dmerr_vals_all_r_nf)
    assert len(mvals_all_r_nf) == len(dmvals_all_r_nf) == len(merr_vals_all_r_nf) == len(dmerr_vals_all_r_nf)
    
    mvals_all_s = np.concatenate(mvals_all_s)
    dmvals_all_s = np.concatenate(dmvals_all_s)
    merr_vals_all_s = np.concatenate(merr_vals_all_s)
    dmerr_vals_all_s = np.concatenate(dmerr_vals_all_s)
    assert len(mvals_all_s) == len(dmvals_all_s) == len(merr_vals_all_s) == len(dmerr_vals_all_s)
    
    mvals_all_s_nf = np.concatenate(mvals_all_s_nf)
    dmvals_all_s_nf = np.concatenate(dmvals_all_s_nf)
    merr_vals_all_s_nf = np.concatenate(merr_vals_all_s_nf)
    dmerr_vals_all_s_nf = np.concatenate(dmerr_vals_all_s_nf)
    assert len(mvals_all_s_nf) == len(dmvals_all_s_nf) == len(merr_vals_all_s_nf) == len(dmerr_vals_all_s_nf)
    
    dmvals_all_d = np.concatenate(dmvals_all_d)
    dmerr_vals_all_d = np.concatenate(dmerr_vals_all_d)
    dmvals_all_d_nf = np.concatenate(dmvals_all_d_nf)
    dmerr_vals_all_d_nf = np.concatenate(dmerr_vals_all_d_nf)
    assert len(dmvals_all_d) == len(dmerr_vals_all_d)
    
    return mvals_all_r, dmvals_all_r, merr_vals_all_r, dmerr_vals_all_r, \
        mvals_all_r_nf, dmvals_all_r_nf, merr_vals_all_r_nf, dmerr_vals_all_r_nf, \
        mvals_all_s, dmvals_all_s, merr_vals_all_s, dmerr_vals_all_s, \
        mvals_all_s_nf, dmvals_all_s_nf, merr_vals_all_s_nf, dmerr_vals_all_s_nf, \
        dmvals_all_d, dmerr_vals_all_d, \
        dmvals_all_d_nf, dmerr_vals_all_d_nf
        