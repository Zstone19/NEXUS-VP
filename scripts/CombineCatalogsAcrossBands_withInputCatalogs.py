import os
import numpy as np
from astropy.table import Table, vstack

from astropy.coordinates import SkyCoord
import astropy.units as u


def match_tables(tab_diff, tab_total, matching_radius=0.1):
    coords_diff = SkyCoord(tab_diff['RA_DiffStackSub'].data, tab_diff['DEC_DiffStackSub'].data, unit=(u.deg, u.deg))
    coords_nex = SkyCoord(tab_total['ra'].data, tab_total['dec'].data, unit=(u.deg, u.deg))
    
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
    return matched_ind1, matched_ind2, matched_d2d



if __name__ == '__main__':
    bands = ['F200W', 'F444W']
    names = ['wide', 'deep']
    epochs = ['01', '03']
    apers = [0.2, 0.3, 0.5] #arcsec
    
    name_prefix = '{}{}_{}{}'.format(names[0], epochs[0], names[1], epochs[1])
    matching_radius = 0.1 #arcsec
    

    maindir = '/data6/stone28/nexus/'
    corr_dir = '/data6/stone28/nexus/correction_curves_{}/'.format(name_prefix)
    fname_edr = '/data3/web/nexus/edr/nircam/catalog/wide_ep1_01_src_catalog_v1.fits'
    fname_zcat = '/data3/web/nexus_collab/nircam/catalogs/nexus_Wide_ep01+Deep_ep01+ep02_nircam_catalog_with_zphot.fits'
    zcat = Table.read(fname_zcat)
    
    fname_vi_f200w = maindir + 'nexus_{}_nuclear_variability_cutouts_VI/bad_sources_F200W.txt'.format(name_prefix)
    fname_vi_f444w = maindir + 'nexus_{}_nuclear_variability_cutouts_VI/bad_sources_F444W.txt'.format(name_prefix)

    dat_f200w_stacked = Table.read(maindir + 'nexus_{}_stacked_sources_{}.fits'.format(name_prefix, bands[0]))
    dat_f444w_stacked = Table.read(maindir + 'nexus_{}_stacked_sources_{}.fits'.format(name_prefix, bands[1]))

    corrdat_dm_mean = Table.read(corr_dir + 'corr_dm_mean.fits')
    corrdat_dm_std = Table.read(corr_dir + 'corr_dm_std.fits')
    corrdat_dmerr_med = Table.read(corr_dir + 'corr_dmerr_med.fits')
    
    ##################################
    #Combine catalogs
    
    print('Combining catalogs')
    
    cols = ['FLUX_AUTO', 'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER', 'ELONGATION', 'FLAGS', 'IMAFLAGS_ISO', 'CLASS_STAR', 'ISOAREA_IMAGE', 'DM_AUTO', 'DMERR_AUTO', 'DM_APER', 'DMERR_APER',
            'DM_FINAL', 'DMERR_FINAL', 'DM_FINAL_TYPE', 'DIFF_MAG_FINAL', 'DIFF_FLUX_FINAL', 'DIFF_FLUXERR_FINAL']
    suffixes = ['DiffStackSub', 'DiffStackSFFT', 'REFStack', 'SCIStack']
    suffixes_new = ['DIRECTSUB', 'SFFT', 'REF', 'SCI']


    #Match using coords
    ind1_f200w, ind2_f200w, d2d_f200w = match_tables(dat_f200w_stacked, zcat, matching_radius=matching_radius)
    ind1_f444w, ind2_f444w, d2d_f444w = match_tables(dat_f444w_stacked, zcat, matching_radius=matching_radius)

    dat_f200w_stacked = dat_f200w_stacked[ind2_f200w].copy()
    dat_f444w_stacked = dat_f444w_stacked[ind2_f444w].copy()
    dat_f200w_stacked['ID'] = zcat['id'][ind1_f200w].data
    dat_f444w_stacked['ID'] = zcat['id'][ind1_f444w].data



    id_f200w = dat_f200w_stacked['ID'].data
    id_f444w = dat_f444w_stacked['ID'].data

    match_ind1 = []
    match_ind2 = []
    nomatch_ind1 = []
    nomatch_ind2 = []
    for i, id1 in enumerate(id_f200w):
        if id1 in id_f444w:
            match_ind1.append(i)        
            ind2 = np.argwhere(id_f444w == id1).flatten()[0]
            match_ind2.append(ind2)
        else:
            nomatch_ind1.append(i)

    for i, id2 in enumerate(id_f444w):
        if id2 in id_f200w:
            continue
        else:
            nomatch_ind2.append(i)

    matched_f200w = dat_f200w_stacked[match_ind1].copy()
    matched_f444w = dat_f444w_stacked[match_ind2].copy()
    unmatched_f200w = dat_f200w_stacked[nomatch_ind1].copy()
    unmatched_f444w = dat_f444w_stacked[nomatch_ind2].copy()



    #Matched
    tab1 = Table()
    tab1['RA'] = np.nanmedian( np.vstack([matched_f200w['RA_REFStack'].data,  matched_f444w['RA_REFStack'].data, matched_f200w['RA_DiffStackSub'], matched_f444w['RA_DiffStackSub']]), axis=0)
    tab1['DEC'] = np.nanmedian( np.vstack([matched_f200w['DEC_REFStack'].data,  matched_f444w['DEC_REFStack'].data, matched_f200w['DEC_DiffStackSub'], matched_f444w['DEC_DiffStackSub']]), axis=0)
    tab1['RA_F200W_EPOCH'] = matched_f200w['RA_REFStack'].data
    tab1['DEC_F200W_EPOCH'] = matched_f200w['DEC_REFStack'].data
    tab1['RA_F444W_EPOCH'] = matched_f444w['RA_REFStack'].data
    tab1['DEC_F444W_EPOCH'] = matched_f444w['DEC_REFStack'].data
    tab1['RA_F200W_DIFF'] = matched_f200w['RA_DiffStackSub'].data
    tab1['DEC_F200W_DIFF'] = matched_f200w['DEC_DiffStackSub'].data
    tab1['RA_F444W_DIFF'] = matched_f444w['RA_DiffStackSub'].data
    tab1['DEC_F444W_DIFF'] = matched_f444w['DEC_DiffStackSub'].data

    tab1['SATURATED_F200W'] = matched_f200w['SATURATED'].data
    tab1['SATURATED_F444W'] = matched_f444w['SATURATED'].data
    tab1['ID'] = matched_f200w['ID'].data
    tab1['OBSERVED_F200W'] = np.ones(len(matched_f200w), dtype=bool)
    tab1['OBSERVED_F444W'] = np.ones(len(matched_f444w), dtype=bool)

    for col in cols:
        for s, snew in zip(suffixes, suffixes_new):
            
            if ('DM' in col) and (s in ['REFStack', 'SCIStack']):
                continue
            
            if 'FINAL' in col:
                continue

            tab1[col + '_F200W_' + snew] = matched_f200w[col + '_' + s].data
            tab1[col + '_F444W_' + snew] = matched_f444w[col + '_' + s].data
            

    for col in cols:
        if not ('FINAL' in col):
            continue
        
        tab1[col + '_F200W'] = matched_f200w[col].data
        tab1[col + '_F444W'] = matched_f444w[col].data
            
            

    #Only in F200W
    tab2 = Table()
    tab2['RA'] = np.nanmedian( np.vstack([unmatched_f200w['RA_REFStack'].data, unmatched_f200w['RA_DiffStackSub']]), axis=0)
    tab2['DEC'] = np.nanmedian( np.vstack([unmatched_f200w['DEC_REFStack'].data, unmatched_f200w['DEC_DiffStackSub']]), axis=0)
    tab2['RA_F200W_EPOCH'] = unmatched_f200w['RA_REFStack'].data
    tab2['DEC_F200W_EPOCH'] = unmatched_f200w['DEC_REFStack'].data
    tab2['RA_F200W_DIFF'] = unmatched_f200w['RA_DiffStackSub'].data
    tab2['DEC_F200W_DIFF'] = unmatched_f200w['DEC_DiffStackSub'].data
    tab2['RA_F444W_EPOCH'] = np.full(len(unmatched_f200w), np.nan, dtype=float)
    tab2['DEC_F444W_EPOCH'] = np.full(len(unmatched_f200w), np.nan, dtype=float)
    tab2['RA_F444W_DIFF'] = np.full(len(unmatched_f200w), np.nan, dtype=float)
    tab2['DEC_F444W_DIFF'] = np.full(len(unmatched_f200w), np.nan, dtype=float)

    tab2['SATURATED_F200W'] = unmatched_f200w['SATURATED'].data
    tab2['SATURATED_F444W'] = np.zeros(len(unmatched_f200w), dtype=bool)
    tab2['ID'] = unmatched_f200w['ID'].data
    tab2['OBSERVED_F200W'] = np.ones(len(unmatched_f200w), dtype=bool)
    tab2['OBSERVED_F444W'] = np.zeros(len(unmatched_f200w), dtype=bool)


    for col in cols:
        for s, snew in zip(suffixes, suffixes_new):

            if ('DM' in col) and (s in ['REFStack', 'SCIStack']):
                continue

            if 'FINAL' in col:
                continue

            arr = unmatched_f200w[col + '_' + s].data



            tab2[col + '_F200W_' + snew] = arr
            
            if ('FLAG' in col) or ('AREA' in col):
                tab2[col + '_F444W_' + snew] = np.full_like(arr, -1, dtype=int)
            else:
                tab2[col + '_F444W_' + snew] = np.full_like(arr, np.nan, dtype=float)
                
    for col in cols:
        if not ('FINAL' in col):
            continue
        
        tab2[col + '_F200W'] = unmatched_f200w[col].data
        tab2[col + '_F444W'] = np.full_like(unmatched_f200w[col].data, np.nan, dtype=unmatched_f200w[col].data.dtype)
                
                

                
    #Only in F444W
    tab3 = Table()
    tab3['RA'] = np.nanmedian( np.vstack([unmatched_f444w['RA_REFStack'].data, unmatched_f444w['RA_DiffStackSub']]), axis=0)
    tab3['DEC'] = np.nanmedian( np.vstack([unmatched_f444w['DEC_REFStack'].data, unmatched_f444w['DEC_DiffStackSub']]), axis=0)
    tab3['RA_F200W_EPOCH'] = np.full(len(unmatched_f444w), np.nan, dtype=float)
    tab3['DEC_F200W_EPOCH'] = np.full(len(unmatched_f444w), np.nan, dtype=float)
    tab3['RA_F444W_EPOCH'] = unmatched_f444w['RA_REFStack'].data
    tab3['DEC_F444W_EPOCH'] = unmatched_f444w['DEC_REFStack'].data
    tab3['RA_F200W_DIFF'] = np.full(len(unmatched_f444w), np.nan, dtype=float)
    tab3['DEC_F200W_DIFF'] = np.full(len(unmatched_f444w), np.nan, dtype=float)
    tab3['RA_F444W_DIFF'] = unmatched_f444w['RA_DiffStackSub'].data
    tab3['DEC_F444W_DIFF'] = unmatched_f444w['DEC_DiffStackSub'].data

    tab3['SATURATED_F200W'] = np.zeros(len(unmatched_f444w), dtype=bool)
    tab3['SATURATED_F444W'] = unmatched_f444w['SATURATED'].data
    tab3['ID'] = unmatched_f444w['ID'].data
    tab3['OBSERVED_F200W'] = np.zeros(len(unmatched_f444w), dtype=bool)
    tab3['OBSERVED_F444W'] = np.ones(len(unmatched_f444w), dtype=bool)


    for col in cols:
        for s, snew in zip(suffixes, suffixes_new):

            if ('DM' in col) and (s in ['REFStack', 'SCIStack']):
                continue
            
            if 'FINAL' in col:
                continue

            arr = unmatched_f444w[col + '_' + s].data   
                
                
                
            tab3[col + '_F444W_' + snew] = arr
            
            if ('FLAG' in col) or ('AREA' in col):
                tab3[col + '_F200W_' + snew] = np.full_like(arr, -1, dtype=int)
            else:
                tab3[col + '_F200W_' + snew] = np.full_like(arr, np.nan, dtype=float)
                

    for col in cols:
        if not ('FINAL' in col):
            continue
        
        tab3[col + '_F200W'] = np.full_like(unmatched_f444w[col].data, np.nan, dtype=unmatched_f444w[col].data.dtype)
        tab3[col + '_F444W'] = unmatched_f444w[col].data
                

    tab2 = tab2[tab1.colnames].copy()
    tab3 = tab3[tab1.colnames].copy()

    tab_tot = vstack([tab1, tab2, tab3], join_type='exact')
    
    ##################################
    #Remove unneeded columns
    
    print('Renaming/removing columns')

    for b in bands:
        del tab_tot['ELONGATION_{}_SCI'.format(b)]
        tab_tot.rename_column('ELONGATION_{}_REF'.format(b), 'ELONGATION_{}_EPOCH'.format(b))
        del tab_tot['ELONGATION_{}_SFFT'.format(b)]
        tab_tot.rename_column('ELONGATION_{}_DIRECTSUB'.format(b), 'ELONGATION_{}_DIFF'.format(b))

    cols_rename = ['DM', 'DMERR', 'DIFF_FLUX', 'DIFF_FLUXERR', 'DIFF_MAG']
    for c in cols_rename:
        for b in bands:
            tab_tot.rename_column('{}_FINAL_{}'.format(c,b), '{}_{}_BEST'.format(c,b))

    for b in bands:
        tab_tot.rename_column('DIFF_FLUX_{}_BEST'.format(b), 'FLUX_{}_BEST'.format(b))
        tab_tot.rename_column('DIFF_FLUXERR_{}_BEST'.format(b), 'FLUXERR_{}_BEST'.format(b))
        del tab_tot['DIFF_MAG_{}_BEST'.format(b)]
            
    tab_tot.rename_column('DM_FINAL_TYPE_F200W', 'DM_F200W_BEST_TYPE')
    tab_tot.rename_column('DM_FINAL_TYPE_F444W', 'DM_F444W_BEST_TYPE')
    
    ##################################
    #Combine AUTO and APER columns

    print('Combining AUTO and APER columns')

    cols_combine = ['FLUX', 'FLUXERR']

    for b in bands:
        for dt in ['REF', 'SCI', 'DIRECTSUB', 'SFFT']:
        
            for c in cols_combine:
                arr_tot = np.full((len(tab_tot), 4), np.nan, dtype=float)
                
                arr_tot[:,0] = tab_tot['{}_AUTO_{}_{}'.format(c, b, dt)].data
                arr_tot[:,1:] = tab_tot['{}_APER_{}_{}'.format(c, b, dt)].data
                
                del tab_tot['{}_AUTO_{}_{}'.format(c, b, dt)]
                del tab_tot['{}_APER_{}_{}'.format(c, b, dt)]
                
                tab_tot['{}_{}_{}'.format(c, b, dt)] = arr_tot
            
            
    cols_combine = ['DM', 'DMERR']
    for b in bands:
        for dt in ['DIRECTSUB', 'SFFT']:
            
            for c in cols_combine:
                arr_tot = np.full((len(tab_tot), 4), np.nan, dtype=float)
                
                arr_tot[:,0] = tab_tot['{}_AUTO_{}_{}'.format(c, b, dt)].data
                arr_tot[:,1:] = tab_tot['{}_APER_{}_{}'.format(c, b, dt)].data

                del tab_tot['{}_AUTO_{}_{}'.format(c, b, dt)]
                del tab_tot['{}_APER_{}_{}'.format(c, b, dt)]

                tab_tot['{}_{}_{}'.format(c, b, dt)] = arr_tot
                
    ##################################
    # Add EPOCH DM, DMERR
    
    print('Adding EPOCH DM, DMERR')
    
    for b in bands:
        fr = tab_tot['FLUX_{}_REF'.format(b)].data
        fs = tab_tot['FLUX_{}_SCI'.format(b)].data
        ferr_r = tab_tot['FLUXERR_{}_REF'.format(b)].data
        ferr_s = tab_tot['FLUXERR_{}_SCI'.format(b)].data
        
        mr = -2.5*np.log10(fr) + 23.9
        ms = -2.5*np.log10(fs) + 23.9
        merr_r = (2.5/np.log(10)) * (ferr_r / fr)
        merr_s = (2.5/np.log(10)) * (ferr_s / fs)
        
        dm = ms - mr
        dmerr = np.sqrt(merr_r**2 + merr_s**2)

        tab_tot['DM_{}_EPOCH'.format(b)] = dm
        tab_tot['DMERR_{}_EPOCH'.format(b)] = dmerr
        
    ##################################
    #Get unbiased DM, DMERR
    
    print('Calculating unbiased DM, DMERR, and variability flag')
    
    mlims = [28., 29., 28.5, 28.]

    dt_corr = ['FINAL', 'SUB', 'SFFT', 'CAT']
    dt_cat = ['BEST', 'DIRECTSUB', 'SFFT', 'EPOCH']

    for i in range(len(bands)):
        for n, (dt1, dt2) in enumerate(zip(dt_corr, dt_cat)):
        
            outlier_mask = np.zeros((len(tab_tot), 4), dtype=bool)
            dm_mean_unbias = np.zeros((len(tab_tot), 4), dtype=float)
            dmerr_unbias = np.zeros((len(tab_tot), 4), dtype=float)
            
            sigma_dm = np.zeros((len(tab_tot), 4), dtype=float)
            

            for j in range(4):
                fr = tab_tot['FLUX_{}_REF'.format(bands[i])].data[:,j]        
                fs = tab_tot['FLUX_{}_SCI'.format(bands[i])].data[:,j]
                
                ferr_r = tab_tot['FLUXERR_{}_REF'.format(bands[i])].data[:,j]
                ferr_s = tab_tot['FLUXERR_{}_SCI'.format(bands[i])].data[:,j]
                    


                mag_r = -2.5 * np.log10(np.abs(fr)) + 23.9
                mag_s = -2.5 * np.log10(np.abs(fs)) + 23.9  
                
                magerr_r = np.abs(2.5 * ferr_r / (fr * np.log(10)))
                magerr_s = np.abs(2.5 * ferr_s / (fs * np.log(10)))   
                mag_avg = (mag_r + mag_s) / 2.  
                

                dm_diff = tab_tot['DM_{}_{}'.format(bands[i], dt2)].data[:,j]
                dmerr_diff = tab_tot['DMERR_{}_{}'.format(bands[i], dt2)].data[:,j]

                
                if j == 0:
                    aper_str = 'AUTO'
                else:
                    aper_str = '{:.1f}'.format(apers[j-1])
                
                mask = (corrdat_dm_mean['BAND'] == bands[i]) & (corrdat_dm_mean['APER'] == aper_str) & (corrdat_dm_mean['DIFF_TYPE'] == dt1)
                mstar_dm_mean = corrdat_dm_mean['MSTAR'].data[mask][0]
                coeffs0_dm_mean = corrdat_dm_mean['LINEAR_COEFFS'].data[mask][0]
                coeffs1_dm_mean = corrdat_dm_mean['POLY_COEFFS'].data[mask][0]
                
                mask = (corrdat_dm_std['BAND'] == bands[i]) & (corrdat_dm_std['APER'] == aper_str) & (corrdat_dm_std['DIFF_TYPE'] == dt1)
                mstar_dm_std = corrdat_dm_std['MSTAR'].data[mask][0]
                coeffs0_dm_std = corrdat_dm_std['LINEAR_COEFFS'].data[mask][0]
                coeffs1_dm_std = corrdat_dm_std['POLY_COEFFS'].data[mask][0]
                
                mask = (corrdat_dmerr_med['BAND'] == bands[i]) & (corrdat_dmerr_med['APER'] == aper_str) & (corrdat_dmerr_med['DIFF_TYPE'] == dt1)
                mstar_dmerr_med = corrdat_dmerr_med['MSTAR'].data[mask][0]
                coeffs0_dmerr_med = corrdat_dmerr_med['LINEAR_COEFFS'].data[mask][0]
                coeffs1_dmerr_med = corrdat_dmerr_med['POLY_COEFFS'].data[mask][0]
                
                
                #Get outlier mask/dm mean bias/dmerr med correction
                mask_dm_mean = mag_avg < mstar_dm_mean
                mask_dm_std = mag_avg < mstar_dm_std
                mask_dmerr_med = mag_avg < mstar_dmerr_med
                
                bias = np.zeros(len(mag_avg))
                dmerr_med = np.zeros(len(mag_avg))
                dm_std = np.zeros(len(mag_avg))

                #dm mean bias
                bias[mask_dm_mean] = coeffs0_dm_mean[0]*mag_avg[mask_dm_mean] + coeffs0_dm_mean[1]
                bias[~mask_dm_mean] = np.polyval(coeffs1_dm_mean, mag_avg[~mask_dm_mean])
                dm_mean_unbias[:,j] = dm_diff - bias

                    
                #dmerr med correction
                dmerr_med[mask_dmerr_med] = coeffs0_dmerr_med[0]*mag_avg[mask_dmerr_med] + coeffs0_dmerr_med[1]
                dmerr_med[~mask_dmerr_med] = np.polyval(coeffs1_dmerr_med, mag_avg[~mask_dmerr_med])
                
                dm_std[mask_dm_std] = coeffs0_dm_std[0]*mag_avg[mask_dm_std] + coeffs0_dm_std[1]
                dm_std[~mask_dm_std] = np.polyval(coeffs1_dm_std, mag_avg[~mask_dm_std])  

                        
                corr = dm_std - dmerr_med            
                dmerr_unbias[:,j] = np.nanmax([ dmerr_diff + corr, np.full(len(corr), 0.07) ], axis=0)
                    
                outlier_mask[:,j] = ( np.abs(dm_diff-bias) > 3*dm_std ) & (mag_avg <= mlims[j])
                sigma_dm[:,j] = dm_std
                    

            tab_tot['VARIABLE_{}_{}'.format(bands[i], dt2)] = outlier_mask
            tab_tot['DM_DEBIASED_{}_{}'.format(bands[i], dt2)] = dm_mean_unbias
            tab_tot['DMERR_DEBIASED_{}_{}'.format(bands[i], dt2)] = dmerr_unbias
            tab_tot['SIGMA_DM_{}_{}'.format(bands[i], dt2)] = sigma_dm
            
    ##################################
    # Match to EDR to get NID
    
    edr_dat = Table.read(fname_edr)

    coords_cat = SkyCoord(ra=tab_tot['RA'].data*u.degree, dec=tab_tot['DEC'].data*u.degree)
    coords_edr = SkyCoord(ra=edr_dat['RA'].data*u.degree, dec=edr_dat['DEC'].data*u.degree)
    
    idx, d2d, _ = coords_cat.match_to_catalog_sky(coords_edr)
    match_mask = d2d.arcsec < 0.1

    nid = np.zeros(len(tab_tot), dtype=int) - 1
    nid[match_mask] = edr_dat['ID'][idx[match_mask]]

    tab_tot['NID'] = nid
    
    print('\t {} not matched to EDR'.format(np.sum(nid == -1)))
    print('\t {} F200W variables not matched to EDR'.format(np.sum( (tab_tot['VARIABLE_F200W_BEST'].data[:,1]) & (nid == -1) )))
    print('\t {} F444W variables not matched to EDR'.format(np.sum( (tab_tot['VARIABLE_F444W_BEST'].data[:,1]) & (nid == -1) )))
    print('\t {} unique variables not matched to EDR'.format(np.sum( ((tab_tot['VARIABLE_F200W_BEST'].data[:,1]) | (tab_tot['VARIABLE_F444W_BEST'].data[:,1])) & (nid == -1) )))
    
    ##################################
    #Using ID, get redshift
    
    zspec_arr = np.full(len(tab_tot), np.nan, dtype=float)
    zphot_arr = np.full(len(tab_tot), np.nan, dtype=float)
    
    gmag_arr = np.full(len(tab_tot), np.nan, dtype=float)
    rmag_arr = np.full(len(tab_tot), np.nan, dtype=float)
    imag_arr = np.full(len(tab_tot), np.nan, dtype=float)
    zmag_arr = np.full(len(tab_tot), np.nan, dtype=float)
    ymag_arr = np.full(len(tab_tot), np.nan, dtype=float)
    
    f090w_mag = np.full(len(tab_tot), np.nan, dtype=float)
    f115w_mag = np.full(len(tab_tot), np.nan, dtype=float)
    f150w_mag = np.full(len(tab_tot), np.nan, dtype=float)
    f200w_mag = np.full(len(tab_tot), np.nan, dtype=float)
    f210m_mag = np.full(len(tab_tot), np.nan, dtype=float)
    f356w_mag = np.full(len(tab_tot), np.nan, dtype=float)
    f360m_mag = np.full(len(tab_tot), np.nan, dtype=float)
    f444w_mag = np.full(len(tab_tot), np.nan, dtype=float)
    
    for i in range(len(tab_tot)):
        
        ind = np.argwhere(zcat['id'] == tab_tot['ID'][i]).flatten()
        if len(ind) == 0:
            continue
            
        ind = ind[0]
        zphot = zcat['zphot'][ind]
        zspec = zcat['zspec'][ind]
        
        if zphot > 0.:
            zphot_arr[i] = zphot
        if zspec > 0.:
            zspec_arr[i] = zspec
            
        gmag_arr[i] = zcat['g_mag'][ind]
        rmag_arr[i] = zcat['r_mag'][ind]
        imag_arr[i] = zcat['i_mag'][ind]
        zmag_arr[i] = zcat['z_mag'][ind]
        ymag_arr[i] = zcat['y_mag'][ind]
        
        f090w_mag[i] = zcat['F090W_mag'][ind]
        f115w_mag[i] = zcat['F115W_mag'][ind]
        f150w_mag[i] = zcat['F150W_mag'][ind]
        f200w_mag[i] = zcat['F200W_mag'][ind]
        f210m_mag[i] = zcat['F210M_mag'][ind]
        f356w_mag[i] = zcat['F356W_mag'][ind]
        f360m_mag[i] = zcat['F360M_mag'][ind]
        f444w_mag[i] = zcat['F444W_mag'][ind]
            
    tab_tot['Z_PHOT'] = zphot_arr  
    tab_tot['Z_SPEC'] = zspec_arr      
    
    tab_tot['G_MAG'] = gmag_arr
    tab_tot['R_MAG'] = rmag_arr
    tab_tot['I_MAG'] = imag_arr
    tab_tot['Z_MAG'] = zmag_arr
    tab_tot['Y_MAG'] = ymag_arr
    
    tab_tot['F090W_MAG'] = f090w_mag
    tab_tot['F115W_MAG'] = f115w_mag
    tab_tot['F150W_MAG'] = f150w_mag
    tab_tot['F200W_MAG'] = f200w_mag
    tab_tot['F210M_MAG'] = f210m_mag
    tab_tot['F356W_MAG'] = f356w_mag
    tab_tot['F360M_MAG'] = f360m_mag
    tab_tot['F444W_MAG'] = f444w_mag
    
    ##################################
    #Get average mag
    
    for b in bands:
        fr = tab_tot['FLUX_{}_REF'.format(b)].data
        fs = tab_tot['FLUX_{}_SCI'.format(b)].data
        
        vals = -2.5*np.log10( np.abs(np.nanprod([fr,fs], axis=0)) )/2 + 23.9
        
        mask = tab_tot['OBSERVED_{}'.format(b)].data
        mag_avg = np.full(vals.shape, np.nan, dtype=float)
        mag_avg[mask] = vals[mask]
        
        tab_tot['MAG_AVG_{}'.format(b)] = mag_avg

    ##################################    
    #Rearrange columns
    
    print('Rearranging columns')

    cols_rearrange = ['ID', 'NID', 'RA', 'DEC', 'Z_PHOT', 'Z_SPEC', 
                      'G_MAG', 'R_MAG', 'I_MAG', 'Z_MAG', 'Y_MAG',
                      'F090W_MAG', 'F115W_MAG', 'F150W_MAG', 'F200W_MAG', 'F210M_MAG', 'F356W_MAG', 'F360M_MAG', 'F444W_MAG',
                       'OBSERVED_F200W', 'OBSERVED_F444W',
                       'RA_F200W_EPOCH', 'DEC_F200W_EPOCH', 'RA_F444W_EPOCH', 'DEC_F444W_EPOCH', 
                       'RA_F200W_DIFF', 'DEC_F200W_DIFF', 'RA_F444W_DIFF', 'DEC_F444W_DIFF',
                       'SATURATED_F200W', 'SATURATED_F444W']

    cols_rearrange += ['MAG_AVG_{}'.format(b) for b in bands]
    cols_rearrange += ['FLUX_{}_{}'.format(b, dt) for b in bands for dt in suffixes_new+['BEST']]
    cols_rearrange += ['FLUXERR_{}_{}'.format(b, dt) for b in bands for dt in suffixes_new+['BEST']]
    cols_rearrange += ['DM_{}_{}'.format(b, dt) for b in bands for dt in ['BEST', 'DIRECTSUB', 'SFFT', 'EPOCH']]
    cols_rearrange += ['DMERR_{}_{}'.format(b, dt) for b in bands for dt in ['BEST', 'DIRECTSUB', 'SFFT', 'EPOCH']]
    cols_rearrange += ['SIGMA_DM_{}_{}'.format(b, dt) for b in bands for dt in ['BEST', 'DIRECTSUB', 'SFFT', 'EPOCH']]
    cols_rearrange += ['DM_DEBIASED_{}_{}'.format(b, dt) for b in bands for dt in ['BEST', 'DIRECTSUB', 'SFFT', 'EPOCH']]
    cols_rearrange += ['DMERR_DEBIASED_{}_{}'.format(b, dt) for b in bands for dt in ['BEST', 'DIRECTSUB', 'SFFT', 'EPOCH']]
    cols_rearrange += ['DM_{}_BEST_TYPE'.format(b) for b in bands]
    cols_rearrange += ['VARIABLE_{}_{}'.format(b, dt) for b in bands for dt in ['BEST', 'DIRECTSUB', 'SFFT', 'EPOCH']]

    cols_rearrange += ['ELONGATION_{}_{}'.format(b, dt) for b in bands for dt in ['EPOCH', 'DIFF']]
    cols_rearrange += ['ISOAREA_IMAGE_{}_{}'.format(b, dt) for b in bands for dt in suffixes_new]
    cols_rearrange += ['FLAGS_{}_{}'.format(b, dt) for b in bands for dt in suffixes_new]
    cols_rearrange += ['IMAFLAGS_ISO_{}_{}'.format(b, dt) for b in bands for dt in suffixes_new]
    cols_rearrange += ['CLASS_STAR_{}_{}'.format(b, dt) for b in bands for dt in suffixes_new]

    tab_tot = tab_tot[cols_rearrange].copy()
    
    
    cols_dm = ['DM_{}_{}'.format(b, dt) for b in bands for dt in ['BEST', 'DIRECTSUB', 'SFFT', 'EPOCH']]
    cols_dmerr = ['DMERR_{}_{}'.format(b, dt) for b in bands for dt in ['BEST', 'DIRECTSUB', 'SFFT', 'EPOCH']]
    
    ##################################
    #Rename initial DM columns to DM_RAW
    
    for c in cols_dm:
        tab_tot.rename_column(c, c.replace('DM', 'DM_RAW'))
        cols_rearrange[cols_rearrange.index(c)] = c.replace('DM', 'DM_RAW')
    for c in cols_dmerr:
        tab_tot.rename_column(c, c.replace('DMERR', 'DMERR_RAW'))
        cols_rearrange[cols_rearrange.index(c)] = c.replace('DMERR', 'DMERR_RAW')
        
    tab_tot.rename_column('OBSERVED_F200W', 'TWO_EPOCHS_F200W')
    cols_rearrange[cols_rearrange.index('OBSERVED_F200W')] = 'TWO_EPOCHS_F200W'
    tab_tot.rename_column('OBSERVED_F444W', 'TWO_EPOCHS_F444W')
    cols_rearrange[cols_rearrange.index('OBSERVED_F444W')] = 'TWO_EPOCHS_F444W'
    
    ##################################
    #Get visual inspection flag
    
    if os.path.exists(fname_vi_f200w) and os.path.exists(fname_vi_f444w):
        bad_ids_f200w = np.loadtxt(fname_vi_f200w, dtype=int)
        bad_ids_f444w = np.loadtxt(fname_vi_f444w, dtype=int)
        
        vi_flag_f200w = np.zeros(len(tab_tot), dtype=bool)
        vi_flag_f444w = np.zeros(len(tab_tot), dtype=bool)
        for i in range(len(tab_tot)):
            if tab_tot['ID'][i] in bad_ids_f200w:
                continue
            else: 
                vi_flag_f200w[i] = tab_tot['VARIABLE_F200W_BEST'].data[i,1]
                
            if tab_tot['ID'][i] in bad_ids_f444w:
                continue
            else: 
                vi_flag_f444w[i] = tab_tot['VARIABLE_F444W_BEST'].data[i,1]
        
        tab_tot['MASK_VISUAL_INSPECTION_F200W'] = vi_flag_f200w
        tab_tot['MASK_VISUAL_INSPECTION_F444W'] = vi_flag_f444w
        
    else:
        print('No visual inspection files found; setting all to False')
        tab_tot['MASK_VISUAL_INSPECTION_F200W'] = np.zeros(len(tab_tot), dtype=bool)
        tab_tot['MASK_VISUAL_INSPECTION_F444W'] = np.zeros(len(tab_tot), dtype=bool)
    
    ####################################
    #Put in fiducial values
    
    for b in bands:
        #Flux
        tab_tot['REF_FLUX_{}'.format(b)] = tab_tot['FLUX_{}_REF'.format(b)].data[:,1]
        tab_tot['SCI_FLUX_{}'.format(b)] = tab_tot['FLUX_{}_SCI'.format(b)].data[:,1]
        tab_tot['DIFF_FLUX_{}'.format(b)] = tab_tot['FLUX_{}_BEST'.format(b)].data[:,1]      
        
        #Fluxerr
        tab_tot['REF_FLUXERR_{}'.format(b)] = tab_tot['FLUXERR_{}_REF'.format(b)].data[:,1]
        tab_tot['SCI_FLUXERR_{}'.format(b)] = tab_tot['FLUXERR_{}_SCI'.format(b)].data[:,1]
        tab_tot['DIFF_FLUXERR_{}'.format(b)] = tab_tot['FLUXERR_{}_BEST'.format(b)].data[:,1]   
        
        #DM
        tab_tot['DM_RAW_{}'.format(b)] = tab_tot['DM_RAW_{}_BEST'.format(b)].data[:,1]
        tab_tot['DMERR_RAW_{}'.format(b)] = tab_tot['DMERR_RAW_{}_BEST'.format(b)].data[:,1]
        tab_tot['SIGMA_DM_{}'.format(b)] = tab_tot['SIGMA_DM_{}_BEST'.format(b)].data[:,1]
        tab_tot['DM_DEBIASED_{}'.format(b)] = tab_tot['DM_DEBIASED_{}_BEST'.format(b)].data[:,1]
        tab_tot['DMERR_DEBIASED_{}'.format(b)] = tab_tot['DMERR_DEBIASED_{}_BEST'.format(b)].data[:,1]
        tab_tot['VARIABLE_{}'.format(b)] = tab_tot['VARIABLE_{}_BEST'.format(b)].data[:,1]  
        
    cols_rearrange += ['REF_FLUX_{}'.format(b) for b in bands]
    cols_rearrange += ['SCI_FLUX_{}'.format(b) for b in bands]
    cols_rearrange += ['DIFF_FLUX_{}'.format(b) for b in bands]
    cols_rearrange += ['REF_FLUXERR_{}'.format(b) for b in bands]
    cols_rearrange += ['SCI_FLUXERR_{}'.format(b) for b in bands]
    cols_rearrange += ['DIFF_FLUXERR_{}'.format(b) for b in bands]
    cols_rearrange += ['DM_RAW_{}'.format(b) for b in bands]
    cols_rearrange += ['DMERR_RAW_{}'.format(b) for b in bands]
    cols_rearrange += ['SIGMA_DM_{}'.format(b) for b in bands]
    cols_rearrange += ['DM_DEBIASED_{}'.format(b) for b in bands]
    cols_rearrange += ['DMERR_DEBIASED_{}'.format(b) for b in bands]
    cols_rearrange += ['VARIABLE_{}'.format(b) for b in bands]
    cols_rearrange += ['MASK_VISUAL_INSPECTION_{}'.format(b) for b in bands]

    tab_tot = tab_tot[cols_rearrange].copy()
    
    ##################################
    # Use only epoch RA/DEC

    tab_tot['RA'] = np.nanmedian( np.vstack([tab_tot['RA_F200W_EPOCH'].data, tab_tot['RA_F444W_EPOCH'].data]), axis=0)
    tab_tot['DEC'] = np.nanmedian( np.vstack([tab_tot['DEC_F200W_EPOCH'].data, tab_tot['DEC_F444W_EPOCH'].data]), axis=0)  

    ##################################
    #Save (full)
    
    print('Saving combined catalog to file')

    tab_tot.sort('ID')
    tab_tot.write(maindir + 'nexus_{}_stacked_sources_allband_internal.fits.gz'.format(name_prefix), 
                  overwrite=True)
    
    ##################################
    # Save (published)
    
    for b in bands:
        for t in ['EPOCH', 'DIFF']:
            del tab_tot['RA_{}_{}'.format(b, t)]
            del tab_tot['DEC_{}_{}'.format(b, t)]  
            
    tab_tot.write(maindir + 'nexus_{}_stacked_sources_allband.fits.gz'.format(name_prefix), 
                  overwrite=True)