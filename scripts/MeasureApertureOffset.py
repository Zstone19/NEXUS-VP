import numpy as np

from p_tqdm import p_map
import warnings
warnings.filterwarnings('ignore')

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.nddata import Cutout2D


names = ['wide', 'deep']
epochs = ['01', '01']
prefix1 = '{}{}'.format(names[0], epochs[0])
prefix2 = '{}{}'.format(names[1], epochs[1])
name_prefix = '{}{}_{}{}'.format(names[0], epochs[0], names[1], epochs[1])


maindir = '/data6/stone28/nexus/'
sfftdir_f200w = maindir + 'sfft_nexus_{}_F200W/'.format(name_prefix)
sfftdir_f444w = maindir + 'sfft_nexus_{}_F444W/'.format(name_prefix)
subdir_f200w = maindir + 'zogy_nexus_{}_F200W/'.format(name_prefix)
subdir_f444w = maindir + 'zogy_nexus_{}_F444W/'.format(name_prefix)

with fits.open(sfftdir_f200w + 'output/nexus_{}_F200W.sfftdiff.decorr.combined.fits'.format(prefix2)) as hdul:
    im_f200w_sfftdiff = hdul[0].data
    wcs_f200w_diff = WCS(hdul[0].header)
with fits.open(sfftdir_f444w + 'output/nexus_{}_F444W.sfftdiff.decorr.combined.fits'.format(prefix2)) as hdul:
    im_f444w_sfftdiff = hdul[0].data
    wcs_f444w_diff = WCS(hdul[0].header)
with fits.open(subdir_f200w + 'output/nexus_{}_F200W.subdiff.fits'.format(prefix2)) as hdul:
    im_f200w_subdiff = hdul[0].data
with fits.open(subdir_f444w + 'output/nexus_{}_F444W.subdiff.fits'.format(prefix2)) as hdul:
    im_f444w_subdiff = hdul[0].data
    
mask_f200w = (im_f200w_sfftdiff == 0.) | np.isnan(im_f200w_sfftdiff) | (im_f200w_subdiff == 0.) | np.isnan(im_f200w_subdiff)
mask_f444w = (im_f444w_sfftdiff == 0.) | np.isnan(im_f444w_sfftdiff) | (im_f444w_subdiff == 0.) | np.isnan(im_f444w_subdiff)

im_f200w_sfftdiff[mask_f200w] = np.nan
im_f444w_sfftdiff[mask_f444w] = np.nan
im_f200w_subdiff[mask_f200w] = np.nan
im_f444w_subdiff[mask_f444w] = np.nan

catdat = Table.read(maindir + 'nexus_{}_stacked_sources_allband_internal.fits.gz'.format(name_prefix))


def get_coord_offsets(i):
    # objid = catdat['ID'][i]
    obs_f200w = catdat['TWO_EPOCHS_F200W'][i]
    obs_f444w = catdat['TWO_EPOCHS_F444W'][i]
    
    ra_diff_f200w = catdat['RA_F200W_DIFF'][i]
    dec_diff_f200w = catdat['DEC_F200W_DIFF'][i]
    ra_diff_f444w = catdat['RA_F444W_DIFF'][i]
    dec_diff_f444w = catdat['DEC_F444W_DIFF'][i]
    
    dm_f200w = catdat['DM_DEBIASED_F200W_BEST'][i,1]
    dm_f444w = catdat['DM_DEBIASED_F444W_BEST'][i,1]
    
    ps_f200w = .03
    ps_f444w = .06
    
    nside_f200w = int(np.ceil(.5/ps_f200w))
    nside_f444w = int(np.ceil(.5/ps_f444w))
    size_f200w = (nside_f200w, nside_f200w)
    size_f444w = (nside_f444w, nside_f444w)
     

    #Get pixel for peak flux in 0.2" aperture
    if obs_f200w:
        coords = SkyCoord(ra=ra_diff_f200w*u.deg, dec=dec_diff_f200w*u.deg)

        #Get cutout
        cutout_sfft = Cutout2D(im_f200w_sfftdiff, coords, size_f200w, wcs=wcs_f200w_diff)
        cutout_sub = Cutout2D(im_f200w_subdiff, coords, size_f200w, wcs=wcs_f200w_diff)

        sfft_fine = True
        sub_fine = True
        if np.all(np.isnan(cutout_sfft.data)) | np.all(cutout_sfft.data == 0.):
            sfft_fine = False
        if np.all(np.isnan(cutout_sub.data)) | np.all(cutout_sub.data == 0.):
            sub_fine = False


        #Max flux pixel
        if dm_f200w < 0:
            if sfft_fine:
                ypeak_sfft, xpeak_sfft = np.unravel_index( np.nanargmax(cutout_sfft.data), cutout_sfft.data.shape )
            if sub_fine:
                ypeak_sub, xpeak_sub = np.unravel_index( np.nanargmax(cutout_sub.data), cutout_sub.data.shape )
        else:
            if sfft_fine:
                ypeak_sfft, xpeak_sfft = np.unravel_index( np.nanargmin(cutout_sfft.data), cutout_sfft.data.shape )
            if sub_fine:
                ypeak_sub, xpeak_sub = np.unravel_index( np.nanargmin(cutout_sub.data), cutout_sub.data.shape )

        if sfft_fine:
            ra_f200w_sfft, dec_f200w_sfft = cutout_sfft.wcs.wcs_pix2world(xpeak_sfft, ypeak_sfft, 0)
            coord_sfft = SkyCoord(ra=ra_f200w_sfft*u.deg, dec=dec_f200w_sfft*u.deg)
        else:
            ra_f200w_sfft, dec_f200w_sfft = np.nan, np.nan
            
        if sub_fine:
            ra_f200w_sub, dec_f200w_sub = cutout_sub.wcs.wcs_pix2world(xpeak_sub, ypeak_sub, 0)
            coord_sub = SkyCoord(ra=ra_f200w_sub*u.deg, dec=dec_f200w_sub*u.deg)
        else:
            ra_f200w_sub, dec_f200w_sub = np.nan, np.nan

        #Offset in arcsec
        if sfft_fine:
            # offset_f200w_sfft = np.sqrt( (ra_f200w_sfft - ra_diff_f200w)**2 + (dec_f200w_sfft - dec_diff_f200w)**2 )*3600
            offset_f200w_sfft = coord_sfft.separation(coords).arcsec
        else:
            offset_f200w_sfft = np.nan
            
        if sub_fine:
            # offset_f200w_sub = np.sqrt( (ra_f200w_sub - ra_diff_f200w)**2 + (dec_f200w_sub - dec_diff_f200w)**2 )*3600
            offset_f200w_sub = coord_sub.separation(coords).arcsec
        else:
            offset_f200w_sub = np.nan
        
    else:
        ra_f200w_sfft = np.nan
        dec_f200w_sfft = np.nan
        ra_f200w_sub = np.nan
        dec_f200w_sub = np.nan
        offset_f200w_sfft = np.nan
        offset_f200w_sub = np.nan
        
        


    if obs_f444w:
        coords = SkyCoord(ra=ra_diff_f444w*u.deg, dec=dec_diff_f444w*u.deg)
        
        #Get cutout
        cutout_sfft = Cutout2D(im_f444w_sfftdiff, coords, size_f444w, wcs=wcs_f444w_diff)
        cutout_sub = Cutout2D(im_f444w_subdiff, coords, size_f444w, wcs=wcs_f444w_diff)
        
        sfft_fine = True
        sub_fine = True
        if np.all(np.isnan(cutout_sfft.data)) | np.all(cutout_sfft.data == 0.):
            sfft_fine = False
        if np.all(np.isnan(cutout_sub.data)) | np.all(cutout_sub.data == 0.):
            sub_fine = False

        #Max flux pixel
        if dm_f444w < 0:
            if sfft_fine:
                ypeak_sfft, xpeak_sfft = np.unravel_index( np.nanargmax(cutout_sfft.data), cutout_sfft.data.shape )
            if sub_fine:
                ypeak_sub, xpeak_sub = np.unravel_index( np.nanargmax(cutout_sub.data), cutout_sub.data.shape )
        else:
            if sfft_fine:
                ypeak_sfft, xpeak_sfft = np.unravel_index( np.nanargmin(cutout_sfft.data), cutout_sfft.data.shape )
            if sub_fine:
                ypeak_sub, xpeak_sub = np.unravel_index( np.nanargmin(cutout_sub.data), cutout_sub.data.shape )

        if sfft_fine:
            ra_f444w_sfft, dec_f444w_sfft = cutout_sfft.wcs.wcs_pix2world(xpeak_sfft, ypeak_sfft, 0)
            coord_sfft = SkyCoord(ra=ra_f444w_sfft*u.deg, dec=dec_f444w_sfft*u.deg)
        else:
            ra_f444w_sfft, dec_f444w_sfft = np.nan, np.nan    
        
        if sub_fine:
            ra_f444w_sub, dec_f444w_sub = cutout_sub.wcs.wcs_pix2world(xpeak_sub, ypeak_sub, 0)
            coord_sub = SkyCoord(ra=ra_f444w_sub*u.deg, dec=dec_f444w_sub*u.deg)
        else:
            ra_f444w_sub, dec_f444w_sub = np.nan, np.nan

        #Offset in arcsec
        if sfft_fine:
            # offset_f444w_sfft = np.sqrt( (ra_f444w_sfft - ra_diff_f444w)**2 + (dec_f444w_sfft - dec_diff_f444w)**2 )*3600
            offset_f444w_sfft = coord_sfft.separation(coords).arcsec
        else:
            offset_f444w_sfft = np.nan            
            
        if sub_fine:
            # offset_f444w_sub = np.sqrt( (ra_f444w_sub - ra_diff_f444w)**2 + (dec_f444w_sub - dec_diff_f444w)**2 )*3600
            offset_f444w_sub = coord_sub.separation(coords).arcsec
        else:
            offset_f444w_sub = np.nan
        
    else:
        ra_f444w_sfft = np.nan
        dec_f444w_sfft = np.nan
        ra_f444w_sub = np.nan
        dec_f444w_sub = np.nan
        offset_f444w_sfft = np.nan
        offset_f444w_sub = np.nan
        
    coords_f200w_sfft = [ra_f200w_sfft, dec_f200w_sfft]
    coords_f200w_sub = [ra_f200w_sub, dec_f200w_sub]
    coords_f444w_sfft = [ra_f444w_sfft, dec_f444w_sfft]
    coords_f444w_sub = [ra_f444w_sub, dec_f444w_sub]
    offset_f200w = [offset_f200w_sfft, offset_f200w_sub]
    offset_f444w = [offset_f444w_sfft, offset_f444w_sub]

    return offset_f200w, offset_f444w, coords_f200w_sfft, coords_f200w_sub, coords_f444w_sfft, coords_f444w_sub

output = p_map(get_coord_offsets, range(len(catdat)), num_cpus=60)


dist_offsets_f200w = np.array([out[0] for out in output])
dist_offsets_f444w = np.array([out[1] for out in output])
coords_f200w_sfft = np.array([out[2] for out in output])
coords_f200w_sub = np.array([out[3] for out in output])
coords_f444w_sfft = np.array([out[4] for out in output])
coords_f444w_sub = np.array([out[5] for out in output])



#Save 
outdat = Table()
outdat['ID'] = catdat['ID']
outdat['RA_F200W'] = catdat['RA_F200W_DIFF']
outdat['DEC_F200W'] = catdat['DEC_F200W_DIFF']
outdat['RA_F444W'] = catdat['RA_F444W_DIFF']
outdat['DEC_F444W'] = catdat['DEC_F444W_DIFF']
outdat['TWO_EPOCHS_F200W'] = catdat['TWO_EPOCHS_F200W']
outdat['TWO_EPOCHS_F444W'] = catdat['TWO_EPOCHS_F444W']

outdat['DIST_F200W_SFFT'] = dist_offsets_f200w[:,0]
outdat['DIST_F200W_SUB'] = dist_offsets_f200w[:,1]
outdat['DIST_F444W_SFFT'] = dist_offsets_f444w[:,0]
outdat['DIST_F444W_SUB'] = dist_offsets_f444w[:,1]

outdat['RA_PEAK_F200W_SFFT'] = coords_f200w_sfft[:,0]
outdat['DEC_PEAK_F200W_SFFT'] = coords_f200w_sfft[:,1]
outdat['RA_PEAK_F200W_SUB'] = coords_f200w_sub[:,0]
outdat['DEC_PEAK_F200W_SUB'] = coords_f200w_sub[:,1]
outdat['RA_PEAK_F444W_SFFT'] = coords_f444w_sfft[:,0]
outdat['DEC_PEAK_F444W_SFFT'] = coords_f444w_sfft[:,1]
outdat['RA_PEAK_F444W_SUB'] = coords_f444w_sub[:,0]
outdat['DEC_PEAK_F444W_SUB'] = coords_f444w_sub[:,1]

outdat.write(maindir + 'nexus_{}_aperture_offsets.csv'.format(name_prefix), format='csv', overwrite=True)