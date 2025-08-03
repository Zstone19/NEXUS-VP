import os
from glob import glob
import multiprocessing as mp

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
import astropy.units as u
from astropy.coordinates import SkyCoord

from psf_utils import run_sextractor, get_spatially_constant_psf, estimate_psf_fwhm, \
    NIRCam_filter_FWHM_new, run_psfex, run_swarp, identify_stars,  get_psf_stars_photutils, \
    get_filter, get_varpsf_corner_positions, get_varpsf_center_positions, get_spatially_var_psf
    
    
    
## empirical PSF FWHM values measured from the Cycle 1 Absolute Flux calibration program, in unit of pixel
# https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions

SW = {
    # SW channel pixel size = 0.031
    'F070W'  : 0.935,
    'F090W'  : 1.065,
    'F115W'  : 1.290,
    'F140M'  : 1.548,
    'F150W'  : 1.613,
    'F162M'  : 1.774,
    'F164N'  : 1.806,
    'F150W2' : None,
    'F182M'  : 2.000,
    'F187N'  : 2.065,
    'F200W'  : 2.129,
    'F210M'  : 2.290,
    'F212N'  : 2.323,
}
LW = {
    # LW channel pixel size = 0.063
    'F250M'  : 1.349,
    'F277W'  : 1.460,
    'F300W'  : 1.587,
    'F322W2' : None,
    'F323N'  : 1.714,
    'F335M'  : 1.762,
    'F356W'  : 1.841,
    'F360M'  : 1.905,
    'F405N'  : 2.159,
    'F410M'  : 2.175,
    'F430M'  : 2.286,
    'F444W'  : 2.302,
    'F460M'  : 2.492,
    'F466N'  : 2.508,
    'F470N'  : 2.540,
    'F480M'  : 2.603
}
NIRCam_filter_FWHM = {
    'SW': SW, 'LW': LW
}

EUCLID = {'NIR_J': np.nan,
          'NIR_H': np.nan,
          'NIR_Y': np.nan}

SNR_WIN_JWST = 100
SNR_WIN_EUCLID = 1


#Steps:
#1. For each band:
#   A. Run SExtractor
#   B. Select point sources
#   C. Construct point source catalog
#   D. Find persistent sources 
#   E. Run PSFEx
#   F. Run photutils
#   G. Stack stars
#   H. Run SWarp
#   I. Update PSFEx psf
#   J. Check psf background

############################################################################################################
############################################################################################################
# Run SExtractor, PSFEx, SWarp, photutils

def run_sextractor_all(maindir, filtername_refgrid, SNR_WIN=100, CLASS_STAR=0.8, nsig=5., ncpu=1):
    paramdir = maindir + 'paramfiles/'
    
    fname_default = paramdir + 'default.sex'
    fname_param_sw = paramdir + 'default_SW.param'
    fname_param_lw = paramdir + 'default_LW.param'
    fname_param_euclid = paramdir + 'default_euclid.param'
    fname_conv = paramdir + 'gauss_4.0_7x7.conv'
    fname_nnw = paramdir + 'default.nnw'
    indir = maindir + 'input/'
    stardir = maindir + 'stars/'
    outdir = maindir + 'output/'

    fnames = glob(indir + '*_data*')
    filters = []
    for i, f in enumerate(fnames):
        filters.append(get_filter(f))
        
    if filtername_refgrid in NIRCam_filter_FWHM['SW'].keys():
        ps_out = .03
    elif filtername_refgrid in NIRCam_filter_FWHM['LW'].keys():
        ps_out = .03
    elif filtername_refgrid in EUCLID.keys():
        ps_out = .1            

        
    prefixes = []
    outdirs = []
    fnames_weight = []
    fnames_flag = []
    fwhm_vals = []
    for i in range(len(fnames)):
        list_of_str = os.path.basename(fnames[i]).split('.')[0].split('_')[:-1]
        prefixes.append('_'.join(list_of_str))
        
        outdirs.append(outdir + filters[i] + '/')    
        fnames_weight.append(fnames[i].replace('_data.fits', '_err.fits'))
        fnames_flag.append(fnames[i].replace('_data.fits', '_mask.fits'))
        
        if filters[i][0].upper() == 'F':
            fwhm_vals.append(None)
        elif filters[i] in EUCLID.keys():
            fname_psf = fnames[i].replace('_data.fits', '.psf')
            fwhm_vals.append(fits.open(fname_psf)[0].header['FWHM'])
    

    
    for i in range(len(filters)):
        output_fname = outdirs[i] + prefixes[i] + '.cat'
        
        if filters[i][0].upper() == 'F':
            SNR_WIN = SNR_WIN_JWST
        elif filters[i] in EUCLID.keys():
            SNR_WIN = SNR_WIN_EUCLID

        # generate config and run sextractor
        if not os.path.exists(output_fname):        
            run_sextractor(indir, outdirs[i], fname_default, 
                           fname_param_sw, fname_param_lw, fname_param_euclid, 
                           fname_conv, fname_nnw, 
                           os.path.basename(fnames[i]), os.path.basename(fnames_weight[i]), os.path.basename(fnames_flag[i]),
                           filters[i], prefixes[i], fwhm_vals[i], ps_out, nsig, ncpu)

        # preselect point-like sources
        identify_stars(output_fname, SNR_WIN=SNR_WIN, CLASS_STAR=CLASS_STAR)        
        
    return


def run_psfex_all(maindir, filtername_refgrid, oversampling_vals=None, ncpu=1):
    if oversampling_vals is None:
        oversampling_vals = [1]
        
    fname_default_c = maindir + 'paramfiles/default_c.psfex'
    fname_default_v = maindir + 'paramfiles/default_v.psfex'
    fname_default_v2 = maindir + 'paramfiles/default_v2.psfex'
    indir = maindir + 'input/'
    stardir = maindir + 'stars/'
    outdir = maindir + 'output/'
    
    
    
    fnames = glob(indir + '*_data*')
    filters = []
    for i, f in enumerate(fnames):
        filters.append(get_filter(f))
        
    if filtername_refgrid in NIRCam_filter_FWHM['SW'].keys():
        ps_out = .03
    elif filtername_refgrid in NIRCam_filter_FWHM['LW'].keys():
        ps_out = .03
    elif filtername_refgrid in EUCLID.keys():
        ps_out = .1   
        

    prefixes = []
    outdirs = []
    fnames_ps_candidate = []
    fnames_flag = []
    fwhm_vals = []
    for i in range(len(fnames)):
        list_of_str = os.path.basename(fnames[i]).split('.')[0].split('_')[:-1]
        prefixes.append('_'.join(list_of_str))
        
        outdirs.append(outdir + filters[i] + '/')    
        fnames_ps_candidate.append(outdirs[i] + prefixes[i] + '_pointsource_candidates.cat')
        fnames_flag.append(fnames[i].replace('_data.fits', '_mask.fits'))
        
        if filters[i][0].upper() == 'F':
            fwhm_vals.append(None)
        else:
            fname_psf = fnames[i].replace('_data.fits', '.psf')
            fwhm_vals.append( fits.open(fname_psf)[0].header['FWHM'] )
        

    for i in range(len(filters)):
        
        if filters[i][0].upper() == 'F':
            SNR_WIN = SNR_WIN_JWST
        elif filters[i] in EUCLID.keys():
            SNR_WIN = SNR_WIN_EUCLID      
        
        for j in range(len(oversampling_vals)):
            #SPATIALLY CONSTANT        
            fname_out = outdirs[i] + prefixes[i] + '_pointsource_candidates_c_{}.psf'.format(oversampling_vals[j])    
            if not os.path.exists(fname_out):
                run_psfex(outdirs[i], 
                          fnames_ps_candidate[i], fname_default_c, 
                          oversampling_vals[j], filters[i], prefixes[i], SNR_WIN, ps_out, True, False, ncpu)
            else:
                print('PSFEx already run for {}, oversampling={}, constant'.format(filters[i], oversampling_vals[j]))
                
            fname_out = outdirs[i] + prefixes[i] + '_psfex_PSF_c_{}'.format( str(oversampling_vals[j]) ) + '.fits'
            if not os.path.exists(fname_out):
                get_spatially_constant_psf(outdirs[i] + prefixes[i] + '_pointsource_candidates_c_{}.psf'.format(oversampling_vals[j]),
                                        outdirs[i] + prefixes[i] + '_psfex_PSF_c_{}.fits'.format(oversampling_vals[j]),
                                        filters[i], oversampling_vals[j], fwhm_vals[i], ps_out)
            else: 
                print('Constant PSF obtained already for {}, oversampling={}'.format(filters[i], oversampling_vals[j]))            
            
            #SPATIALLY VARIABLE
            fname_out = outdirs[i] + prefixes[i] + '_pointsource_candidates_v_{}.psf'.format(oversampling_vals[j])
            if not os.path.exists(fname_out):
                run_psfex(outdirs[i], 
                          fnames_ps_candidate[i], fname_default_v, 
                          oversampling_vals[j], filters[i], prefixes[i], SNR_WIN, ps_out, False, False, ncpu)
            else:
                print('PSFEx already run for {}, oversampling={}, variable'.format(filters[i], oversampling_vals[j]))
                
            fname_out = outdirs[i] + prefixes[i] + '_pointsource_candidates_v2_{}.psf'.format(oversampling_vals[j])
            if not os.path.exists(fname_out):
                run_psfex(outdirs[i], 
                          fnames_ps_candidate[i], fname_default_v2, 
                          oversampling_vals[j], filters[i], prefixes[i], SNR_WIN, ps_out, False, True, ncpu)
            else:
                print('PSFEx already run for {}, oversampling={}, variable2'.format(filters[i], oversampling_vals[j]))

            fname_out = outdirs[i] + prefixes[i] + '_psfex_PSF_v2_{}'.format( str(oversampling_vals[j]) ) + '.fits'
            if not os.path.exists(fname_out):
                corners = get_varpsf_corner_positions(fnames_flag[i])
                centers = get_varpsf_center_positions(corners)
                
                get_spatially_var_psf(centers['x'], centers['y'], 
                                    outdirs[i] + prefixes[i] + '_pointsource_candidates_v2_{}.psf'.format(oversampling_vals[j]), 
                                    outdirs[i] + prefixes[i] + '_psfex_PSF_v2_{}.fits'.format(oversampling_vals[j]), 
                                    filters[i], oversampling_vals[j], ps_out=ps_out)
            else:
                print('Variable PSF obtained already for {}, oversampling={}, variable2'.format(filters[i], oversampling_vals[j]))

    return


#NOT FINISHED
def run_swarp_all(maindir, oversampling_vals=None, fwhm_only=True):
    indir = maindir + 'input/'
    stardir = maindir + 'stars/'
    outdir = maindir + 'output/'
    
    fname_default = maindir + 'paramfiles/default.swarp'
    
    fnames = glob(indir + '*_data*')
    filters = []
    for f in fnames:
        filters.append(get_filter(f))
        
    prefixes = []
    outdirs = []
    stardirs = []
    for i in range(len(fnames)):
        list_of_str = os.path.basename(fnames[i]).split('.')[0].spilt('_')[:-1]
        prefixes.append('_'.join(list_of_str))        
        outdirs.append(outdir + filters[i] + '/')    
        stardirs.append(stardir + filters[i] + '/')
        
        
    for i in range(len(filters)):
        fname_stars = stardirs[i] + prefixes[i] + '_stars.ipac'
        _ = run_swarp(outdirs[i], stardirs[i], fname_default, fname_stars, 
                      filters[i], prefixes[i], oversampling_vals,  fwhm_only)

    return


#NOT FINISHED
def run_photutils_all(maindir, 
                      oversampling_vals=None, save_stars=False, fwhm_only=True, os_save_stars=1,
                      ncpu=1):
    
    if oversampling_vals is None:
        oversampling_vals = [2]
        
        
    indir = maindir + 'input/'
    stardir = maindir + 'stars/'
    outdir = maindir + 'output/'
    
    fnames = glob(indir + '*_data*')
    filters = []
    for f in fnames:
        filters.append(get_filter(f))
        
        
    prefixes = []
    outdirs = []
    stardirs = []
    fnames_flag = []
    fnames_sexcat = []
    fnames_pointsource_candidates = []
    fnames_sexseg = []
    for i in range(len(fnames)):
        list_of_str = os.path.basename(fnames[i]).split('.')[0].split('_')[:-1]
        prefixes.append('_'.join(list_of_str))        
        outdirs.append(outdir + filters[i] + '/')    
        stardirs.append(stardir + filters[i] + '/')
        
        fnames_flag.append(fnames[i].replace('_data.fits', '_mask.fits'))
        fnames_sexcat.append(outdirs[-1] + prefixes[i] + '_select.cat')
        fnames_pointsource_candidates.append(outdirs[-1] + prefixes[i] + '_pointsource_candidates.cat')
        fnames_sexseg.append(outdirs[-1] + prefixes[i] + '_sex_seg.fits')
    
        
    if not fwhm_only:
        for i in range(len(filters)):
            for j in range(len(oversampling_vals)):
                print('Getting photutils PSF for {}, os={}'.format(filters[i], oversampling_vals[j]))                
                get_psf_stars_photutils(outdirs[i], stardirs[i],
                                        fnames[i], fnames_flag[i],
                                        fnames_sexcat[i], fnames_pointsource_candidates[i], fnames_sexseg[i],
                                        filters[i], prefixes[i], oversampling_vals[j], save_stars, os_save_stars,
                                        ncpu)
                                        

    else:
        for i in range(len(filters)):
            for j in range(len(oversampling_vals)):
                ps = .03/oversampling_vals[j]
                
                if filters[i] in NIRCam_filter_FWHM['SW'].keys():
                    fwhm_estimate = NIRCam_filter_FWHM['SW'][filters[i]] * oversampling_vals[j]
                else:
                    fwhm_estimate = NIRCam_filter_FWHM['LW'][filters[i]] * oversampling_vals[j]*2
                    
                fname_psf = outdirs[i] + prefixes[i] + '_photutils_PSF_c_{}.fits'.format(oversampling_vals[j])
                fit_psf_and_update_header(fname_psf, fwhm_estimate, ps, centers=None)


    return

############################################################################################################
############################################################################################################
# Find point sources
            
def get_master_pointsource_catalog(maindir):
    
    indir = maindir + 'input/'
    outdir = maindir + 'output/'
    
    fnames_og = glob(indir + '*_data*')
    filters_og = []
    for f in fnames_og:
        filters_og.append(get_filter(f))
        
        
    #Sort by filter wl    
    wls = []
    for f in filters_og:
        if (f in NIRCam_filter_FWHM['SW'].keys()) or (f in NIRCam_filter_FWHM['LW'].keys()):
            wls.append(int(f[1:-1]))
        elif f in EUCLID.keys(): #approx equal to NIRCam
            if f == 'NIR_J':
                wls.append(115)
            if f == 'NIR_H':
                wls.append(150)
            if f == 'NIR_Y':
                wls.append(90)
            

    sort_ind = np.argsort(wls)
    fnames = [fnames_og[i] for i in sort_ind]
    filters = [filters_og[i] for i in sort_ind]
        
    prefixes = []
    outdirs = []
    fnames_weight = []
    fnames_flag = []
    for i in range(len(fnames)):
        list_of_str = os.path.basename(fnames[i]).split('.')[0].split('_')[:-1]
        prefixes.append('_'.join(list_of_str))
        
        outdirs.append(outdir + filters[i] + '/')    
        fnames_weight.append(fnames[i].replace('_data.fits', '_err.fits'))
        fnames_flag.append(fnames[i].replace('_data.fits', '_mask.fits'))
    
    
    
    if not os.path.exists(outdir + 'master_pointsource_candidates.ecsv'):
        #Need F070W to be first
        for i in range(len(filters)):
            fname_sexcat = outdirs[i] + prefixes[i] + '_select.cat'
            
            hdul = fits.open(fname_sexcat)
            catdat = Table(hdul[2].data)
            del catdat['VIGNET']
            
            fname_catout = outdirs[i] + prefixes[i] + '_select.ipac'
            catdat.write(fname_catout, format='ascii.ipac', overwrite=True)
            hdul.close()
            
            catdat['N'] = 2
            # if filters[i] == 'F070W':
            if i == 0:
                master_catdat = catdat['ALPHA_J2000', 'DELTA_J2000', 'N']
            else:
                master_coords = SkyCoord(master_catdat['ALPHA_J2000']*u.deg, master_catdat['DELTA_J2000']*u.deg)
                
                temp_coords = SkyCoord(catdat['ALPHA_J2000']*u.deg, catdat['DELTA_J2000']*u.deg)
                idx, d2d, _ = temp_coords.match_to_catalog_sky(master_coords)
                mask = d2d.arcsec < .1
                
                unmatched_dat = catdat[~mask]['ALPHA_J2000', 'DELTA_J2000', 'N']
                idx, d2d, _ = master_coords.match_to_catalog_sky(temp_coords)
                mask = d2d.arcsec < .1
                
                master_catdat['N'][mask] += 1
                master_catdat = vstack((master_catdat, unmatched_dat))
                
        fname_out = outdir + 'master_pointsource_candidates.ecsv'
        master_catdat.write(fname_out, format='ascii.ecsv', overwrite=True)
        
    else:
        master_catdat = Table.read(outdir + 'master_pointsource_candidates.ecsv', format='ascii.ecsv')
        
        
    
    master_catdat_highn = master_catdat[master_catdat['N'] > 1].copy()
    
    
    for i in range(len(filters)):
        if not os.path.exists(outdirs[i] + prefixes[i] + '_pointsource_candidates.cat'):
        
            fname_sexcat = outdirs[i] + prefixes[i] + '_select.cat'
            hdul = fits.open(fname_sexcat)
            catdat = Table(hdul[2].data)
            
            master_coords_highn = SkyCoord(master_catdat_highn['ALPHA_J2000']*u.deg, master_catdat_highn['DELTA_J2000']*u.deg)
            sex_coords = SkyCoord(catdat['ALPHA_J2000']*u.deg, catdat['DELTA_J2000']*u.deg)
            
            idx, d2d, _ = sex_coords.match_to_catalog_sky(master_coords_highn)
            mask = d2d.arcsec < .1
            
            hdul[2].data = hdul[2].data[mask]
            fname_out = outdirs[i] + prefixes[i] + '_pointsource_candidates.cat'
            hdul.writeto(fname_out, overwrite=True)


    return            
            
############################################################################################################
############################################################################################################