import shutil
import os
import subprocess
import logging
#import ctypes
#from functools import partial

import toml
# from mpire import WorkerPool
# import multiprocessing as mp
# from tqdm import tqdm
# from numba import njit, prange
# from numba_progress import ProgressBar

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.convolution import convolve_fft


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
SW_new = {
    # SW channel pixel size = 0.031
    'F070W'  : 2.1,
    'F090W'  : 2.1,
    'F115W'  : 2.1,
    'F150W'  : 2.1,
    'F200W'  : 2.5,
}
LW_new = {
    # LW channel pixel size = 0.063
    'F277W'  : 1.87,
    'F356W'  : 2.2,
    'F444W'  : 2.5,
    'F480M'  : 2.8
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
NIRCam_filter_FWHM_new = {
    'SW': SW_new, 'LW': LW_new
}

EUCLID = {
    'NIR_J': np.nan,
    'NIR_H': np.nan,
    'NIR_Y': np.nan
}



def setup_logger(name, log_file=None, level=logging.INFO):
    """To setup as many loggers as you want"""

    formatter = logging.Formatter('[%(asctime)s] [%(name)-25s] [%(levelname)s] %(message)s')
    
    if log_file is not None:
        handler = logging.FileHandler(log_file)        
    else:
        handler = logging.StreamHandler()
    
    handler.setFormatter(formatter)      

    logger = logging.getLogger(name)
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)

    logger.setLevel(level)
    
    if not len(logger.handlers): 
        logger.addHandler(handler)

    return handler, logger


def reset_logger(logger):
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    return



def get_mag_zp(fname_data, channel):
    
    with fits.open(fname_data) as hdu:
        hdr = hdu[0].header
    
    if 'MAGZERO' in hdr.keys():
        return hdr['MAGZERO']
    
    # elif channel in ['SW', 'LW']:
    #     PIXAR_SR = hdr['PIXAR_SR']  # Nominal pixel area in steradians
    #     mag_zp = -2.5 * np.log10(PIXAR_SR * 1e6) + 8.9
        
    #     exptime = hdr['XPOSURE']
    #     photmjsr = hdr['PHOTMJSR']
        
    #     zp_new = mag_zp + 2.5 * np.log10(exptime/photmjsr)

    # return zp_new
    
    
########################################################################################################################
########################################################################################################################
########################################################################################################################
#Main function for SExtractor

def run_sextractor(diff_outdir, ref_name, sci_name, outdir, paramdir, filtername_grid, logger, 
                   ra=None, dec=None, 
                   fwhm_px_grid=None, skysub=False, difftype='sfft',
                   gkerhw=11, 
                   nsig_input=2., nsig_diff=2., 
                   minarea_input=15, minarea_diff=15,
                   zp=23.9, ncpu=1):
    
    ### If ForceConv == REF, ZP=SCI
    ###    DIFF = SCI - (REF * MATCH_KERNEL)
    ### If ForceConv == SCI, ZP=REF    
    ###    DIFF = REF - (SCI * MATCH_KERNEL)
    
    npx_boundary = int(2*gkerhw)

    if (ra is not None) and (dec is not None):
        center_coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))



    if difftype == 'sfft':
        fname = diff_outdir + 'output/{}.sfftdiff.decorr.fits'.format(sci_name)
        fname_snr = diff_outdir + 'output/{}.sfftdiff.decorr.snr.fits'.format(sci_name)

        if skysub:
            fname_ref = diff_outdir + 'input/{}.skysub.fits'.format(ref_name)
        else:
            fname_ref = diff_outdir + 'input/{}.fits'.format(ref_name)

        if skysub:
            fname_sci = diff_outdir + 'input/{}.skysub.fits'.format(sci_name)
        else:
            fname_sci = diff_outdir + 'input/{}.fits'.format(sci_name)    
            
    elif difftype == 'sfft_combined':
        fname = diff_outdir + 'output/{}.sfftdiff.decorr.combined.fits'.format(sci_name)
        fname_snr = diff_outdir + 'output/{}.sfftdiff.decorr.snr.combined.fits'.format(sci_name)

        if skysub:
            fname_ref = diff_outdir + 'output/{}.skysub.combined.fits'.format(ref_name)
        else:
            fname_ref = diff_outdir + 'input/{}.fits'.format(ref_name)

        if skysub:
            fname_sci = diff_outdir + 'output/{}.skysub.combined.fits'.format(sci_name)
        else:
            fname_sci = diff_outdir + 'input/{}.fits'.format(sci_name)   
            
    elif difftype == 'sub':
        fname = diff_outdir + 'output/{}.subdiff.fits'.format(sci_name)     
        fname_ref = diff_outdir + 'input/{}.fits'.format(ref_name)
        fname_sci = diff_outdir + 'input/{}.fits'.format(sci_name)
        
        

    fname_ref_noise = diff_outdir + 'noise/{}.noise.fits'.format(ref_name)
    fname_sci_noise = diff_outdir + 'noise/{}.noise.fits'.format(sci_name)
    
    fname_ref_mask = diff_outdir + 'input/{}.maskin.fits'.format(ref_name)
    fname_sci_mask = diff_outdir + 'input/{}.maskin.fits'.format(sci_name)
        
    fname_ref_new = outdir + 'REF.fits'
    fname_sci_new = outdir + 'SCI.fits'
    fname_ref_noise_new = outdir + 'REF.noise.fits'
    fname_sci_noise_new = outdir + 'SCI.noise.fits'
    fname_diff_noise_new = outdir + 'DIFF.noise.fits'
    fname_maskin_new = outdir + 'maskin.fits'


        
    logger.info('Found input image: {}'.format(fname))        
    logger.info('Found REF image: {}'.format(fname_ref))
    logger.info('Found SCI image: {}'.format(fname_sci))
        
        
    #Make outdir
    os.makedirs(outdir, exist_ok=True)
    
    fname_in = outdir + 'DIFF.fits'
    fname_in_neg = outdir + 'DIFF.neg.fits'
    
    with fits.open(fname_ref) as hdu:
        im_ref = hdu[0].data
        hdr_ref = hdu[0].header
    with fits.open(fname_sci) as hdu:
        im_sci = hdu[0].data
    with fits.open(fname_ref_mask) as hdu:
        im_ref_mask = hdu[0].data.astype(bool)
    with fits.open(fname_sci_mask) as hdu:
        im_sci_mask = hdu[0].data.astype(bool)
        
    im_maskin = im_ref_mask | im_sci_mask
    im_maskin[im_ref == 0.] = True
    im_maskin[im_sci == 0.] = True
    im_maskin[np.isnan(im_ref)] = True
    im_maskin[np.isnan(im_sci)] = True
        
    if (ra is None) or (dec is None):
        wcs_r = WCS(hdr_ref)
        N0, N1 = im_ref.shape
        rac, decc = wcs_r.wcs_pix2world(N1/2, N0/2, 0)
        center_coords = SkyCoord(ra=rac, dec=decc, unit=(u.deg, u.deg))

    if difftype == 'sfft':
        cutout_shape = (np.array(im_ref.shape) - npx_boundary*2) * u.pixel
    elif difftype in ['sub', 'sfft_combined']:
        cutout_shape = np.array(im_ref.shape) * u.pixel
    
    ################################################################################################################################################################
    #Copy DIFF files
    with fits.open(fname) as hdu:
        im = hdu[0].data
    with fits.open(fname_ref) as hdu:
        hdr = hdu[0].header
    
    im[im_maskin] = np.nan
    
    wcs = WCS(hdr)
    cutout = Cutout2D(im, center_coords, cutout_shape, wcs=wcs)
    fits.writeto(fname_in, cutout.data, cutout.wcs.to_header(), overwrite=True)

    logger.info('Wrote DIFF image to {}'.format(fname_in))
    
    with fits.open(fname) as hdu:
        im = hdu[0].data * -1
    with fits.open(fname_ref) as hdu:
        hdr = hdu[0].header
        
    im[im_maskin] = np.nan

    wcs = WCS(hdr)    
    cutout = Cutout2D(im, center_coords, cutout_shape, wcs=wcs)
    fits.writeto(fname_in_neg, cutout.data, cutout.wcs.to_header(), overwrite=True)

    logger.info('Wrote negated DIFF image to {}'.format(fname_in_neg))
    

    ################################################################################################################################################################
    #Copy maskin files

    wcs = WCS(hdr_ref)
    cutout = Cutout2D(im_maskin.astype(np.int16), center_coords, cutout_shape, wcs=wcs)
    fits.writeto(fname_maskin_new, cutout.data, cutout.wcs.to_header(), overwrite=True)

    logger.info('Wrote maskin image to {}'.format(fname_maskin_new))
    
    ################################################################################################################################################################
    #Copy input image files
    with fits.open(fname_sci) as hdu:
        im = hdu[0].data
        hdr = hdu[0].header

    im[im_maskin] = np.nan

    wcs = WCS(hdr)
    cutout = Cutout2D(im, center_coords, cutout_shape, wcs=wcs)
    fits.writeto(fname_sci_new, cutout.data, cutout.wcs.to_header(), overwrite=True)
    
    logger.info('Wrote input image to {}'.format(fname_sci_new))
    
    
    with fits.open(fname_ref) as hdu:
        im = hdu[0].data
        hdr = hdu[0].header


    im[im_maskin] = np.nan

    wcs = WCS(hdr)
    cutout = Cutout2D(im, center_coords, cutout_shape, wcs=wcs)
    fits.writeto(fname_ref_new, cutout.data, cutout.wcs.to_header(), overwrite=True)
    
    logger.info('Wrote REF image to {}'.format(fname_ref_new))
    
    ################################################################################################################################################################
    #Copy input image noise files
    with fits.open(fname_sci_noise) as hdu:
        im = hdu[0].data
        hdr = hdu[0].header

    im[im_maskin] = np.nan

    wcs = WCS(hdr)
    cutout_s = Cutout2D(im, center_coords, cutout_shape, wcs=wcs)
    fits.writeto(fname_sci_noise_new, cutout_s.data, cutout_s.wcs.to_header(), overwrite=True)

    logger.info('Wrote SCI noise image to {}'.format(fname_sci_noise_new))
    
    
    with fits.open(fname_ref_noise) as hdu:
        im = hdu[0].data
        hdr = hdu[0].header
        
    im[im_maskin] = np.nan

    wcs = WCS(hdr)
    cutout_r = Cutout2D(im, center_coords, cutout_shape, wcs=wcs)
    fits.writeto(fname_ref_noise_new, cutout_r.data, cutout_r.wcs.to_header(), overwrite=True)

    logger.info('Wrote REF noise image to {}'.format(fname_ref_noise_new))
    
    
    
    if difftype == 'sfft':
        with fits.open(fname_snr) as hdu:
            im_snr = hdu[0].data
        with fits.open(fname) as hdu:
            im_diff = hdu[0].data
            wcs_diff = WCS(hdu[0].header)
            
        im_snr[im_maskin] = np.nan
        im_diff[im_maskin] = np.nan
        
        im_diff_noise = im_diff / im_snr
        cutout_d = Cutout2D(im_diff_noise, center_coords, cutout_shape, wcs=wcs_diff).data    
        
    elif difftype == 'sfft_combined':
        fname_noise = diff_outdir + 'output/{}.sfftdiff.decorr.noise.combined.fits'.format(sci_name)
        with fits.open(fname_noise) as hdu:
            im_diff_noise = hdu[0].data
            wcs_diff = WCS(hdu[0].header)
            
        cutout_d = Cutout2D(im_diff_noise, center_coords, cutout_shape, wcs=wcs_diff).data        
    
    elif difftype == 'sub':
        cutout_d = np.sqrt(cutout_s.data**2 + cutout_r.data**2)
                
    fits.writeto(fname_diff_noise_new, cutout_d, cutout_s.wcs.to_header(), overwrite=True)
    logger.info('Wrote DIFF image to {}'.format(fname_diff_noise_new))

    ################################################################################################################################################################

    #Get pixel scale
    filtername_grid = filtername_grid.upper()
    if filtername_grid in SW.keys():
        ps_grid = .031
        channel_grid = 'SW'
    elif filtername_grid in LW.keys():
        ps_grid = .063
        channel_grid = 'LW'        
    elif filtername_grid in EUCLID.keys():
        ps_grid = .1
        channel_grid = 'euclid'
        
    logger.info('Pixel scale: {} "/pix'.format(ps_grid))
    logger.info('Channel: {}'.format(channel_grid))
    
    
    

    #Move config files
    logger.info('Copying config files to {}'.format(outdir))
    shutil.copy(paramdir + 'default.sex', outdir + 'default.sex')
    shutil.copy(paramdir + 'default_dualmode.sex', outdir + 'default_dualmode.sex')
    shutil.copy(paramdir + 'gauss_4.0_7x7.conv', outdir)
    shutil.copy(paramdir + 'default.nnw', outdir)
    shutil.copy(paramdir + 'default_dualmode.param', outdir + 'default.param')


    fname_def = paramdir + 'default.sex'
    fname_def_dual = paramdir + 'default_dualmode.sex'
    
    with open(fname_def_dual, 'r') as f:
        content_r = f.readlines()
    with open(fname_def_dual, 'r') as f:
        content_s = f.readlines()

    with open(fname_def_dual, 'r') as f:
        content_neg_r = f.readlines()
    with open(fname_def_dual, 'r') as f:
        content_neg_s = f.readlines()

    with open(fname_def, 'r') as f:
        content = f.readlines()
    with open(fname_def, 'r') as f:
        content_neg = f.readlines()

    with open(fname_def, 'r') as f:
        content_og_r = f.readlines()
    with open(fname_def, 'r') as f:
        content_og_s = f.readlines()

    with open(fname_def_dual, 'r') as f:
        content_og_sr = f.readlines()
    with open(fname_def_dual, 'r') as f:
        content_og_rs = f.readlines()
    
    #Get photometric aperature size
    if channel_grid in ['SW', 'LW']:
        phot_aper = NIRCam_filter_FWHM_new[channel_grid][filtername_grid]*5*5
        seeing_fwhm = NIRCam_filter_FWHM[channel_grid][filtername_grid] #px
        
        phot_aper_og = NIRCam_filter_FWHM[channel_grid][filtername_grid] *5
        seeing_fwhm_og = NIRCam_filter_FWHM[channel_grid][filtername_grid] #px
    elif channel_grid == 'euclid':
        phot_aper = fwhm_px_grid * 5
        seeing_fwhm = fwhm_px_grid * .1

        phot_aper_og = fwhm_px_grid * 5
        seeing_fwhm_og = fwhm_px_grid * .1
        
    apers = ','.join(  (np.array([1,2,3,4])*seeing_fwhm).astype(str)  )
    apers += ',{}'.format(0.5/ps_grid)

    # seeing_fwhm = 0.158
    # seeing_fwhm_og = 0.158
        
    # logger.info('Photometric aperture size: {} px'.format(phot_aper))
     
    zp_r = zp
    zp_s = zp
    zp_d = zp
    
    # if ForceConv == 'REF':
    #     zp_d = zp_s
    # if ForceConv == 'SCI':
    #     zp_d = zp_r
    
    #Write SExtractor config file (OG FILES)
    for c, label, zpval in zip([content_og_r, content_og_s, content_og_sr, content_og_rs], ['REF', 'SCI', 'SCI_r', 'REF_s'], [zp_r, zp_s, zp_s, zp_r]):     
        if label == 'REF':
            wf = fname_ref_noise_new
            ff = fname_maskin_new
        elif label == 'SCI':
            wf = fname_sci_noise_new
            ff = fname_maskin_new
        elif label == 'SCI_r':
            wf = '{},{}'.format(fname_ref_noise_new, fname_sci_noise_new)
            ff = '{},{}'.format(fname_maskin_new, fname_maskin_new)
        elif label == 'REF_s':
            wf = '{},{}'.format(fname_sci_noise_new, fname_ref_noise_new)
            ff = '{},{}'.format(fname_maskin_new, fname_maskin_new)
        
           
        c[6]   = 'CATALOG_NAME     {}              # name of the output catalog\n'.format(outdir + '{}.cat'.format(label))
        c[7]   = "CATALOG_TYPE     FITS_1.0        # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,\n"
        c[9]   = "PARAMETERS_NAME  {}              # name of the file containing catalog contents\n".format(outdir + 'default.param')
        
        c[14]  = "DETECT_MINAREA   {}              # min. # of pixels above threshold\n".format(minarea_input)
        c[18]  = "DETECT_THRESH    {}              # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n".format(nsig_input)
        c[19]  = "ANALYSIS_THRESH  {}              # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n".format(nsig_input)
        c[22]  = "FILTER_NAME      {}              # name of the file containing the filter\n".format(outdir + 'gauss_4.0_7x7.conv')
        
        if label in ['REF', 'SCI']:
            c[36]  = 'WEIGHT_TYPE      MAP_RMS         # type of WEIGHTing: NONE, BACKGROUND,\n'
        else:
            c[36]  = 'WEIGHT_TYPE      MAP_RMS,MAP_RMS     # type of WEIGHTing: NONE, BACKGROUND,\n'
    
        c[38]  = "WEIGHT_IMAGE     {}              # weight-map filename\n".format(wf)
        c[45]  = 'FLAG_IMAGE       {}              # filename for an input FLAG-image\n'.format(ff)

        c[55]  = "PHOT_APERTURES   {}              # MAG_APER aperture diameter(s) in pixels\n".format(apers)  # 5 times FWHM
        c[63]  = "SATUR_LEVEL      1e9             # level (in ADUs) at which arises saturation\n"
        c[66]  = 'MAG_ZEROPOINT    {}              # magnitude zero-point\n'.format(zpval)
        c[74]  = 'SEEING_FWHM      {}              # stellar FWHM in arcsec\n'.format(seeing_fwhm*ps_grid)
        c[75]  = "STARNNW_NAME     {}              # Neural-Network_Weight table filename\n".format(outdir + 'default.nnw')

        c[95]  = 'CHECKIMAGE_NAME  {}              # Filename for the check-image\n'.format(outdir + '{}_sexseg.fits'.format(label))
        c[122] = 'NTHREADS         {}              # 1 single thread\n'.format(ncpu)
    
        
    fname_config_og_r = outdir + 'default_REF.sex'
    with open(fname_config_og_r, 'w') as f:
        f.writelines(content_og_r)
    logger.info('Wrote SExtractor config file to {}'.format(fname_config_og_r))
        
    fname_config_og_s = outdir + 'default_SCI.sex'
    with open(fname_config_og_s, 'w') as f:
        f.writelines(content_og_s)
    logger.info('Wrote SExtractor config file to {}'.format(fname_config_og_s))
    
    fname_config_og_sr = outdir + 'default_SCI_r.sex'
    with open(fname_config_og_sr, 'w') as f:
        f.writelines(content_og_sr)
    logger.info('Wrote SExtractor config file to {}'.format(fname_config_og_sr))
    
    fname_config_og_rs = outdir + 'default_REF_s.sex'
    with open(fname_config_og_rs, 'w') as f:
        f.writelines(content_og_rs)
    logger.info('Wrote SExtractor config file to {}'.format(fname_config_og_rs))
        
        
        
        
    #Write SExtractor config file (POSITIVE)
    for c, label in zip([content_r, content_s, content], ['r', 's', 'raw']):
        if label == 'r':
            wf = '{},{}'.format(fname_ref_noise_new, fname_diff_noise_new)
            ff = '{},{}'.format(fname_maskin_new, fname_maskin_new)
        elif label == 's':
            wf = '{},{}'.format(fname_sci_noise_new, fname_diff_noise_new)
            ff = '{},{}'.format(fname_maskin_new, fname_maskin_new)
        elif label in ['raw']:
            wf = fname_diff_noise_new
            ff = fname_maskin_new
            
        if label == 'raw':
            nsig = nsig_diff
            minarea = minarea_diff
        else:
            nsig = nsig_input
            minarea = minarea_input
        
        c[6]   = 'CATALOG_NAME     {}              # name of the output catalog\n'.format(outdir + 'DIFF_{}.cat'.format(label))
        c[7]   = "CATALOG_TYPE     FITS_1.0        # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,\n"
        c[9]   = "PARAMETERS_NAME  {}              # name of the file containing catalog contents\n".format(outdir + 'default.param')
        
        c[14]  = "DETECT_MINAREA   {}              # min. # of pixels above threshold\n".format(minarea)
        c[18]  = "DETECT_THRESH    {}              # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n".format(nsig)
        c[19]  = "ANALYSIS_THRESH  {}              # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n".format(nsig)
        c[22]  = "FILTER_NAME      {}              # name of the file containing the filter\n".format(outdir + 'gauss_4.0_7x7.conv')
        
        if label == 'raw':
            c[36]  = 'WEIGHT_TYPE      MAP_RMS         # type of WEIGHTing: NONE, BACKGROUND,\n'
        else:
            c[36]  = 'WEIGHT_TYPE      MAP_RMS,MAP_RMS         # type of WEIGHTing: NONE, BACKGROUND,\n'
        
        c[38]  = "WEIGHT_IMAGE     {}              # weight-map filename\n".format(wf)
        c[45]  = 'FLAG_IMAGE       {}              # filename for an input FLAG-image\n'.format(ff)

        c[55]  = "PHOT_APERTURES   {}              # MAG_APER aperture diameter(s) in pixels\n".format(apers)  # 5 times FWHM
        c[63]  = "SATUR_LEVEL      1e9             # level (in ADUs) at which arises saturation\n"
        c[66]  = 'MAG_ZEROPOINT    {}              # magnitude zero-point\n'.format(zp_d)
        c[74]  = 'SEEING_FWHM      {}              # stellar FWHM in arcsec\n'.format(seeing_fwhm*ps_grid)
        c[75]  = "STARNNW_NAME     {}              # Neural-Network_Weight table filename\n".format(outdir + 'default.nnw')

        c[95]  = 'CHECKIMAGE_NAME  {}              # Filename for the check-image\n'.format(outdir + 'DIFF_sexseg_{}.fits'.format(sci_name, label))
        c[122] = 'NTHREADS         {}              # 1 single thread\n'.format(ncpu)

    fname_config_r = outdir + 'default_DIFF_r.sex'
    with open(fname_config_r, 'w') as f:
        f.writelines(content_r)
    logger.info('Wrote SExtractor config file to {}'.format(fname_config_r))
        
    fname_config_s = outdir + 'default_DIFF_s.sex'
    with open(fname_config_s, 'w') as f:
        f.writelines(content_s)
    logger.info('Wrote SExtractor config file to {}'.format(fname_config_s))
    
    fname_config_raw = outdir + 'default_DIFF_raw.sex'
    with open(fname_config_raw, 'w') as f:
        f.writelines(content)
    logger.info('Wrote SExtractor config file to {}'.format(fname_config_raw))
        


    #Write SExtractor config file (NEGATIVE)
    for c, label in zip([content_neg_r, content_neg_s, content_neg], ['r', 's', 'raw']):
        if label == 'r':
            wf = '{},{}'.format(fname_ref_noise_new, fname_diff_noise_new)
            ff = '{},{}'.format(fname_maskin_new, fname_maskin_new)
        elif label == 's':
            wf = '{},{}'.format(fname_sci_noise_new, fname_diff_noise_new)
            ff = '{},{}'.format(fname_maskin_new, fname_maskin_new)
        elif label in ['raw']:
            wf = fname_diff_noise_new
            ff = fname_maskin_new
            
        if label == 'raw':
            nsig = nsig_diff
            minarea = minarea_diff
        else:
            nsig = nsig_input
            minarea = minarea_input
            


        c[6]   = 'CATALOG_NAME     {}              # name of the output catalog\n'.format(outdir + 'DIFF_{}.neg.cat'.format(label))
        c[7]   = "CATALOG_TYPE     FITS_1.0        # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,\n"
        c[9]   = "PARAMETERS_NAME  {}              # name of the file containing catalog contents\n".format(outdir + 'default.param')
        
        c[14]  = "DETECT_MINAREA   {}              # min. # of pixels above threshold\n".format(minarea)
        c[18]  = "DETECT_THRESH    {}              # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n".format(nsig)
        c[19]  = "ANALYSIS_THRESH  {}              # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n".format(nsig)
        c[22]  = "FILTER_NAME      {}              # name of the file containing the filter\n".format(outdir + 'gauss_4.0_7x7.conv')

        if label == 'raw':
            c[36]  = 'WEIGHT_TYPE      MAP_RMS             # type of WEIGHTing: NONE, BACKGROUND,\n'
        else:
            c[36]  = 'WEIGHT_TYPE      MAP_RMS,MAP_RMS         # type of WEIGHTing: NONE, BACKGROUND,\n'
        
        c[38]  = "WEIGHT_IMAGE     {}              # weight-map filename\n".format(wf) 
        c[45]  = 'FLAG_IMAGE       {}              # filename for an input FLAG-image\n'.format(ff)

        c[55]  = "PHOT_APERTURES   {}              # MAG_APER aperture diameter(s) in pixels\n".format(apers)  # 5 times FWHM
        c[63]  = "SATUR_LEVEL      1e9             # level (in ADUs) at which arises saturation\n"
        c[66]  = 'MAG_ZEROPOINT    {}              # magnitude zero-point\n'.format(zp_d)
        c[74]  = 'SEEING_FWHM      {}              # stellar FWHM in arcsec\n'.format(seeing_fwhm*ps_grid)
        c[75]  = "STARNNW_NAME     {}              # Neural-Network_Weight table filename\n".format(outdir + 'default.nnw')

        c[95]  = 'CHECKIMAGE_NAME  {}              # Filename for the check-image\n'.format(outdir + 'DIFF_sexseg_{}.neg.fits'.format(sci_name, label))
        c[122] = 'NTHREADS         {}              # 1 single thread\n'.format(ncpu)

    fname_config_neg_r = outdir + 'default_DIFF_r.neg.sex'
    with open(fname_config_neg_r, 'w') as f:
        f.writelines(content_neg_r)
    logger.info('Wrote SExtractor config file to {}'.format(fname_config_neg_r))
        

    fname_config_neg_s = outdir + 'default_DIFF_s.neg.sex'
    with open(fname_config_neg_s, 'w') as f:
        f.writelines(content_neg_s)
    logger.info('Wrote SExtractor config file to {}'.format(fname_config_neg_s))
    
    fname_config_neg_raw = outdir + 'default_DIFF_raw.neg.sex'
    with open(fname_config_neg_raw, 'w') as f:
        f.writelines(content_neg)
    logger.info('Wrote SExtractor config file to {}'.format(fname_config_neg_raw))
        
        
    ########################################################################
    #Run SExtractor
    #Use the REF image to detect sources, and measure the mag in the REF/SCI/DIFF images
    
    #ORIGINAL
    logger.info('Running SExtractor on original images')
        #REF
    subprocess.run(['sex', fname_ref_new, '-c', fname_config_og_r], check=True)
        #SCI
    subprocess.run(['sex', fname_sci_new, '-c', fname_config_og_s], check=True)
        #SCI_r
    subprocess.run(['sex', fname_ref_new, fname_sci_new, '-c', fname_config_og_sr], check=True)
        #REF_s
    subprocess.run(['sex', fname_sci_new, fname_ref_new, '-c', fname_config_og_rs], check=True)

    #POSITIVE
    logger.info('Running SExtractor on image')
        #REF
    subprocess.run(['sex', fname_ref_new, fname_in, '-c', fname_config_r], check=True)
        #SCI
    subprocess.run(['sex', fname_sci_new, fname_in, '-c', fname_config_s], check=True)
        #RAW
    subprocess.run(['sex', fname_in, '-c', fname_config_raw], check=True)
    
    #NEGATIVE
    logger.info('Running SExtractor on negated image')
        #REF
    subprocess.run(['sex', fname_ref_new, fname_in_neg, '-c', fname_config_neg_r], check=True)
        #SCI
    subprocess.run(['sex', fname_sci_new, fname_in_neg, '-c', fname_config_neg_s], check=True)
        #RAW
    subprocess.run(['sex', fname_in_neg, '-c', fname_config_neg_raw], check=True)
    
    return

########################################################################################################################
########################################################################################################################
########################################################################################################################
# SFFT

def run_sextractor_all_sfft(sfftdir, outdir, ref_name, sci_name, paramdir, filtername_grid, 
                            cutout_fname=None, fwhm_px_grid=None, skysub=False, gkerhw=11,
                            nsig_input=2., nsig_diff=2., minarea_input=15, minarea_diff=15,
                            cutout_subset=None, zp=23.9, ncpu=1):
    
    log_fname = outdir + 'source_identify.log'
    handler, logger = setup_logger('source_identify', log_file=log_fname, level=logging.INFO)
    
    logger.info('Running SExtractor on all images')
    logger.info('\t SFFT directory: {}'.format(sfftdir))
    logger.info('\t Output directory: {}'.format(outdir))
    logger.info('\t REF name: {}'.format(ref_name))
    logger.info('\t SCI name: {}'.format(sci_name))
    logger.info('\t Filter: {}'.format(filtername_grid))
    if cutout_fname is not None:
        logger.info('\t Cutout file: {}'.format(cutout_fname))
    logger.info('\t Number of CPUs: {}'.format(ncpu))
    
    if cutout_fname is not None:
        if os.path.basename(cutout_fname) == 'cutout_info.txt':
            ras, decs, _, _, _, _, _, cutout_names = np.loadtxt(cutout_fname, dtype=object, unpack=True, skiprows=1)
            cutout_names = cutout_names.astype(str)
            ras = ras.astype(float)
            decs = decs.astype(float)
            ncutout = len(cutout_names)

        else:
            cutout_names, ras, decs = np.loadtxt(cutout_fname, dtype=object, unpack=True, usecols=[0,1,2])
            cutout_names = np.atleast_1d(cutout_names)
            ras = np.atleast_1d(ras)
            decs = np.atleast_1d(decs)

            cutout_names = cutout_names.astype(str)
            ras = ras.astype(float)
            decs = decs.astype(float)
            ncutout = len(cutout_names)
            
            
        if cutout_subset is not None:
            cutout_subset = np.array(cutout_subset)
            cutout_names = cutout_names[cutout_subset]
            ras = ras[cutout_subset]
            decs = decs[cutout_subset]
            ncutout = len(cutout_names)
            
        logger.info('Found {} cutouts'.format(ncutout))
        logger.info('Running SExtractor on all cutouts')
        reset_logger(logger)


        for i in range(ncutout):
            indir_i = sfftdir + 'output_{}/'.format(cutout_names[i])
            outdir_i = outdir + 'sources_{}/'.format(cutout_names[i])
            
            handler, logger = setup_logger('source_identify', log_file=log_fname, level=logging.INFO)
            logger.info('Running SExtractor on cutout {}/{}: {}'.format(i+1, ncutout, cutout_names[i]))
            reset_logger(logger)
            
            os.makedirs(outdir_i, exist_ok=True)
            _, logger_i = setup_logger('source_identify', log_file=outdir_i + 'source_identify.log', level=logging.INFO)
            run_sextractor(indir_i, ref_name, sci_name, outdir_i, paramdir, filtername_grid, 
                           logger, ras[i], decs[i], 
                           fwhm_px_grid=fwhm_px_grid, skysub=skysub, gkerhw=gkerhw, difftype='sfft',
                            nsig_input=nsig_input, nsig_diff=nsig_diff, minarea_input=minarea_input, minarea_diff=minarea_diff,
                           zp=zp, ncpu=ncpu)
            reset_logger(logger_i)
          
        logger.info('Finished running SExtractor on all cutouts')
            
    else:
        indir = sfftdir #+ 'output/'            
        outdir = outdir + 'sources/'
        
        os.makedirs(outdir, exist_ok=True)
        logger.info('Running SExtractor on difference image')
        
        run_sextractor(indir, ref_name, sci_name, outdir, paramdir, filtername_grid,
                       logger, ra=None, dec=None,
                       fwhm_px_grid=fwhm_px_grid, skysub=skysub, gkerhw=gkerhw, difftype='sfft',
                       nsig_input=nsig_input, nsig_diff=nsig_diff, minarea_input=minarea_input, minarea_diff=minarea_diff,
                       zp=zp, ncpu=ncpu)
        
        logger.info('Finished running SExtractor on difference image')            
        reset_logger(logger)
    

    return


def run_fromfile_sfft(sfft_config_fname, outdir, zp=23.9, 
                nsig_input=2., nsig_diff=2.,
                minarea_input=15, minarea_diff=15):
    
    config = toml.load(sfft_config_fname)
    general = config['inputs']
    cutout = config['cutout']
    sfft = config['sfft']
    preprocess = config['preprocess']
    
    skysub = preprocess['skysub']
    
    cutout_run = cutout['run']
    cutout_fname = cutout['filename']
    cutout_subset = cutout['subset']
    if cutout_fname == '':
        cutout_fname = None

    sfftdir = general['maindir']
    paramdir = general['paramdir']
    ref_name = general['ref_name']
    sci_name = general['sci_name']
    filtername_grid = general['filtername_grid']
    ncpu = general['ncpu']
    
    gkerhw = sfft['general']['GKerHW']
    
    if cutout_run and (cutout_fname is None):
        cutout_fname = sfftdir + 'cutout_info.txt'
        
    if cutout_subset == '':
        cutout_subset = None

    os.makedirs(outdir, exist_ok=True)

    run_sextractor_all_sfft(sfftdir, outdir, ref_name, sci_name, paramdir, filtername_grid, 
                            cutout_fname=cutout_fname, skysub=skysub, gkerhw=gkerhw,
                            nsig_input=nsig_input, nsig_diff=nsig_diff, minarea_input=minarea_input, minarea_diff=minarea_diff,
                            cutout_subset=cutout_subset, zp=zp, ncpu=ncpu)
    
    return

########################################################################################################################
########################################################################################################################
########################################################################################################################
#Direct Subtraction

def run_sextractor_all_sub(subdir, outdir, ref_name, sci_name, paramdir, filtername_grid, 
                           nsig_input=2., nsig_diff=2., minarea_input=15, minarea_diff=15,
                           zp=23.9, ncpu=1):
    
    log_fname = outdir + 'source_identify.log'
    handler, logger = setup_logger('source_identify', log_file=log_fname, level=logging.INFO)
    
    logger.info('Running SExtractor on all images')
    logger.info('\t DirectSub directory: {}'.format(subdir))
    logger.info('\t Output directory: {}'.format(outdir))
    logger.info('\t REF name: {}'.format(ref_name))
    logger.info('\t SCI name: {}'.format(sci_name))
    logger.info('\t Filter: {}'.format(filtername_grid))
    

    indir = subdir #+ 'output/'            
    outdir = outdir + 'sources/'
    
    os.makedirs(outdir, exist_ok=True)
    logger.info('Running SExtractor on difference image')
    
    run_sextractor(indir, ref_name, sci_name, outdir, paramdir, filtername_grid,
                    logger, ra=None, dec=None,
                    fwhm_px_grid=None, skysub=False, gkerhw=11, difftype='sub',
                    nsig_input=nsig_input, nsig_diff=nsig_diff, minarea_input=minarea_input, minarea_diff=minarea_diff,
                    zp=zp, ncpu=ncpu)
    
    logger.info('Finished running SExtractor on difference image')            
    reset_logger(logger)

    return


########################################################################################################################
########################################################################################################################
########################################################################################################################
#Combined SFFT

def run_sextractor_sfft_combined(sfftdir, outdir, ref_name, sci_name, paramdir, filtername_grid, 
                                 skysub=False, gkerhw=11,
                                 nsig_input=2., nsig_diff=2., minarea_input=15, minarea_diff=15,
                                 zp=23.9, ncpu=1):
    
    log_fname = outdir + 'source_identify.log'
    handler, logger = setup_logger('source_identify', log_file=log_fname, level=logging.INFO)
    
    logger.info('Running SExtractor on all images')
    logger.info('\t SFFT directory: {}'.format(sfftdir))
    logger.info('\t Output directory: {}'.format(outdir))
    logger.info('\t REF name: {}'.format(ref_name))
    logger.info('\t SCI name: {}'.format(sci_name))
    logger.info('\t Filter: {}'.format(filtername_grid))
    

    indir = sfftdir          
    outdir = outdir + 'sources/'
    
    os.makedirs(outdir, exist_ok=True)
    logger.info('Running SExtractor on difference image')
    
    run_sextractor(indir, ref_name, sci_name, outdir, paramdir, filtername_grid,
                    logger, ra=None, dec=None,
                    fwhm_px_grid=None, skysub=skysub, gkerhw=gkerhw, difftype='sfft_combined',
                    nsig_input=nsig_input, nsig_diff=nsig_diff, minarea_input=minarea_input, minarea_diff=minarea_diff,
                    zp=zp, ncpu=ncpu)
    
    logger.info('Finished running SExtractor on difference image')            
    reset_logger(logger)

    return


def run_fromfile_sfft_combined(sfft_config_fname, outdir, zp=23.9, 
                              nsig_input=2., nsig_diff=2.,
                              minarea_input=15, minarea_diff=15):
    
    config = toml.load(sfft_config_fname)
    general = config['inputs']
    sfft = config['sfft']
    preprocess = config['preprocess']
    
    skysub = preprocess['skysub']

    sfftdir = general['maindir']
    paramdir = general['paramdir']
    ref_name = general['ref_name']
    sci_name = general['sci_name']
    filtername_grid = general['filtername_grid']
    ncpu = general['ncpu']
    
    gkerhw = sfft['general']['GKerHW']

    os.makedirs(outdir, exist_ok=True)

    run_sextractor_sfft_combined(sfftdir, outdir, ref_name, sci_name, paramdir, filtername_grid, 
                                 skysub=skysub, gkerhw=gkerhw,
                                 nsig_input=nsig_input, nsig_diff=nsig_diff, minarea_input=minarea_input, minarea_diff=minarea_diff,
                                 zp=zp, ncpu=ncpu)
    
    return