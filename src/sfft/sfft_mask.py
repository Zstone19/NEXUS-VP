import shutil
import os
import subprocess
import ctypes
from functools import partial

from mpire import WorkerPool
import multiprocessing as mp
from tqdm import tqdm
from numba import njit, prange
from numba_progress import ProgressBar

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats




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




def get_mag_zp(fname_data, channel):
    
    with fits.open(fname_data) as hdu:
        hdr = hdu[0].header
    
    if 'MAGZERO' in hdr.keys():
        return hdr['MAGZERO']
    
    elif 'MAG_ZP' in hdr.keys():
        return hdr['MAG_ZP']
    
    elif channel in ['SW', 'LW']:
        PIXAR_SR = hdr['PIXAR_SR']  # Nominal pixel area in steradians
        mag_zp = -2.5 * np.log10(PIXAR_SR * 1e6) + 8.9
        
        exptime = hdr['XPOSURE']
        photmjsr = hdr['PHOTMJSR']
        
        zp_new = mag_zp + 2.5 * np.log10(exptime/photmjsr)

    return zp_new





def apply_mask_total(obj_ind, mask_flat, sexseg):
    inds = np.argwhere(sexseg == obj_ind)
    flat_inds = np.array([ np.ravel_multi_index(i, sexseg.shape) for i in inds ], dtype=int)
    
    for f in flat_inds:
        mask_flat[f] = 0

    return


#@njit(fastmath=True, parallel=True)
def apply_masksat_maskin(obj_inds, mask, im_cc, nsig, bkgsd, sexseg):
        
    for i in range(len(obj_inds)):
        mask |= ( sexseg == obj_inds[i] ) & (im_cc > nsig * bkgsd)

    return mask


def apply_matchmask(obj_ind_r, obj_ind_s, mask_flat, sexseg_r, sexseg_s):
    mask = ~( (sexseg_r == obj_ind_r) & (sexseg_s == obj_ind_s) )
    inds = np.argwhere(mask)
    
    flat_inds = np.array([ np.ravel_multi_index(i, sexseg_r.shape) for i in inds ], dtype=int)
    
    for f in flat_inds:
        mask_flat[f] = 0
        
    return



@njit
def check_nsig_bkg(ind, sexseg, bkgstd_mask_ind):
    
    for mask_inds in bkgstd_mask_ind:
        if sexseg[mask_inds[0], mask_inds[1]] == ind:
            return True
            
    return False


    
def initial_global_fit(maindir, paramdir, 
              ref_name, sci_name, filtername_ref, filtername_sci, filtername_grid, 
              skysub, logger, ncpu=1):
    
    """Make the masks for SFFT to use. This mask is a binary one, with 1 for the pixels to be used and 0 for the pixels to be masked.
    This requires the cross-convolved images, noise images, and input image masks. The masks will be output as {ref_name}.mask4sfft.fits
    and {sci_name}.mask4sfft.fits in the mask/ directory.
    
    
    The mask will reject pixels that:
     - Are background pixels (from the SExtractor segmentation image)
     - Only exist in one image (from the SExtractor catalogs) AND
     - Have FLAGS == 0 in either SExtractor catalog AND
     - Have a value < 3*stddev of the background in either one of the cross-convolved images OR
     - Have a difference in MAG_AUTO > 1 between the reference and science images
     
     The mask will also reject pixels contained in saturated stars or other saturated sources.
     
    
    
    Arguments:
    ----------
    maindir : str
        The main directory for the sfft section of the NEXUS Variability Pipeline
    paramdir : str
        The directory containing the SExtractor configuration files
    ref_name : str
        The name of the reference image
    sci_name : str
        The name of the science image
    filtername_ref : str
        The name of the filter used for the REF image
    filtername_sci : str
        The name of the filter used for the SCI image
    filtername_grid : str
        The name of the filter used for the grid in both images
    skysub : bool
        Whether the input images have been sky-subtracted
    conv_ref : bool
        Whether the reference image has been cross-convolved
    conv_sci : bool
        Whether the science image has been cross-convolved
    logger : logging.Logger
        The logger to use for logging
    saturation_ref : float
        The saturation value for the reference image. Default is 0.01.
    saturation_sci : float
        The saturation value for the science image. Default is 0.01.
    bkgstd_ref_global : float
        The global background standard deviation for the reference image. Default is np.inf.
    bkgstd_sci_global : float
        The global background standard deviation for the science image. Default is np.inf.
    global_fit : bool
        Whether to perform a global fit or not. Default is False. If True, will not create a 
        mask, just run SExtractor.
    ncpu : int
        The number of CPUs to use for SExtractor. Default is 1.
    
    
    Returns:
    -------
    None

    """
    
    sex_path = 'sex'
    
    #Define directories
    indir = maindir + 'input/'
    psfdir = maindir + 'psf/'
    noisedir = maindir + 'noise/'
    outdir = maindir + 'mask/'
    
    if 'output' in maindir:
        maindir_global = '/'.join( maindir.split('/')[:-2] ) + '/'
    else:
        maindir_global = maindir
        
    maskdir_global = maindir_global + 'mask/'
        
    
    
    #Define file names
    if skysub:
        fname_ref = indir + '{}.skysub.fits'.format(ref_name)
        fname_sci = indir + '{}.skysub.fits'.format(sci_name)  
    else:
        fname_ref = indir + '{}.fits'.format(ref_name)
        fname_sci = indir + '{}.fits'.format(sci_name)

    
    fname_ref_maskin = indir + '{}.maskin.fits'.format(ref_name)
    fname_sci_maskin = indir + '{}.maskin.fits'.format(sci_name)
    
    fname_ref_noise = noisedir + '{}.noise.fits'.format(ref_name)
    fname_sci_noise = noisedir + '{}.noise.fits'.format(sci_name)
            
    ########################################################################################################################################################
    #Run SExtractor

    #Get channel and pixel scale
    filtername_ref = filtername_ref.upper()    
    if filtername_ref in SW.keys():
        channel_ref = 'SW'
        # ps_in_ref = .031    #arcsec/px
    elif filtername_ref in LW.keys():
        channel_ref = 'LW'
        # ps_in_ref = .063   #arcsec/px
    elif filtername_ref in EUCLID.keys():
        channel_ref = 'euclid'
        # ps_in_ref = .1     #arcsec/px
        
        fname_psf = psfdir + '{}.psf.fits'.format(ref_name)
        
    else:
        logger.error('Input REF filter name not part of NIRCam/Euclid')
        raise ValueError
    
    logger.info('Input REF channel: {}'.format(channel_ref))    
    
    
    filtername_sci = filtername_sci.upper()    
    if filtername_sci in SW.keys():
        channel_sci = 'SW'
        # ps_in_sci = .031    #arcsec/px
    elif filtername_sci in LW.keys():
        channel_sci = 'LW'
        # ps_in_sci = .063   #arcsec/px
    elif filtername_sci in EUCLID.keys():
        channel_sci = 'euclid'
        # ps_in_sci = .1     #arcsec/px
        
        fname_psf = psfdir + '{}.psf.fits'.format(sci_name)
        
    else:
        logger.error('Input SCI filter name not part of NIRCam/Euclid')
        raise ValueError
    

    logger.info('Input SCI channel: {}'.format(channel_sci))    

    filtername_grid = filtername_grid.upper()
    # if filtername_grid in SW.keys():
    #     ps_grid = .031
    # elif filtername_grid in LW.keys():
    #     ps_grid = .063
    # elif filtername_grid in EUCLID.keys():
    #     ps_grid = .1
    
    
    
    gauss_fname = 'gauss_4.0_7x7.conv'
    
    #Move config files
    logger.info('Copying config files to {}'.format(outdir))
    shutil.copy(paramdir + 'default.sex', outdir)
    shutil.copy(paramdir + gauss_fname, outdir)
    shutil.copy(paramdir + 'default.nnw', outdir)
    shutil.copy(paramdir + 'default_{}.param'.format(channel_ref), outdir + 'default_ref.param')
    shutil.copy(paramdir + 'default_{}.param'.format(channel_sci), outdir + 'default_sci.param')
    
    with open(paramdir + 'default.sex', 'r') as f:
        content_ref = f.readlines()
    with open(paramdir + 'default.sex', 'r') as f:
        content_sci = f.readlines()


    #Get zeropoints
    logger.info('Getting zeropoints for input images')
    mag_zp_ref = get_mag_zp(fname_ref, channel_ref)
    mag_zp_sci = get_mag_zp(fname_sci, channel_sci)
    logger.info('\t REF zeropoint: {:.3f}'.format(mag_zp_ref))
    logger.info('\t SCI zeropoint: {:.3f}'.format(mag_zp_sci))

    ########################################################################
    #Write REF SExtractor param file

    with fits.open(fname_ref) as hdul:
        ps_in_ref = hdul[0].header['CDELT1'] * 3600.    
    with fits.open(fname_sci) as hdul:
        ps_in_sci = hdul[0].header['CDELT1'] * 3600.
        
    logger.info('Input REF pixel scale: {} arcsec/px'.format(ps_in_ref))
    logger.info('Input SCI pixel scale: {} arcsec/px'.format(ps_in_sci))

    
    if filtername_grid == filtername_ref:
        ps_grid = ps_in_ref
    elif filtername_grid == filtername_sci:
        ps_grid = ps_in_sci

    
    if channel_ref in ['SW', 'LW']:
        phot_aper = NIRCam_filter_FWHM_new[channel_ref][filtername_ref]*5*ps_in_ref/ps_grid
        seeing_fwhm = NIRCam_filter_FWHM_new[channel_ref][filtername_ref] * ps_in_ref
    elif channel_ref == 'euclid':
        phot_aper = fwhm_px_ref * 5 * ps_in_ref / ps_grid
        seeing_fwhm = fwhm_px_ref * ps_in_ref
    
    
    logger.info('Writing SExtractor config file for REF')
    
    content_ref[6]   = 'CATALOG_NAME     {}              # name of the output catalog\n'.format(outdir + ref_name + '.cat')
    content_ref[7]   = "CATALOG_TYPE     FITS_LDAC       # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,\n"
    content_ref[9]   = "PARAMETERS_NAME  {}              # name of the file containing catalog contents\n".format(outdir + 'default_ref.param')
    content_ref[14]  = "DETECT_MINAREA   5              # min. # of pixels above threshold\n"
    content_ref[18]  = "DETECT_THRESH    1.0             # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n"
    content_ref[19]  = "ANALYSIS_THRESH  1.0             # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n"
    content_ref[22]  = "FILTER_NAME      {}              # name of the file containing the filter\n".format(outdir + gauss_fname)
    content_ref[25]  = "DEBLEND_NTHRESH  64             # Number of deblending sub-thresholds\n"
    content_ref[26]  = "DEBLEND_MINCONT  1e-3           # Minimum contrast parameter for deblending\n" 
    
    content_ref[38]  = 'WEIGHT_IMAGE     {}              # weight-map filename\n'.format(fname_ref_noise)
    content_ref[44]  = 'FLAG_IMAGE       {}              # filename for an input FLAG-image\n'.format(fname_ref_maskin)

    content_ref[55]  = "PHOT_APERTURES   {}              # MAG_APER aperture diameter(s) in pixels\n".format(phot_aper)  # 5 times FWHM
    # content_ref[63]  = "SATUR_LEVEL      {}              # level (in ADUs) at which arises saturation\n".format(saturation_ref)
    content_ref[66]  = 'MAG_ZEROPOINT    {}              # magnitude zero-point\n'.format(mag_zp_ref)
    
    content_ref[74]  = "SEEING_FWHM      {}              # stellar FWHM in arcsec\n".format(seeing_fwhm)    
    content_ref[75]  = "STARNNW_NAME     {}              # Neural-Network_Weight table filename\n".format(outdir + 'default.nnw')
    content_ref[81]  = "BACK_SIZE        64              # Background mesh: <size> or <width>,<height> default 64\n"

    content_ref[95]  = 'CHECKIMAGE_NAME  {}              # Filename for the check-image\n'.format(outdir + '{}_sexseg.fits'.format(ref_name))
    content_ref[122] = 'NTHREADS         {}              # 1 single thread\n'.format(ncpu)
    # content_ref[131] = "PSF_NAME         {}              # File containing the PSF model\n".format(fname_psf_ref)

    fname_ref_config = outdir + 'default_{}.sex'.format(ref_name)
    with open(fname_ref_config, 'w') as f:
        f.writelines(content_ref)
        
    #Run SExtractor for REF
    if os.path.exists(outdir + '{}_sexseg.fits'.format(ref_name)) and os.path.exists(outdir + '{}.cat'.format(ref_name)):
        logger.info('Found SExtractor output for REF, skipping')
    else:    
        logger.info('Running SExtractor for REF')
        with open(maindir + 'sfft.log', 'a') as flog:
            subprocess.run([sex_path, fname_ref, '-c', fname_ref_config], check=True, text=True, stdout=flog)
        logger.info('Finished running SExtractor for REF')
    
    ########################################################################
    #Write REF SExtractor param file
    if channel_sci in ['SW', 'LW']:
        phot_aper = NIRCam_filter_FWHM_new[channel_sci][filtername_sci]*5*ps_in_sci/ps_grid
        seeing_fwhm = NIRCam_filter_FWHM_new[channel_sci][filtername_sci] * ps_in_sci
    elif channel_sci == 'euclid':
        phot_aper = fwhm_px_sci * 5 * ps_in_sci / ps_grid
        seeing_fwhm = fwhm_px_sci * ps_in_sci
    
    
    logger.info('Writing SExtractor config file for SCI')
    
    content_sci[6]   = 'CATALOG_NAME     {}              # name of the output catalog\n'.format(outdir + sci_name + '.cat')
    content_sci[7]   = "CATALOG_TYPE     FITS_LDAC       # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,\n"
    content_sci[9]   = "PARAMETERS_NAME  {}              # name of the file containing catalog contents\n".format(outdir + 'default_sci.param')
    content_sci[14]  = "DETECT_MINAREA   5              # min. # of pixels above threshold\n"
    content_sci[18]  = "DETECT_THRESH    1.0             # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n"
    content_sci[19]  = "ANALYSIS_THRESH  1.0             # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n"
    content_sci[22]  = "FILTER_NAME      {}              # name of the file containing the filter\n".format(outdir + gauss_fname)
    content_sci[25]  = "DEBLEND_NTHRESH  64             # Number of deblending sub-thresholds\n"
    content_sci[26]  = "DEBLEND_MINCONT  1e-3           # Minimum contrast parameter for deblending\n" 
    
    content_sci[38]  = 'WEIGHT_IMAGE     {}              # weight-map filename\n'.format(fname_sci_noise)
    content_sci[44]  = 'FLAG_IMAGE       {}              # filename for an input FLAG-image\n'.format(fname_sci_maskin)

    content_sci[55]  = "PHOT_APERTURES   {}              # MAG_APER aperture diameter(s) in pixels\n".format(phot_aper)  # 5 times FWHM
    # content_sci[63]  = "SATUR_LEVEL      {}              # level (in ADUs) at which arises saturation\n".format(saturation_sci)
    content_sci[66]  = 'MAG_ZEROPOINT    {}              # magnitude zero-point\n'.format(mag_zp_sci)
    
    content_sci[74]  = "SEEING_FWHM      {}              # stellar FWHM in arcsec\n".format(seeing_fwhm)
    content_sci[75]  = "STARNNW_NAME     {}              # Neural-Network_Weight table filename\n".format(outdir + 'default.nnw')
    content_ref[81]  = "BACK_SIZE        64              # Background mesh: <size> or <width>,<height> default 64\n"

    content_sci[95]  = 'CHECKIMAGE_NAME  {}              # Filename for the check-image\n'.format(outdir + '{}_sexseg.fits'.format(sci_name))
    content_sci[122] = 'NTHREADS         {}               # 1 single thread\n'.format(ncpu)
    # content_sci[131] = "PSF_NAME         {}              # File containing the PSF model\n".format(fname_psf_sci)
    
    fname_sci_config = outdir + 'default_{}.sex'.format(sci_name)
    with open(fname_sci_config, 'w') as f:
        f.writelines(content_sci)

    #Run SExtractor for REF
    if os.path.exists(outdir + '{}_sexseg.fits'.format(sci_name)) and os.path.exists(outdir + '{}.cat'.format(sci_name)):
        logger.info('Found SExtractor output for SCI, skipping')
    else:    
        logger.info('Running SExtractor for SCI')
        with open(maindir + 'sfft.log', 'a') as f:
            subprocess.run([sex_path, fname_sci, '-c', fname_sci_config], check=True, stdout=f)
        logger.info('Finished running SExtractor for SCI')

    return


def make_mask(maindir, paramdir, ref_name, sci_name, filtername_ref, filtername_sci, filtername_grid, 
              skysub, conv_ref, conv_sci, logger, saturation_ref=.01, saturation_sci=.01, 
              bkgstd_ref_global=np.inf, bkgstd_sci_global=np.inf,
              ra=None, dec=None, npx_side=None, ncpu=1):
    
    """Make the masks for SFFT to use. This mask is a binary one, with 1 for the pixels to be used and 0 for the pixels to be masked.
    This requires the cross-convolved images, noise images, and input image masks. The masks will be output as {ref_name}.mask4sfft.fits
    and {sci_name}.mask4sfft.fits in the mask/ directory.
    
    
    The mask will reject pixels that:
     - Are background pixels (from the SExtractor segmentation image)
     - Only exist in one image (from the SExtractor catalogs) AND
     - Have FLAGS == 0 in either SExtractor catalog AND
     - Have a value < 3*stddev of the background in either one of the cross-convolved images OR
     - Have a difference in MAG_AUTO > 1 between the reference and science images
     
     The mask will also reject pixels contained in saturated stars or other saturated sources.
     
    
    
    Arguments:
    ----------
    maindir : str
        The main directory for the sfft section of the NEXUS Variability Pipeline
    paramdir : str
        The directory containing the SExtractor configuration files
    ref_name : str
        The name of the reference image
    sci_name : str
        The name of the science image
    filtername_ref : str
        The name of the filter used for the REF image
    filtername_sci : str
        The name of the filter used for the SCI image
    filtername_grid : str
        The name of the filter used for the grid in both images
    skysub : bool
        Whether the input images have been sky-subtracted
    conv_ref : bool
        Whether the reference image has been cross-convolved
    conv_sci : bool
        Whether the science image has been cross-convolved
    logger : logging.Logger
        The logger to use for logging
    saturation_ref : float
        The saturation value for the reference image. Default is 0.01.
    saturation_sci : float
        The saturation value for the science image. Default is 0.01.
    bkgstd_ref_global : float
        The global background standard deviation for the reference image. Default is np.inf.
    bkgstd_sci_global : float
        The global background standard deviation for the science image. Default is np.inf.
    global_fit : bool
        Whether to perform a global fit or not. Default is False. If True, will not create a 
        mask, just run SExtractor.
    ncpu : int
        The number of CPUs to use for SExtractor. Default is 1.
    
    
    Returns:
    -------
    None

    """
    
    nsig = 3.
    # sex_path = '/home/stone28/software/sextractor/install/bin/sex'
    sex_path = 'sex'
    
    
    #Define directories
    indir = maindir + 'input/'
    psfdir = maindir + 'psf/'
    noisedir = maindir + 'noise/'
    outdir = maindir + 'mask/'
    
    if 'output' in maindir:
        maindir_global = '/'.join( maindir.split('/')[:-2] ) + '/'
    else:
        maindir_global = maindir
        
    maskdir_global = maindir_global + 'mask/'
        
    
    
    #Define file names
    if skysub:
        fname_ref = indir + '{}.skysub.fits'.format(ref_name)
        fname_sci = indir + '{}.skysub.fits'.format(sci_name)  
    else:
        fname_ref = indir + '{}.fits'.format(ref_name)
        fname_sci = indir + '{}.fits'.format(sci_name)

    
    fname_ref_maskin = indir + '{}.maskin.fits'.format(ref_name)
    fname_sci_maskin = indir + '{}.maskin.fits'.format(sci_name)
    
    if conv_ref:
        fname_ref_cc = maindir + 'output/{}.crossconvd.fits'.format(ref_name)
    else:
        fname_ref_cc = fname_ref
        
    if conv_sci:
        fname_sci_cc = maindir + 'output/{}.crossconvd.fits'.format(sci_name)
    else:
        fname_sci_cc = fname_sci
    
    fname_ref_noise = noisedir + '{}.noise.fits'.format(ref_name)
    fname_sci_noise = noisedir + '{}.noise.fits'.format(sci_name)
    
    fname_out_ref = outdir + '{}.mask4sfft.fits'.format(ref_name)
    fname_out_sci = outdir + '{}.mask4sfft.fits'.format(sci_name)
    
    fname_psf_ref = psfdir + '{}.psf.fits'.format(ref_name)
    fname_psf_sci = psfdir + '{}.psf.fits'.format(sci_name)
    
    with fits.open(fname_ref) as hdu:
        im1 = hdu[0].data
    with fits.open(fname_sci) as hdu:
        im2 = hdu[0].data
        
    assert im1.shape == im2.shape, 'Input images must have the same shape'
    
    if os.path.exists(fname_out_ref) and os.path.exists(fname_out_sci):
        logger.info('Found masks, skipping')
        return

    #Make sure images have been cross-convolved, and input masks are present
    logger.info('Checking for input files')
    if not os.path.exists( fname_ref_cc ):
        logger.error('Cross-convolved REF image does not exist')
        raise FileNotFoundError('Cross-convolved REF image does not exist')
        
    if not os.path.exists( fname_sci_cc ):
        logger.error('Cross-convolved SCI image does not exist')
        raise FileNotFoundError('Cross-convolved SCI image does not exist')
    
    if not os.path.exists( fname_ref_maskin ):
        logger.error('Input REF mask file does not exist')
        raise FileNotFoundError('Input REF mask file does not exist')
    
    if not os.path.exists( fname_sci_maskin ):
        logger.error('Input SCI mask file does not exist')
        raise FileNotFoundError('Input SCI mask file does not exist')
    
    ########################################################################################################################################################
    #Run SExtractor

    #Get channel and pixel scale
    filtername_ref = filtername_ref.upper()    
    if filtername_ref in SW.keys():
        channel_ref = 'SW'
        # ps_in_ref = .031    #arcsec/px
    elif filtername_ref in LW.keys():
        channel_ref = 'LW'
        # ps_in_ref = .063   #arcsec/px
    elif filtername_ref in EUCLID.keys():
        channel_ref = 'euclid'
        # ps_in_ref = .1     #arcsec/px
        
        fname_psf = psfdir + '{}.psf.fits'.format(ref_name)
        with fits.open(fname_psf) as hdu:
            fwhm_px_ref = hdu[0].header['FWHM']
        
    else:
        logger.error('Input REF filter name not part of NIRCam/Euclid')
        raise ValueError
    
    with fits.open(fname_ref) as hdul:
        ps_in_ref = hdu[0].header['CDELT1'] * 3600.

    
    logger.info('Input REF channel: {}'.format(channel_ref))
    logger.info('Input REF pixel scale: {} arcsec/px'.format(ps_in_ref))
    
    
    
    filtername_sci = filtername_sci.upper()    
    if filtername_sci in SW.keys():
        channel_sci = 'SW'
        # ps_in_sci = .031    #arcsec/px
    elif filtername_sci in LW.keys():
        channel_sci = 'LW'
        # ps_in_sci = .063   #arcsec/px
    elif filtername_sci in EUCLID.keys():
        channel_sci = 'euclid'
        # ps_in_sci = .1     #arcsec/px
        
        fname_psf = psfdir + '{}.psf.fits'.format(sci_name)
        with fits.open(fname_psf) as hdu:
            fwhm_px_sci = hdu[0].header['FWHM']
        
    else:
        logger.error('Input SCI filter name not part of NIRCam/Euclid')
        raise ValueError
    
    with fits.open(fname_sci) as hdul:
        ps_in_sci = hdu[0].header['CDELT1'] * 3600.
    
    logger.info('Input SCI channel: {}'.format(channel_sci))
    logger.info('Input SCI pixel scale: {} arcsec/px'.format(ps_in_sci))
    

    filtername_grid = filtername_grid.upper()
    # if filtername_grid in SW.keys():
    #     ps_grid = .031
    # elif filtername_grid in LW.keys():
    #     ps_grid = .063
    # elif filtername_grid in EUCLID.keys():
    #     ps_grid = .1
    
    if filtername_grid == filtername_ref:
        ps_grid = ps_in_ref
    elif filtername_grid == filtername_sci:
        ps_grid = ps_in_sci
    
    
    
    gauss_fname = 'gauss_4.0_7x7.conv'
    
    #Move config files
    logger.info('Copying config files to {}'.format(outdir))
    shutil.copy(paramdir + 'default.sex', outdir)
    shutil.copy(paramdir + gauss_fname, outdir)
    shutil.copy(paramdir + 'default.nnw', outdir)
    shutil.copy(paramdir + 'default_{}.param'.format(channel_ref), outdir + 'default_ref.param')
    shutil.copy(paramdir + 'default_{}.param'.format(channel_sci), outdir + 'default_sci.param')
    
    with open(paramdir + 'default.sex', 'r') as f:
        content_ref = f.readlines()
    with open(paramdir + 'default.sex', 'r') as f:
        content_sci = f.readlines()


    #Get zeropoints
    logger.info('Getting zeropoints for input images')
    mag_zp_ref = get_mag_zp(fname_ref, channel_ref)
    mag_zp_sci = get_mag_zp(fname_sci, channel_sci)
    
    logger.info('\t REF zeropoint: {:.3f}'.format(mag_zp_ref))
    logger.info('\t SCI zeropoint: {:.3f}'.format(mag_zp_sci))

    ########################################################################
    #Write REF SExtractor param file
    if channel_ref in ['SW', 'LW']:
        phot_aper = NIRCam_filter_FWHM_new[channel_ref][filtername_ref]*5*ps_in_ref/ps_grid
        seeing_fwhm = NIRCam_filter_FWHM_new[channel_ref][filtername_ref] * ps_in_ref
    elif channel_ref == 'euclid':
        phot_aper = fwhm_px_ref * 5 * ps_in_ref / ps_grid
        seeing_fwhm = fwhm_px_ref * ps_in_ref
    
    
    logger.info('Writing SExtractor config file for REF')
    
    content_ref[6]   = 'CATALOG_NAME     {}              # name of the output catalog\n'.format(outdir + ref_name + '.cat')
    content_ref[7]   = "CATALOG_TYPE     FITS_LDAC       # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,\n"
    content_ref[9]   = "PARAMETERS_NAME  {}              # name of the file containing catalog contents\n".format(outdir + 'default_ref.param')
    content_ref[14]  = "DETECT_MINAREA   5              # min. # of pixels above threshold\n"
    content_ref[18]  = "DETECT_THRESH    1.0             # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n"
    content_ref[19]  = "ANALYSIS_THRESH  1.0             # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n"
    content_ref[22]  = "FILTER_NAME      {}              # name of the file containing the filter\n".format(outdir + gauss_fname)
    content_ref[25]  = "DEBLEND_NTHRESH  64             # Number of deblending sub-thresholds\n"
    content_ref[26]  = "DEBLEND_MINCONT  1e-3           # Minimum contrast parameter for deblending\n" 
    
    content_ref[38]  = 'WEIGHT_IMAGE     {}              # weight-map filename\n'.format(fname_ref_noise)
    content_ref[44]  = 'FLAG_IMAGE       {}              # filename for an input FLAG-image\n'.format(fname_ref_maskin)

    content_ref[55]  = "PHOT_APERTURES   {}              # MAG_APER aperture diameter(s) in pixels\n".format(phot_aper)  # 5 times FWHM
    # content_ref[63]  = "SATUR_LEVEL      {}              # level (in ADUs) at which arises saturation\n".format(saturation_ref)
    content_ref[66]  = 'MAG_ZEROPOINT    {}              # magnitude zero-point\n'.format(mag_zp_ref)
    
    content_ref[74]  = "SEEING_FWHM      {}              # stellar FWHM in arcsec\n".format(seeing_fwhm)    
    content_ref[75]  = "STARNNW_NAME     {}              # Neural-Network_Weight table filename\n".format(outdir + 'default.nnw')
    content_ref[81]  = "BACK_SIZE        64              # Background mesh: <size> or <width>,<height> default 64\n"

    content_ref[95]  = 'CHECKIMAGE_NAME  {}              # Filename for the check-image\n'.format(outdir + '{}_sexseg.fits'.format(ref_name))
    content_ref[122] = 'NTHREADS         {}              # 1 single thread\n'.format(ncpu)
    # content_ref[131] = "PSF_NAME         {}              # File containing the PSF model\n".format(fname_psf_ref)

    fname_ref_config = outdir + 'default_{}.sex'.format(ref_name)
    with open(fname_ref_config, 'w') as f:
        f.writelines(content_ref)
        
    #Run SExtractor for REF
    if os.path.exists(outdir + '{}_sexseg.fits'.format(ref_name)) and os.path.exists(outdir + '{}.cat'.format(ref_name)):
        logger.info('Found SExtractor output for REF, skipping')
    else:    
        logger.info('Running SExtractor for REF')
        with open(maindir + 'sfft.log', 'a') as flog:
            subprocess.run([sex_path, fname_ref, '-c', fname_ref_config], check=True, text=True, stdout=flog)
        logger.info('Finished running SExtractor for REF')
    
    ########################################################################
    #Write REF SExtractor param file
    if channel_sci in ['SW', 'LW']:
        phot_aper = NIRCam_filter_FWHM_new[channel_sci][filtername_sci]*5*ps_in_sci/ps_grid
        seeing_fwhm = NIRCam_filter_FWHM_new[channel_sci][filtername_sci] * ps_in_sci
    elif channel_sci == 'euclid':
        phot_aper = fwhm_px_sci * 5 * ps_in_sci / ps_grid
        seeing_fwhm = fwhm_px_sci * ps_in_sci
    
    
    logger.info('Writing SExtractor config file for SCI')
    
    content_sci[6]   = 'CATALOG_NAME     {}              # name of the output catalog\n'.format(outdir + sci_name + '.cat')
    content_sci[7]   = "CATALOG_TYPE     FITS_LDAC       # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,\n"
    content_sci[9]   = "PARAMETERS_NAME  {}              # name of the file containing catalog contents\n".format(outdir + 'default_sci.param')
    content_sci[14]  = "DETECT_MINAREA   5              # min. # of pixels above threshold\n"
    content_sci[18]  = "DETECT_THRESH    1.0             # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n"
    content_sci[19]  = "ANALYSIS_THRESH  1.0             # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n"
    content_sci[22]  = "FILTER_NAME      {}              # name of the file containing the filter\n".format(outdir + gauss_fname)
    content_sci[25]  = "DEBLEND_NTHRESH  64             # Number of deblending sub-thresholds\n"
    content_sci[26]  = "DEBLEND_MINCONT  1e-3           # Minimum contrast parameter for deblending\n" 
    
    content_sci[38]  = 'WEIGHT_IMAGE     {}              # weight-map filename\n'.format(fname_sci_noise)
    content_sci[44]  = 'FLAG_IMAGE       {}              # filename for an input FLAG-image\n'.format(fname_sci_maskin)

    content_sci[55]  = "PHOT_APERTURES   {}              # MAG_APER aperture diameter(s) in pixels\n".format(phot_aper)  # 5 times FWHM
    # content_sci[63]  = "SATUR_LEVEL      {}              # level (in ADUs) at which arises saturation\n".format(saturation_sci)
    content_sci[66]  = 'MAG_ZEROPOINT    {}              # magnitude zero-point\n'.format(mag_zp_sci)
    
    content_sci[74]  = "SEEING_FWHM      {}              # stellar FWHM in arcsec\n".format(seeing_fwhm)
    content_sci[75]  = "STARNNW_NAME     {}              # Neural-Network_Weight table filename\n".format(outdir + 'default.nnw')
    content_ref[81]  = "BACK_SIZE        64              # Background mesh: <size> or <width>,<height> default 64\n"

    content_sci[95]  = 'CHECKIMAGE_NAME  {}              # Filename for the check-image\n'.format(outdir + '{}_sexseg.fits'.format(sci_name))
    content_sci[122] = 'NTHREADS         {}               # 1 single thread\n'.format(ncpu)
    # content_sci[131] = "PSF_NAME         {}              # File containing the PSF model\n".format(fname_psf_sci)
    
    fname_sci_config = outdir + 'default_{}.sex'.format(sci_name)
    with open(fname_sci_config, 'w') as f:
        f.writelines(content_sci)
                
    #Run SExtractor for REF
    if os.path.exists(outdir + '{}_sexseg.fits'.format(sci_name)) and os.path.exists(outdir + '{}.cat'.format(sci_name)):
        logger.info('Found SExtractor output for SCI, skipping')
    else:    
        logger.info('Running SExtractor for SCI')
        with open(maindir + 'sfft.log', 'a') as f:
            subprocess.run([sex_path, fname_sci, '-c', fname_sci_config], check=True, stdout=f)
        logger.info('Finished running SExtractor for SCI')
    
    ########################################################################################################################################################
    #Make mask
    
    logger.info('Reading global SExtrcator output')
    with fits.open(maskdir_global + '{}.cat'.format(ref_name)) as hdul:
        catdat_ref_global = Table(hdul[2].data)
    with fits.open(maskdir_global + '{}.cat'.format(sci_name)) as hdul:
        catdat_sci_global = Table(hdul[2].data)
        
    with fits.open(maskdir_global + '{}_sexseg.fits'.format(ref_name)) as hdul:
        seg_ref_global = hdul[0].data
    with fits.open(maskdir_global + '{}_sexseg.fits'.format(sci_name)) as hdul:
        seg_sci_global = hdul[0].data
        
    with fits.open(maindir_global + 'input/{}.fits'.format(ref_name)) as hdul:
        hdr_ref_global = hdul[0].header
        wcs_ref_global = WCS(hdul[0].header)
    with fits.open(maindir_global + 'input/{}.fits'.format(sci_name)) as hdul:
        hdr_sci_global = hdul[0].header
        wcs_sci_global = WCS(hdul[0].header)
    
    logger.info('\t Number of objects in global REF catalog: {}'.format(len(catdat_ref_global)))
    logger.info('\t Number of objects in global SCI catalog: {}'.format(len(catdat_sci_global)))
    
    
    logger.info('Reading SExtractor output')
    
    with fits.open(outdir + '{}.cat'.format(ref_name)) as hdul:
        catdat_ref = Table(hdul[2].data)
        
    with fits.open(outdir + '{}.cat'.format(sci_name)) as hdul:
        catdat_sci = Table(hdul[2].data)
        
    with fits.open(outdir + '{}_sexseg.fits'.format(ref_name)) as hdul:
        seg_ref = hdul[0].data
        
    with fits.open(outdir + '{}_sexseg.fits'.format(sci_name)) as hdul:
        seg_sci = hdul[0].data
        
    with fits.open(fname_ref_cc) as hdul:
        im_ref_cc = hdul[0].data
        
    with fits.open(fname_sci_cc) as hdul:
        im_sci_cc = hdul[0].data
        
    with fits.open(fname_ref_maskin) as hdul:
        im_mref = hdul[0].data.astype(bool)
    with fits.open(fname_sci_maskin) as hdul:
        im_msci = hdul[0].data.astype(bool)
        
    im_mtot = im_mref | im_msci
        
    logger.info('\t Number of objects in REF catalog: {}'.format(len(catdat_ref)))
    logger.info('\t Number of objects in SCI catalog: {}'.format(len(catdat_sci)))
    
    
    #Crop global segmentation maps to the size of the input images
    logger.info('Cropping global segmentation maps to the size of the input images')
    if (ra is not None) or (dec is not None) or (npx_side is not None):
        coord_cent = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
        
        seg_ref_global = Cutout2D(seg_ref_global, position=coord_cent, size=(npx_side[0]*u.pixel, npx_side[1]*u.pixel), wcs=wcs_sci_global).data.astype(int)
        seg_sci_global = Cutout2D(seg_sci_global, position=coord_cent, size=(npx_side[0]*u.pixel, npx_side[1]*u.pixel), wcs=wcs_ref_global).data.astype(int)
    
    
    #Only keep global objects in the cutout segmentation maps
    unique_ind_ref_global = np.unique(seg_ref_global[seg_ref_global > 0])
    unique_ind_sci_global = np.unique(seg_sci_global[seg_sci_global > 0])
    
    objmask_ref_global = np.array([ x in unique_ind_ref_global for x in catdat_ref_global['NUMBER'].data ])
    objmask_sci_global = np.array([ x in unique_ind_sci_global for x in catdat_sci_global['NUMBER'].data ])

    catdat_ref_global = catdat_ref_global[objmask_ref_global].copy()
    catdat_sci_global = catdat_sci_global[objmask_sci_global].copy()
    
    catdat_ref_global.write(outdir + '{}.global.cat'.format(ref_name), overwrite=True, format='fits')
    catdat_sci_global.write(outdir + '{}.global.cat'.format(sci_name), overwrite=True, format='fits')
    fits.writeto(outdir + '{}_sexseg.global.fits'.format(ref_name), seg_ref_global, hdr_ref_global, overwrite=True)
    fits.writeto(outdir + '{}_sexseg.global.fits'.format(sci_name), seg_sci_global, hdr_sci_global, overwrite=True)



    #Match gobal catalogs to local catalogs
    logger.info('Matching global catalogs to local catalogs')
    
    coords_global_ref = SkyCoord(ra=catdat_ref_global['ALPHA_J2000'], dec=catdat_ref_global['DELTA_J2000'], unit=(u.deg, u.deg))
    coords_global_sci = SkyCoord(ra=catdat_sci_global['ALPHA_J2000'], dec=catdat_sci_global['DELTA_J2000'], unit=(u.deg, u.deg))
    coords_ref = SkyCoord(ra=catdat_ref['ALPHA_J2000'], dec=catdat_ref['DELTA_J2000'], unit=(u.deg, u.deg))
    coords_sci = SkyCoord(ra=catdat_sci['ALPHA_J2000'], dec=catdat_sci['DELTA_J2000'], unit=(u.deg, u.deg))
    
    if len(coords_global_ref) == 0 or len(coords_ref) == 0:
        catdat_matched_global_ref = Table(names=catdat_ref_global.colnames)
        catdat_matched_ref = Table(names=catdat_ref.colnames)
    else:        
        idx, d2d, _ = coords_ref.match_to_catalog_sky(coords_global_ref)
        mask = d2d.arcsec < 0.1
        catdat_matched_global_ref = catdat_ref_global[idx[mask]].copy()
        catdat_matched_ref = catdat_ref[mask].copy()
    
    
    if len(coords_global_sci) == 0 or len(coords_sci) == 0:
        catdat_matched_global_sci = Table(names=catdat_sci_global.colnames)
        catdat_matched_sci = Table(names=catdat_sci.colnames)
    else:
        idx, d2d, _ = coords_sci.match_to_catalog_sky(coords_global_sci)
        mask = d2d.arcsec < 0.1
        catdat_matched_global_sci = catdat_sci_global[idx[mask]].copy()
        catdat_matched_sci = catdat_sci[mask].copy()
    
    logger.info('\t Number of objects in matched REF catalog: {}'.format(len(catdat_matched_ref)))
    logger.info('\t Number of objects in matched SCI catalog: {}'.format(len(catdat_matched_sci)))
    
    
    
        
        
    #Get background stddev in cross-convolved images 
    logger.info('Getting background stddev in cross-convolved images')       
    # bkgstd_ref = np.nanstd(im_ref_cc[(seg_ref == 0) & (~im_mtot)])
    # bkgstd_sci = np.nanstd(im_sci_cc[(seg_sci == 0) & (~im_mtot)])
    
    _, _, bkgstd_ref = sigma_clipped_stats(im_ref_cc, sigma=3.0, maxiters=None, mask=(im_mtot | (seg_ref > 0)) )
    _, _, bkgstd_sci = sigma_clipped_stats(im_sci_cc, sigma=3.0, maxiters=None, mask=(im_mtot | (seg_sci > 0)) )
    logger.info('\t Local REF background stddev: {:.2e}'.format(bkgstd_ref))
    logger.info('\t Local SCI background stddev: {:.2e}'.format(bkgstd_sci))
    
    
    if bkgstd_ref_global < bkgstd_ref:
        logger.info('\t Using global REF background stddev')
        bkgstd_ref = bkgstd_ref_global
    else:
        logger.info('\t Using local REF background stddev')
        
    if bkgstd_sci_global < bkgstd_sci:
        logger.info('\t Using global SCI background stddev')
        bkgstd_sci = bkgstd_sci_global
    else:
        logger.info('\t Using local SCI background stddev')    
    
    
    #Get rid of saturated stars
    logger.info('Removing saturated stars from REF and SCI catalogs')
    
    #Use global catalogs for bright sources, local catalogs for dimmer ones    
        #Bright (f_aper > 30)
    bad_mask_r_global = (   (catdat_ref_global['FLUX_APER'] > 45) & (catdat_ref_global['FLUX_APER'] < 150 ) & (catdat_ref_global['CLASS_STAR'] > .9)   )  |  ( (catdat_ref_global['FLUX_APER'] > 150) & (catdat_ref_global['CLASS_STAR'] > .7) )
    bad_mask_s_global = (   (catdat_sci_global['FLUX_APER'] > 45) & (catdat_sci_global['FLUX_APER'] < 150 ) & (catdat_sci_global['CLASS_STAR'] > .9)   )  |  ( (catdat_sci_global['FLUX_APER'] > 150) & (catdat_sci_global['CLASS_STAR'] > .7) )
        #Dim (f_aper < 30)
    bad_mask_r = ( (catdat_ref['CLASS_STAR'] > .98) & (catdat_ref['FLUX_APER'] > 1) & (catdat_ref['ELONGATION'] < 1.3) )
    bad_mask_s = ( (catdat_sci['CLASS_STAR'] > .98) & (catdat_sci['FLUX_APER'] > 1) & (catdat_sci['ELONGATION'] < 1.3) )

    #If there are bright sources in the local catalogs, but not in the global ones, put them in the local saturation catalogs
    bad_mask_r_local_bright = (catdat_ref['FLUX_APER'] > 45) | (catdat_ref['MAG_AUTO'] < 17)
    bad_mask_s_local_bright = (catdat_sci['FLUX_APER'] > 45) | (catdat_sci['MAG_AUTO'] < 17)
    catdat_ref_local_bright = catdat_ref[bad_mask_r_local_bright].copy()
    catdat_sci_local_bright = catdat_sci[bad_mask_s_local_bright].copy()
    
    logger.info('\t Number of local bright sources in REF catalog: {}'.format(len(catdat_ref_local_bright)))
    logger.info('\t Number of local bright sources in SCI catalog: {}'.format(len(catdat_sci_local_bright)))
    
    
    #If pixels overlap between a "local only" bright source and a global source, the local one is most likely a large (unsaturated) source near the edge of the image
    #Remove these from the local saturation catalogs
    fake_bright_mask_r = np.zeros(len(catdat_ref_local_bright), dtype=bool)
    fake_bright_mask_s = np.zeros(len(catdat_sci_local_bright), dtype=bool)
    if (len(catdat_ref_local_bright) > 0) and (len(catdat_ref_global) > 0):

        for i in tqdm(  range(len(catdat_ref_local_bright))  ):
            coords_ref_local_bright = SkyCoord([catdat_ref_local_bright['ALPHA_J2000'][i]], [catdat_ref_local_bright['DELTA_J2000'][i]], unit=(u.deg, u.deg))
            idx, d2d, _ = coords_ref_local_bright.match_to_catalog_sky(coords_global_ref)

            ind_local = np.argmin(d2d.arcsec)
            src_local = catdat_ref_local_bright[ind_local]
            src_global = catdat_ref_global[idx[ind_local]]      
            
            n_local = (seg_ref == src_local['NUMBER']).sum()
            n_global = (seg_ref_global == src_global['NUMBER']).sum()
            npx = np.max([n_local, n_global])
            
            overlap_px = ((seg_ref == src_local['NUMBER']) & (seg_ref_global == src_global['NUMBER'])).sum()
            
            
            if overlap_px > 0.25 * npx:        
                fake_bright_mask_r[i] = True
                
                
    if ( len(catdat_sci_local_bright) > 0 ) and (len(catdat_sci_global) > 0):
        for i in tqdm(  range(len(catdat_sci_local_bright))  ):
            coords_sci_local_bright = SkyCoord([catdat_sci_local_bright['ALPHA_J2000'][i]], [catdat_sci_local_bright['DELTA_J2000'][i]], unit=(u.deg, u.deg))
            idx, d2d, _ = coords_sci_local_bright.match_to_catalog_sky(coords_global_sci)

            ind_local = np.argmin(d2d.arcsec)
            src_local = catdat_sci_local_bright[ind_local]
            src_global = catdat_sci_global[idx[ind_local]]         
            
            n_local = (seg_sci == src_local['NUMBER']).sum()
            n_global = (seg_sci_global == src_global['NUMBER']).sum()
            npx = np.max([n_local, n_global])
            
            overlap_px = ((seg_sci == src_local['NUMBER']) & (seg_sci_global == src_global['NUMBER'])).sum()
            
            
            if overlap_px > 0.25 * npx:        
                fake_bright_mask_s[i] = True

    logger.info('\t Number of fake bright sources in REF catalog: {}'.format(fake_bright_mask_r.sum()))
    logger.info('\t Number of fake bright sources in SCI catalog: {}'.format(fake_bright_mask_s.sum()))

    bad_mask_r_local_bright[bad_mask_r_local_bright][fake_bright_mask_r] = False
    bad_mask_s_local_bright[bad_mask_s_local_bright][fake_bright_mask_s] = False
    catdat_ref_local_bright = catdat_ref_local_bright[~fake_bright_mask_r].copy()
    catdat_sci_local_bright = catdat_sci_local_bright[~fake_bright_mask_s].copy()



    #Initial saturation catalogs
    catdat_ref_sat = catdat_ref[bad_mask_r].copy()
    catdat_sci_sat = catdat_sci[bad_mask_s].copy()
    catdat_ref_sat_global = catdat_ref_global[bad_mask_r_global].copy()
    catdat_sci_sat_global = catdat_sci_global[bad_mask_s_global].copy()
    
    #Want no overlap between the two catalogs
    #If a source is in the local and global saturation catalogs, remove it from the local catalog
    #Segmentation map for the global catalog is most likely better than the local one for bright sources
    coords_ref_sat = SkyCoord(catdat_ref_sat['ALPHA_J2000'], catdat_ref_sat['DELTA_J2000'], unit=(u.deg, u.deg))
    coords_sci_sat = SkyCoord(catdat_sci_sat['ALPHA_J2000'], catdat_sci_sat['DELTA_J2000'], unit=(u.deg, u.deg))
    coords_ref_sat_global = SkyCoord(catdat_ref_sat_global['ALPHA_J2000'], catdat_ref_sat_global['DELTA_J2000'], unit=(u.deg, u.deg))
    coords_sci_sat_global = SkyCoord(catdat_sci_sat_global['ALPHA_J2000'], catdat_sci_sat_global['DELTA_J2000'], unit=(u.deg, u.deg))

    if len(coords_ref_sat_global) == 0 or len(coords_ref_sat) == 0:
        matched_ref_sat = Table(names=catdat_ref_sat_global.colnames)
        matched_ref_sat_global = Table(names=catdat_ref_sat.colnames)
    else:
        idx, d2d, _ = coords_ref_sat.match_to_catalog_sky(coords_ref_sat_global)
        mask = d2d.arcsec < 0.1
        matched_ref_sat = catdat_ref_sat_global[idx[mask]].copy()
        matched_ref_sat_global = catdat_ref_sat[mask].copy()
    
    if len(coords_sci_sat_global) == 0 or len(coords_sci_sat) == 0:
        matched_sci_sat = Table(names=catdat_sci_sat_global.colnames)
        matched_sci_sat_global = Table(names=catdat_sci_sat.colnames)
    else:
        idx, d2d, _ = coords_sci_sat.match_to_catalog_sky(coords_sci_sat_global)
        mask = d2d.arcsec < 0.1
        matched_sci_sat = catdat_sci_sat_global[idx[mask]].copy()
        matched_sci_sat_global = catdat_sci_sat[mask].copy()
    
    
    #Mask for sources in the local catalogs that are also in the global ones
    #Choose the global catalog if the source is in both
    global_better_ref = np.array([ x in matched_ref_sat['NUMBER'].data for x in catdat_ref['NUMBER'].data ], dtype=bool)
    global_better_sci = np.array([ x in matched_sci_sat['NUMBER'].data for x in catdat_sci['NUMBER'].data ], dtype=bool)

    #Mask for bright stars in the local catalogs that are not in the (full) global ones
    inds_local_only_r = catdat_ref_local_bright[ np.array([ not (x in catdat_matched_ref['NUMBER'].data) for x in catdat_ref_local_bright['NUMBER'].data ], dtype=bool) ]['NUMBER'].data
    inds_local_only_s = catdat_sci_local_bright[ np.array([ not (x in catdat_matched_ref['NUMBER'].data) for x in catdat_sci_local_bright['NUMBER'].data ], dtype=bool) ]['NUMBER'].data
    local_only_r = np.array([ x in inds_local_only_r for x in catdat_ref['NUMBER'].data ], dtype=bool)
    local_only_s = np.array([ x in inds_local_only_s for x in catdat_sci['NUMBER'].data ], dtype=bool)

    
    catdat_ref_sat = catdat_ref[ (bad_mask_r | local_only_r) & (~global_better_ref)].copy()
    catdat_sci_sat = catdat_sci[ (bad_mask_s | local_only_s) & (~global_better_sci)].copy()
    catdat_ref_sat_global = catdat_ref_global[bad_mask_r_global].copy()
    catdat_sci_sat_global = catdat_sci_global[bad_mask_s_global].copy()
    logger.info('\t Number of saturated sources in REF catalog: {}'.format(len(catdat_ref_sat)))
    logger.info('\t Number of saturated sources in SCI catalog: {}'.format(len(catdat_sci_sat)))
    logger.info('\t Number of saturated sources in REF global catalog: {}'.format(len(catdat_ref_sat_global)))
    logger.info('\t Number of saturated sources in SCI global catalog: {}'.format(len(catdat_sci_sat_global)))
    
    
    #Dim (f < 30)
        #Remove saturated sources from the global catalogs
    catdat_ref = catdat_ref[(~bad_mask_r) & (catdat_ref['FLUX_APER'] < 30) & (~global_better_ref)].copy()
    catdat_sci = catdat_sci[(~bad_mask_s) & (catdat_sci['FLUX_APER'] < 30) & (~global_better_sci)].copy()
    #Bright (f >= 30)
    catdat_ref_global = catdat_ref_global[(~bad_mask_r_global) & (catdat_ref_global['FLUX_APER'] >= 30)].copy()
    catdat_sci_global = catdat_sci_global[(~bad_mask_s_global) & (catdat_sci_global['FLUX_APER'] >= 30)].copy()
    
    catdat_ref_sat.write(outdir + '{}_saturated_sources.cat'.format(ref_name), format='fits', overwrite=True)
    catdat_sci_sat.write(outdir + '{}_saturated_sources.cat'.format(sci_name), format='fits', overwrite=True)
    catdat_ref_sat_global.write(outdir + '{}_saturated_sources_bright.cat'.format(ref_name), format='fits', overwrite=True)
    catdat_sci_sat_global.write(outdir + '{}_saturated_sources_bright.cat'.format(sci_name), format='fits', overwrite=True)


    #Get rid of objects with no px > nsig*stddev
    logger.info('Removing objects with no px > {}*stddev in cross-convolved images from catalogs'.format(nsig))
    with WorkerPool(n_jobs=ncpu) as pool:
        func = partial(check_nsig_bkg, sexseg=seg_ref.astype(int), bkgstd_mask_ind=np.argwhere(im_ref_cc > nsig*bkgstd_ref) )
        good_mask_r = pool.map(func, iter(catdat_ref['NUMBER'].data), iterable_len=len(catdat_ref), 
                              progress_bar=True, progress_bar_style='rich')

        func = partial(check_nsig_bkg, sexseg=seg_sci.astype(int), bkgstd_mask_ind=np.argwhere(im_sci_cc > nsig*bkgstd_sci) )
        good_mask_s = pool.map(func, iter(catdat_sci['NUMBER'].data), iterable_len=len(catdat_sci), 
                              progress_bar=True, progress_bar_style='rich')
        
        func = partial(check_nsig_bkg, sexseg=seg_ref_global.astype(int), bkgstd_mask_ind=np.argwhere(im_ref_cc > nsig*bkgstd_ref) )
        good_mask_r_global = pool.map(func, iter(catdat_ref_global['NUMBER'].data), iterable_len=len(catdat_ref_global), 
                              progress_bar=True, progress_bar_style='rich')
        
        func = partial(check_nsig_bkg, sexseg=seg_sci_global.astype(int), bkgstd_mask_ind=np.argwhere(im_sci_cc > nsig*bkgstd_sci) )
        good_mask_s_global = pool.map(func, iter(catdat_sci_global['NUMBER'].data), iterable_len=len(catdat_sci_global), 
                              progress_bar=True, progress_bar_style='rich')
        
    good_mask_r = np.array(good_mask_r, dtype=bool)
    good_mask_s = np.array(good_mask_s, dtype=bool)
    good_mask_r_global = np.array(good_mask_r_global, dtype=bool)
    good_mask_s_global = np.array(good_mask_s_global, dtype=bool)
    
    
    catdat_ref = catdat_ref[good_mask_r].copy()
    catdat_sci = catdat_sci[good_mask_s].copy()
    catdat_ref_global = catdat_ref_global[good_mask_r_global].copy()
    catdat_sci_global = catdat_sci_global[good_mask_s_global].copy()
    logger.info('\t Number of sources with no px > {}*stddev in REF: {}'.format(nsig, np.sum(~good_mask_r)))
    logger.info('\t Number of sources with no px > {}*stddev in SCI: {}'.format(nsig, np.sum(~good_mask_s)))
    logger.info('\t Number of sources with no px > {}*stddev in REF global: {}'.format(nsig, np.sum(~good_mask_r_global)))
    logger.info('\t Number of sources with no px > {}*stddev in SCI global: {}'.format(nsig, np.sum(~good_mask_s_global)))


    logger.info('Matching catalogs for REF and SCI')
    coords_ref = SkyCoord(catdat_ref['ALPHA_J2000'], catdat_ref['DELTA_J2000'], unit=(u.deg, u.deg))
    coords_sci = SkyCoord(catdat_sci['ALPHA_J2000'], catdat_sci['DELTA_J2000'], unit=(u.deg, u.deg))
    coords_ref_global = SkyCoord(catdat_ref_global['ALPHA_J2000'], catdat_ref_global['DELTA_J2000'], unit=(u.deg, u.deg))
    coords_sci_global = SkyCoord(catdat_sci_global['ALPHA_J2000'], catdat_sci_global['DELTA_J2000'], unit=(u.deg, u.deg))
    
    #Match coords
    if len(coords_ref) > 0:
        idx, d2d, _ = coords_sci.match_to_catalog_sky(coords_ref)
        mask = d2d.arcsec < .1
        
        matched_ref = catdat_ref[idx[mask]]
        matched_sci = catdat_sci[mask]
    else:
        matched_ref = Table(names=catdat_ref.colnames, dtype=catdat_ref.dtype)
        matched_sci = Table(names=catdat_sci.colnames, dtype=catdat_sci.dtype)
        idx = np.array([])
        mask = np.zeros(len(catdat_sci), dtype=int)
        
    if len(coords_ref_global) > 0:
        idx_global, d2d_global, _ = coords_sci_global.match_to_catalog_sky(coords_ref_global)
        mask_global = d2d_global.arcsec < .1
        
        matched_ref_global = catdat_ref_global[idx_global[mask_global]]
        matched_sci_global = catdat_sci_global[mask_global]
    else:
        matched_ref_global = Table(names=catdat_ref_global.colnames, dtype=catdat_ref_global.dtype)
        matched_sci_global = Table(names=catdat_sci_global.colnames, dtype=catdat_sci_global.dtype)
        idx_global = np.array([])
        mask_global = np.zeros(len(catdat_sci_global), dtype=int)

    logger.info('\t Number of matched sources: {}'.format(len(matched_ref)))
    logger.info('\t Number of matched sources in global catalogs: {}'.format(len(matched_ref_global)))
    
    
    
    
    #Make mask
    logger.info('Constructing SFFT mask')
    sfft_mask = np.ones_like(seg_sci, dtype=int)
    
    #Mask all pixels in input masks
    sfft_mask[im_mref] = 0
    sfft_mask[im_msci] = 0
    
    #Mask all background pixels
    logger.info('Masking background pixels')
    sfft_mask[(seg_ref == 0) & (seg_ref_global == 0)] = 0
    sfft_mask[(seg_sci == 0) & (seg_sci_global == 0)] = 0
    
    #Mask all px <nsig*stddev in cross-convolved images
    logger.info('Masking px with <{}*stddev in cross-convolved images'.format(nsig))
    sfft_mask[im_ref_cc < nsig*bkgstd_ref] = 0
    sfft_mask[im_sci_cc < nsig*bkgstd_sci] = 0

        
        
        
    #Make mask shared for all CPUs
    logger.info('Creating shared mask object')
    sfft_mask_shared = mp.Array(ctypes.c_double, sfft_mask.flatten())
    
    #Mask all px for saturated stars
    logger.info('Masking px for saturated stars')
    
    with WorkerPool(n_jobs=ncpu) as pool:        
        func = partial(apply_mask_total, mask_flat=sfft_mask_shared, sexseg=seg_ref.astype(int))
        _ = pool.map(func, iter(catdat_ref_sat['NUMBER'].data), iterable_len=len(catdat_ref_sat), 
                            progress_bar=True, progress_bar_style='rich')
        
        func = partial(apply_mask_total, mask_flat=sfft_mask_shared, sexseg=seg_sci.astype(int))
        _ = pool.map(func, iter(catdat_sci_sat['NUMBER'].data), iterable_len=len(catdat_sci_sat), 
                            progress_bar=True, progress_bar_style='rich')
        
        func = partial(apply_mask_total, mask_flat=sfft_mask_shared, sexseg=seg_ref_global.astype(int))
        _ = pool.map(func, iter(catdat_ref_sat_global['NUMBER'].data), iterable_len=len(catdat_ref_sat_global), 
                            progress_bar=True, progress_bar_style='rich')
        
        func = partial(apply_mask_total, mask_flat=sfft_mask_shared, sexseg=seg_sci_global.astype(int))
        _ = pool.map(func, iter(catdat_sci_sat_global['NUMBER'].data), iterable_len=len(catdat_sci_sat_global), 
                            progress_bar=True, progress_bar_style='rich')

        

    #Mask all px with unmatched sources and flag==0
    logger.info('Masking px with unmatched sources and FLAGS==0')
    
    unmatched_sci = catdat_sci[~mask].copy()
    unmatched_sci_flag0 = unmatched_sci[unmatched_sci['FLAGS'] == 0].copy()
    
    unmatched_sci_global = catdat_sci_global[~mask_global].copy()
    unmatched_sci_global_flag0 = unmatched_sci_global[unmatched_sci_global['FLAGS'] == 0].copy()
    
    if len(coords_ref) > 0:
        unmatched_ref_mask = np.atleast_1d(  np.ones(len(catdat_ref), dtype=int)  )
        unmatched_ref_mask[idx[mask]] = 0
        
        unmatched_ref_mask = unmatched_ref_mask.astype(bool)
        unmatched_ref = catdat_ref[unmatched_ref_mask].copy()
        unmatched_ref_flag0 = unmatched_ref[unmatched_ref['FLAGS'] == 0].copy()
            
    else:
        unmatched_ref = Table(names=catdat_ref.colnames, dtype=catdat_ref.dtype)
        unmatched_ref_flag0 = Table(names=catdat_ref.colnames, dtype=catdat_ref.dtype) 
    
    
    if len(coords_ref_global) > 0:
        unmatched_ref_mask_global = np.atleast_1d(  np.ones(len(catdat_ref_global), dtype=int)  )
        unmatched_ref_mask_global[idx_global[mask_global]] = 0
        
        unmatched_ref_mask_global = unmatched_ref_mask_global.astype(bool)
        unmatched_ref_global = catdat_ref_global[unmatched_ref_mask_global].copy()
        unmatched_ref_global_flag0 = unmatched_ref_global[unmatched_ref_global['FLAGS'] == 0].copy()
        
    else:
        unmatched_ref_global = Table(names=catdat_ref_global.colnames, dtype=catdat_ref_global.dtype)
        unmatched_ref_global_flag0 = Table(names=catdat_ref_global.colnames, dtype=catdat_ref_global.dtype)

    

    pool = WorkerPool(n_jobs=ncpu)
    
    logger.info('\t Number of unmatched sources in REF: {}'.format(len(unmatched_ref)))
    logger.info('\t Number of unmatched sources in REF (FLAGS=0): {}'.format(len(unmatched_ref_flag0)) )   
    logger.info('\t Applying mask for unmatched sources in REF (using {} CPUs)'.format(ncpu))
    func = partial(apply_mask_total, mask_flat=sfft_mask_shared, sexseg=seg_ref.astype(int))
    _ = pool.map(func, iter(unmatched_ref_flag0['NUMBER'].data), iterable_len=len(unmatched_ref_flag0), 
                        progress_bar=True, progress_bar_style='rich')
    
    logger.info('\t Number of unmatched sources in REF global: {}'.format(len(unmatched_ref_global)))
    logger.info('\t Number of unmatched sources in REF global (FLAGS=0): {}'.format(len(unmatched_ref_global_flag0)) )   
    logger.info('\t Applying mask for unmatched sources in REF global (using {} CPUs)'.format(ncpu))
    func = partial(apply_mask_total, mask_flat=sfft_mask_shared, sexseg=seg_ref_global.astype(int))
    _ = pool.map(func, iter(unmatched_ref_global_flag0['NUMBER'].data), iterable_len=len(unmatched_ref_global_flag0), 
                        progress_bar=True, progress_bar_style='rich')
    
    logger.info('\t Number of unmatched sources in SCI: {}'.format(len(unmatched_sci)))
    logger.info('\t Number of unmatched sources in SCI (FLAGS=0): {}'.format(len(unmatched_sci_flag0)) )
    logger.info('\t Applying mask for unmatched sources in SCI (using {} CPUs)'.format(ncpu))
    func = partial(apply_mask_total, mask_flat=sfft_mask_shared, sexseg=seg_sci.astype(int))
    _ = pool.map(func, iter(unmatched_sci_flag0['NUMBER'].data), iterable_len=len(unmatched_sci_flag0), 
                        progress_bar=True, progress_bar_style='rich')
    
    logger.info('\t Number of unmatched sources in SCI global: {}'.format(len(unmatched_sci_global)))
    logger.info('\t Number of unmatched sources in SCI global (FLAGS=0): {}'.format(len(unmatched_sci_global_flag0)) )
    logger.info('\t Applying mask for unmatched sources in SCI global (using {} CPUs)'.format(ncpu))
    func = partial(apply_mask_total, mask_flat=sfft_mask_shared, sexseg=seg_sci_global.astype(int))
    _ = pool.map(func, iter(unmatched_sci_global_flag0['NUMBER'].data), iterable_len=len(unmatched_sci_global_flag0), 
                        progress_bar=True, progress_bar_style='rich')
    
    pool.join()    


    #Mask all px with matched abs(mag_auto_ref - mag_auto_sci) > 1 and flag==0
    logger.info('Masking px with matched |diff(MAG_AUTO)| > 1 and FLAGS==0')

    mask_flags0 = (matched_sci['FLAGS'] == 0) & (matched_ref['FLAGS'] == 0)
    mask_magdiff = np.abs(matched_sci['MAG_AUTO'] - matched_ref['MAG_AUTO']) > 1
    matched_sci_magdiff = matched_sci[mask_flags0 & mask_magdiff].copy()
    matched_ref_magdiff = matched_ref[mask_flags0 & mask_magdiff].copy()
    
    mask_flags0_global = (matched_sci_global['FLAGS'] == 0) & (matched_ref_global['FLAGS'] == 0)
    mask_magdiff_global = np.abs(matched_sci_global['MAG_AUTO'] - matched_ref_global['MAG_AUTO']) > 1
    matched_sci_magdiff_global = matched_sci_global[mask_flags0_global & mask_magdiff_global].copy()
    matched_ref_magdiff_global = matched_ref_global[mask_flags0_global & mask_magdiff_global].copy()
    
    logger.info('\t Number of matched sources with |diff(MAG_AUTO)| > 1: {}'.format(mask_magdiff.sum()))
    logger.info('\t Number of matched sources with |diff(MAG_AUTO)| > 1 (FLAGS=0): {}'.format(len(matched_sci_magdiff)) )
    
    logger.info('\t Number of matched sources in global catalogs with |diff(MAG_AUTO)| > 1: {}'.format(mask_magdiff_global.sum()))
    logger.info('\t Number of matched sources in global catalogs with |diff(MAG_AUTO)| > 1 (FLAGS=0): {}'.format(len(matched_sci_magdiff_global)) )

    logger.info('\t Applying magdiff mask (using {} CPUs)'.format(ncpu))
    pool = WorkerPool(n_jobs=ncpu)    
    func = partial(apply_mask_total, mask_flat=sfft_mask_shared, sexseg=seg_sci.astype(int))
    _ = pool.map(func, iter(matched_sci_magdiff['NUMBER'].data), iterable_len=len(matched_sci_magdiff), 
                            progress_bar=True, progress_bar_style='rich')
    
    func = partial(apply_mask_total, mask_flat=sfft_mask_shared, sexseg=seg_sci_global.astype(int))
    _ = pool.map(func, iter(matched_sci_magdiff_global['NUMBER'].data), iterable_len=len(matched_sci_magdiff_global), 
                            progress_bar=True, progress_bar_style='rich')
    
    pool.join()    
    
    
    #Mask px for matched sources that don't overlap
    # pool = WorkerPool(n_jobs=ncpu)
    
    # logger.info('Masking px for matched sources that don\'t overlap')
    # func = partial(apply_matchmask, mask_flat=sfft_mask_shared, sexseg_r=seg_ref.astype(int), sexseg_s=seg_sci.astype(int))
    # _ = pool.map(func, zip(matched_ref['NUMBER'].data, matched_sci['NUMBER'].data), iterable_len=len(matched_ref), 
    #                 progress_bar=True, progress_bar_style='rich')
    # pool.join()

    del catdat_ref, catdat_sci
    del catdat_ref_global, catdat_sci_global
    del matched_ref, matched_sci
    del matched_ref_global, matched_sci_global
    del unmatched_ref, unmatched_sci
    del unmatched_ref_global, unmatched_sci_global
    del unmatched_ref_flag0, unmatched_sci_flag0
    del unmatched_ref_global_flag0, unmatched_sci_global_flag0
    del matched_ref_magdiff, matched_sci_magdiff
    del matched_ref_magdiff_global, matched_sci_magdiff_global
    del mask_flags0, mask_magdiff
    del mask_flags0_global, mask_magdiff_global


    #Get mask
    logger.info('Converting shared mask object to array')
    sfft_mask = np.ctypeslib.as_array(sfft_mask_shared.get_obj()).reshape(sfft_mask.shape)
    

    #Save masks
    logger.info('Saving masks')
    fits.writeto(fname_out_ref, sfft_mask, overwrite=True)
    logger.info('\t Saved mask to {}'.format(fname_out_ref))

    fits.writeto(fname_out_sci, sfft_mask, overwrite=True)
    logger.info('\t Saved mask to {}'.format(fname_out_sci))
        

    del seg_ref, seg_sci
    del seg_ref_global, seg_sci_global
    del im1, im2
    del im_ref_cc, im_sci_cc
    del sfft_mask, sfft_mask_shared
    
    return