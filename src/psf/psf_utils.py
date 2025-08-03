import os
import subprocess
import logging
import shutil
import multiprocessing as mp
from mpire import WorkerPool
from functools import partial

import numpy as np

from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.nddata.utils import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy import wcs
from astropy.table import Table

from photutils.centroids import centroid_com, centroid_sources
from photutils.psf import extract_stars, EPSFBuilder, EPSFModel, EPSFStar, EPSFStars
from reproject import reproject_exact
from reproject.mosaicking import reproject_and_coadd

import gc



def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)      

    logger = logging.getLogger(name)
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)

    logger.setLevel(level)
    
    if not len(logger.handlers): 
        logger.addHandler(handler)

    return handler, logger




## empirical PSF FWHM values measured from the Cycle 1 Absolute Flux calibration program, pixel/FWHM
# https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions
# FWHM values in px

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


EUCLID = {'NIR_J': np.nan,
          'NIR_H': np.nan,
          'NIR_Y': np.nan}


def get_filter(fname):
    filters_all = list(NIRCam_filter_FWHM['SW'].keys()) + list(NIRCam_filter_FWHM['LW'].keys()) + list(EUCLID.keys())
    filters_all_lc = [f.lower() for f in filters_all]
    
    try:
        header = fits.getheader(fname)
        return header['FILTER']
    except:
        pass


    fname = os.path.basename(fname)
    try:
        for f in filters_all:
            if f in fname:
                return f
    except:
        pass
    

    try:
        for f in filters_all_lc:
            if f in fname:
                return f.upper()
    except:
        pass
    
    raise ValueError('Filter not found in filename/header')
    return
    

def PSF_poly_3rd(x, y, a, b, c, d, e, f, g, h, i, j):
    return a + b*x + c*x**2 + d*x**3 + e*y + f*x*y + g*x**2*y + h*y**2 + i*x*y**2 + j*x**3

def PSF_poly_2nd(x, y, a, b, c, d, e, f):
    return a+b*x+c*x**2+d*y+e*x*y+f*y**2
    




# VIGNET size is based on enclosed energy curves from https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions
# To enclose >97% of the total light in all corresponding filters:
# SW = 4" (133 pixel with 0.03 pixel size) >~98% F200W
# LW = 6" (201 pixel with 0.03 pixel size) >~97% F444W
# star cutout size = 133 for SW and 201 for LW

############################################################################################################
############################################################################################################
# Run SExtractor, PSFEx, SWarp, photutils

def run_sextractor(indir, outdir, 
                   fname_default_config, fname_sw_param, fname_lw_param, fname_euclid_param,
                   fname_conv, fname_nnw,
                   fname_image, fname_weight, fname_flag,
                   filtername, prefix, fwhm_px=None, ps_out=.03, nsig=5., ncpu=1):
    
    """Sets up the config and runs SExtractor on the input image. The SExtractor will be placed in 
    ``outdir" with the prefix ``prefix". The output catalog will be named ``{prefix}.cat".
    
    Parameters:
    -----------
    indir : str
        The input directory.
    outdir : str
        The output directory.
    fname_default_config : str
        The default SExtractor config filename.
    fname_image : str
        The image filename (no path).
    fname_weight : str
        The weight map filename (no path).
    fname_flag : str
        The flag filename (no path).
    filtername : str
        The name of the JWST filter.
    prefix : str
        The prefix for the output files.
    fwhm_px : float
        The FWHM of the PSF in Euclid pixels (for EUCLID only).
    ps_out : float
        The pixel scale in the image (arcsec/pixel). This may be different from the innate pixel scale of the input image,
        say if the image was resampled onto a larger/finer grid.

    Raises:
        ValueError: _description_
    """
    
    
    logfile = outdir + prefix + '.log'
    _, logger = setup_logger('psf.{}'.format(prefix), logfile)

    logger.info('Prepping SExtractor run for {}'.format(prefix))

    # Load the default config
    f_def = open(fname_default_config, 'r')
    content = f_def.readlines()
    f_def.close()
    
    if (filtername in NIRCam_filter_FWHM_new['SW'].keys()) or (filtername in NIRCam_filter_FWHM_new['LW'].keys()):
        header = fits.getheader(indir + fname_image)
        mag_zp = header['MAG_ZP']
        exptime = header['EXPTIME']
        photmjsr = header['PHOTMJSR']
        
        zp_new = mag_zp #+ 2.5 * np.log10(exptime/photmjsr)

    elif filtername in EUCLID.keys():
        header = fits.getheader(indir + fname_image)
        zp_new = header['MAG_ZP']
    
    
    
    if filtername in SW.keys():
        channel = 'SW'
        ps_in = .031    #arcsec/px
        content[9] = 'PARAMETERS_NAME  {}  # name of the file containing catalog contents\n'.format(fname_sw_param)
    elif filtername in LW.keys():
        channel = 'LW'
        ps_in = .063   #arcsec/px
        content[9] = 'PARAMETERS_NAME  {}  # name of the file containing catalog contents\n'.format(fname_lw_param)
    elif filtername in EUCLID.keys():
        channel = 'EUCLID'
        ps_in = .1  #arcsec/px
        content[9] = 'PARAMETERS_NAME  {}  # name of the file containing catalog contents\n'.format(fname_euclid_param)
    else:
        print('Input filter name not part of NIRCam/EUCLID')
        raise ValueError

    #CANT ASSUME THIS ANYMORE
    #Assume if ref fiter != sci filter, then ref=JWST and sci=EUCLID
    if channel in ['SW', 'LW']:
        phot_aper = NIRCam_filter_FWHM[channel][filtername]*5*ps_in/ps_out  # 5 times FWHM
        seeing_fwhm = NIRCam_filter_FWHM[channel][filtername]*ps_in
    else:
        if fwhm_px is None:
            logger.error('FWHM not provided for EUCLID filter')
            raise ValueError('FWHM not provided for EUCLID filter')

        phot_aper = fwhm_px*(ps_in/ps_out)*5   #px
        seeing_fwhm = fwhm_px*ps_in         #arcsec
    

    #Change config values
    content[6]   = "CATALOG_NAME     {}              # name of the output catalog\n".format(outdir + prefix + '.cat')
    content[7]   = "CATALOG_TYPE     FITS_LDAC       # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,\n"
    content[18]  = "DETECT_THRESH    {}              # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n".format(nsig)
    content[19]  = "ANALYSIS_THRESH  {}              # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n".format(nsig) 
    content[22]  = "FILTER_NAME      {}              # name of the file containing the filter\n".format(fname_conv)
    content[38]  = "WEIGHT_IMAGE     {}              # weight-map filename\n".format(indir + fname_weight)
    content[44]  = "FLAG_IMAGE       {}              # filename for an input FLAG-image\n".format(indir + fname_flag)

    # use 5 x PSF FWHM (*5*2, *2 is due to half pixel drizzling)
    content[55]  = "PHOT_APERTURES   {}              # MAG_APER aperture diameter(s) in pixels\n".format(phot_aper)  # 5 times FWHM
    content[63]  = "SATUR_LEVEL      1e9             # level (in ADUs) at which arises saturation\n"
    content[66]  = "MAG_ZEROPOINT    {}              # magnitude zero-point\n".format(zp_new)

    content[74]  = 'SEEING_FWHM      {:.3f}          # stellar FWHM in arcsec\n'.format(seeing_fwhm)
    content[75]  = 'STARNNW_NAME     {}              # Neural-Network_Weight table filename\n'.format(fname_nnw)
    content[95]  = 'CHECKIMAGE_NAME  {}              # Filename for the check-image\n'.format(outdir + prefix + '_sex_seg.fits')
    content[122] = 'NTHREADS         {}              # 1 single thread\n'.format(int(ncpu))

    # Write the new config
    fname_config = outdir + prefix + '.sex'
    f2 = open(fname_config, 'w')
    f2.writelines(content)
    f2.close()
    logger.info('SExtractor config file created at {}'.format(fname_config))
    
    
    #Run SExtractor
    if not os.path.exists(outdir + prefix + '.cat'):
        logger.info('Running SExtractor for {}'.format(prefix))
        
        f = open(logfile, 'a')
        out = subprocess.run(['sex', indir+fname_image, '-c', fname_config], check=True, stdout=f).stdout
        f.close()
        
        logger.info('SExtractor run complete for {}'.format(prefix))
        

    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
        
    return


def update_psfex_psf_param(outdir, filtername, prefix, oversampling_vals=[1,2], fwhm_px=None, ps_out=.03):
    for oversampling in oversampling_vals:        
        ps = ps_out/oversampling
        
        if filtername in NIRCam_filter_FWHM['SW'].keys():
            fwhm_estimate = NIRCam_filter_FWHM['SW'][filtername] * oversampling * (.03/ps_out)
        elif filtername in NIRCam_filter_FWHM['LW'].keys():
            fwhm_estimate = NIRCam_filter_FWHM['LW'][filtername] * oversampling * (2*.03/ps_out)
        elif filtername in EUCLID.keys():
            if fwhm_px is None:
                print('FWHM not provided for EUCLID filter')
                raise ValueError('FWHM not provided for EUCLID filter')
            
            fwhm_estimate = fwhm_px * oversampling * (3.333333333*.03/ps_out)
    
        fname_psf = outdir + prefix + '_psfex_PSF_c_{}.fits'.format(oversampling)
        fit_psf_and_update_header(fname_psf, fwhm_estimate, ps, centers=None)
        
    return



def run_psfex(outdir, 
              fname_sexcat, fname_default_config, 
              oversampling, filtername, prefix, SNR_WIN=100, ps_out=.03, 
              constant=True, var_2o=False, ncpu=1):

    f = open(fname_default_config, 'r')
    content = f.readlines()
    f.close()
    
    
    
    if filtername in NIRCam_filter_FWHM['SW'].keys():
        psf_size1 = int( 132 * (.03/ps_out) ) +1
        psf_size2 = int( 264 * (.03/ps_out) ) +1
        
        fwhm_range1 = 1.0 * (.03/ps_out)
        fwhm_range2 = 5.0 * (.03/ps_out)
        
        if oversampling == 1:
            # oversampling factor = 1 for SW (0.03 "/pixel)
            content[13] = 'PSF_SAMPLING    1          # Sampling step in pixel units (0.0 = auto)\n'
            # outsize
            content[16] = 'PSF_SIZE        {},{}    # Image size of the PSF model\n'.format(psf_size1, psf_size1)
        elif oversampling == 2:
            # oversampling factor = 2 for SW (0.015 "/pixel)
            content[13] = 'PSF_SAMPLING    0.5        # Sampling step in pixel units (0.0 = auto)\n'
            # outsize
            content[16] = 'PSF_SIZE        {},{}    # Image size of the PSF model\n'.format(psf_size2, psf_size2)

        #FWHM range
        content[40] = 'SAMPLE_FWHMRANGE   {},{}     # Allowed FWHM range\n'.format(fwhm_range1, fwhm_range2)
        
        
    
    elif filtername in NIRCam_filter_FWHM['LW'].keys():
        psf_size1 = int( 200 * (.03/ps_out) ) +1
        psf_size2 = int( 400 * (.03/ps_out) ) +1
        
        fwhm_range1 = 2.0 * (.03/ps_out)
        fwhm_range2 = 9.0 * (.03/ps_out)
        
        if oversampling == 1:
            # oversampling factor = 1 for LW (0.03 "/pixel)
            content[13] = 'PSF_SAMPLING    1                 # Sampling step in pixel units (0.0 = auto)\n'
            # outsize
            content[16] = 'PSF_SIZE        {},{}           # Image size of the PSF model\n'.format(psf_size1, psf_size1)
        elif oversampling == 2:
            # oversampling factor = 2 for LW (0.015 "/pixel)
            content[13] = 'PSF_SAMPLING    0.5               # Sampling step in pixel units (0.0 = auto)\n'
            # outsize
            content[16] = 'PSF_SIZE        {},{}           # Image size of the PSF model\n'.format(psf_size2, psf_size2)

        #FWHM range
        content[40] = 'SAMPLE_FWHMRANGE   {},{}     # Allowed FWHM range\n'.format(fwhm_range1, fwhm_range2)



    elif filtername in EUCLID.keys():
        #(11x11) with .1 "/px = (35x35) with 0.03 "/px
        
        psf_size1 = int( 132 * (.03/ps_out) ) +1
        psf_size2 = int( 264 * (.03/ps_out) ) +1
        
        fwhm_range1 = 1.0 * (.03/ps_out)
        fwhm_range2 = 9.0 * (.03/ps_out)
        
        if oversampling == 1:
            # oversampling factor = 1 for euclid (0.03 "/pixel)
            content[13] = 'PSF_SAMPLING    1                 # Sampling step in pixel units (0.0 = auto)\n'
            # outsize
            content[16] = 'PSF_SIZE        {},{}             # Image size of the PSF model\n'.format(psf_size1, psf_size1)
        elif oversampling == 2:
            # oversampling factor = 2 for euclid (0.015 "/pixel)
            content[13] = 'PSF_SAMPLING    0.5               # Sampling step in pixel units (0.0 = auto)\n'
            # outsize
            content[16] = 'PSF_SIZE        {},{}             # Image size of the PSF model\n'.format(psf_size2, psf_size2)
            

        content[38] = "SAMPLE_AUTOSELECT  N            # Automatically select the FWHM (Y/N) ?\n"
            
        #FWHM range
        content[40] = 'SAMPLE_FWHMRANGE   {},{}     # Allowed FWHM range\n'.format(fwhm_range1, fwhm_range2)


    content[42] = "SAMPLE_MINSN       {}          # Minimum S/N for a source to be used\n".format(SNR_WIN)
        
    #Suffix
    if constant:
        content[85] = 'PSF_SUFFIX      _c_{}.psf          # Filename extension for output PSF filename\n'.format(oversampling)
    else:
        if var_2o:
            content[31] = 'PSFVAR_DEGREES  2               # Polynom degree for each group\n'
            content[85] = 'PSF_SUFFIX      _v2_{}.psf          # Filename extension for output PSF filename\n'.format(oversampling)
        else:
            content[85] = 'PSF_SUFFIX      _v_{}.psf          # Filename extension for output PSF filename\n'.format(oversampling)
    
    content[91] = 'NTHREADS        {}               # Number of simultaneous threads for\n'.format(ncpu)
        
        
    #Save config
    if constant:
        fname_config = outdir + prefix + '_c_{}.psfex'.format(oversampling)
    else:
        if var_2o:
            fname_config = outdir + prefix + '_v2_{}.psfex'.format(oversampling)
        else:
            fname_config = outdir + prefix + '_v_{}.psfex'.format(oversampling)
        
    f = open(fname_config, 'w')
    f.writelines(content)
    f.close()
    
    
    logfile = outdir + prefix + '.log'
    _, logger = setup_logger('psf.{}'.format(prefix), logfile)
    
    #Run PSFEx
    logger.info('Running PSFEx for {}'.format(prefix))
    logger.info('\t\t Assuming constant: {}'.format(constant))
    logger.info('\t\t Using oversampling={}'.format(oversampling))
    logger.info('\t\t Using {} cores'.format(ncpu))
    logger.info('\t\t Using config file {}'.format(fname_config))
    f = open(logfile, 'a')
    out = subprocess.run(['psfex', fname_sexcat, '-c', fname_config], check=True, stdout=f).stdout
    f.close()
    logger.info('PSFEx run complete for {}, oversampling={}'.format(prefix, oversampling))
    
    #Change filename
    logger.info( 'PSFEx PSF saved as {}'.format(fname_sexcat.replace('.cat', '_{}.psf'.format(oversampling))) )    
    
    
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    
    return


# ONLY WORKS FOR JWST NIRCAM
def run_swarp(outdir, stardir, 
              fname_default_config, fname_stars, 
              filtername, prefix, oversampling_vals=None, fwhm_only=True):
    if oversampling_vals is None:
        oversampling_vals = [2]
    
    
    f = open(fname_default_config, 'r')
    content = f.readlines()
    f.close()
    
    if not fwhm_only:
        stardat = Table.read(fname_stars, format='ascii.ipac')
        
        if filtername in NIRCam_filter_FWHM['SW'].keys():
            crpx_in = (int(68), int(68))
        else:
            crpx_in = (int(100), int(100))
            
        fname_temp = stardir + 'temp.fits'
        if not os.path.exists(fname_temp):
            header = fits.getheader(stardir + str(stardat[0]['Index']) + '.fits')
            star = fits.getdata(stardir + str(stardat[0]['Index']) + '.fits')
            
            blank = np.zeros_like(star)
            wcs_temp = wcs.WCS(header)
            wcs_temp.wcs.crpix = crpx_in
            
            CRVAL1, CRVAL2 = wcs_temp.wcs.crval
            
            hdu = fits.PrimaryHDU(blank, header=temp_wcs.to_header())
            hdu.writeto(fname_temp, overwrite=True)
            
        else:
            header = fits.getheader(fname_temp)
            CRVAL1 = header['CRVAL1']
            CRVAL2 = header['CRVAL2']
            
        
        #Get list of stars
        f = open(stardir + 'starlist.dat', 'w')
        f.writelines( fname_temp + '\n' )
        
        for i in range(len(stardat)):
            fname_i = stardir + str(stardat[0]['Index']) + '.fits'
            f.writelines(fname_i + '\n')
        f.close()
        del f

        for i in range(len(oversampling_vals)):
            ps = .03/oversampling_vals[i]
            nsamp = oversampling_vals[i]
            
            if filtername in NIRCam_filter_FWHM['SW'].keys():
                if oversampling_vals[i] == 1:
                    shape = (133, 133)
                    x_cent = 67.
                elif oversampling_vals[i] == 2:
                    shape = (265, 265)
                    x_cent = 133.

            else:
                if oversampling_vals[i] == 1:
                    shape = (201, 201)
                    x_cent = 101.
                elif oversampling_vals[i] == 2:
                    shape = (401, 401)
                    x_cent = 201.
                    
                    
            header['NAXIS1'] = shape[0]
            header['NAXIS2'] = shape[1]
            header['CRPIX1'] = x_cent
            header['CRPIX2'] = x_cent
            header['CDELT1'] = header['CDELT1']/oversampling_vals[i]
            header['CDELT2'] = header['CDELT2']/oversampling_vals[i]
            header.tofile(outdir + '{}_SWarp_{}.head'.format(prefix, nsamp))
            
            
            #SWarp config
            content[4]  = 'IMAGEOUT_NAME          {}_SWarp_PSF_{}.fits                           # Output filename\n'.format(outdir+prefix, nsamp)
            content[5]  = 'WEIGHTOUT_NAME         {}psf.weight.fits                              # Output weight-map filename\n'.format(stardir)
            content[34] = 'CENTER                 {}, {}                                        # Coordinates of the image center\n'.format(CRVAL1, CRVAL2)
            content[36] = 'PIXEL_SCALE            {}                                            # Pixel scale\n'.format(ps)
            content[37] = 'IMAGE_SIZE             {}                                            # Image size (0 = AUTOMATIC)\n'.format(shape[0])
            content[47] = 'OVERSAMPLING           {}                                            # Oversampling in each dimension\n'.format(N_SAMP)


            f = open( stardir + 'constant_psf.swarp', 'w')
            f.writelines(content)
            f.close()
            del f
            
            #Run SWarp
            fname_out = outdir + '{}_SWarp_PSF_{}.fits'.format(prefix, nsamp)
            if not os.path.exists(fname_out):
                os.system('swarp @{} -c {}'.format(stardir + 'starlist.dat', stardir + 'constant_psf.swarp'))
                
    else:
        for i in range(len(oversampling_vals)):
            ps = .03/oversampling_vals[i]
            
            if filtername in NIRCam_filter_FWHM_new['SW'].keys():
                fwhm_estimate = NIRCam_filter_FWHM_new['SW'][filtername] * oversampling_vals[i]
            else:
                fwhm_estimate = NIRCam_filter_FWHM_new['LW'][filtername] * oversampling_vals[i]*2
                
            psf_fname = outdir + '{}_SWarp_PSF_{}.fits'.format(prefix, oversampling_vals[i])
            fit_psf_and_update_header(psf_fname, fwhm_estimate, ps, centers=None)
            
    return




def star_func(i, stardat, seg, im, im_mask, cutout_size):
    idnum = stardat[i]['NUMBER']
    mask_i = ((seg != 0) & (seg != idnum)) | im_mask

    im_i = im.copy()
    im_i[mask_i] = 0
    cutout = Cutout2D(im_i, (stardat[i]['x'], stardat[i]['y']), cutout_size, mode='partial', fill_value=0.)

    #Get and subtract local median bkg
    mean, median, std = sigma_clipped_stats(cutout.data, sigma=2, maxiters=10)
    x_new, y_new = centroid_sources(cutout.data-median, (cutout_size-1)/2, (cutout_size-1)/2, 
                                    box_size=3, centroid_func=centroid_com)
    
    return EPSFStar(cutout.data, cutout_center=(x_new,y_new), origin=cutout.origin_original)


#ONLY WORKS FOR JWST NIRCam
def get_psf_stars_photutils(outdir, stardir,
                            fname_image, fname_mask, 
                            fname_sexcat, fname_pointsource_candidates, fname_sexseg,
                            filtername, prefix, oversampling, save_stars=False, os_save_star=1,
                            ncpu=1):
    
    """Select stars and use them to generate the PSF from photutils. Need to run with oversampling=1 first
    to get the stars and save them in ``stardir". Each star is saved as a separate FITS file labeled by its
    index. The total starlist is saved as an IPAC table in ``stardir" with the name ``{prefix}_stars.ipac".
    The PSF is saved in ``outdir" with the name ``{prefix}_photutils_PSF_{oversampling}.fits".

    Parameters:
    -----------
    outdir : str
        The output directory for the PSF.
    stardir : str
        The directory to save the stars.
    fname_image : str
        The image filename (full path).
    fname_mask : str
        The mask filename (full path).
    fname_sexcat : str
        The SExtractor catalog filename (full path).
    fname_pointsource_candidates : str
        The point source candidates catalog filename (full path).
    fname_sexseg : str
        The SExtractor segmentation map filename (full path).
    filtername : str
        Name of the JWST filter.
    prefix : str
        Prefix for the output files.
    oversampling : int
        The oversampling factor. Can be either 1 or 2.
    save_stars : bool
        Whether to save the stars or not.

    Raises:
        ValueError: _description_
        ValueError: _description_
    """
    
    if filtername in NIRCam_filter_FWHM['SW'].keys():
        cutout_size = 135
        
        if oversampling == 1:
            nsamp = 1
            shape = (133, 133)
        elif oversampling == 2:
            nsamp = 2
            shape = (265, 265)
        else:
            print('Oversampling value not supported')
            raise ValueError
        
    else:
        cutout_size = 203
        
        if oversampling == 1:
            nsamp = 1
            shape = (201, 201)
        elif oversampling == 2:
            nsamp = 2
            shape = (401, 401)
        else:
            print('Oversampling value not supported')
            raise ValueError
        
    
    
    epsf_starlist = []
    if (save_stars) and (oversampling == 2):
        
        #Data
        hdul1 = fits.open(fname_image)
        im = hdul1[0].data
        header = hdul1[0].header
        
        #Sex seg
        hdul2 = fits.open(fname_sexseg)
        seg = hdul2[0].data
        
        #Mask
        hdul3 = fits.open(fname_mask)
        im_mask = hdul3[0].data
        im_mask.astype(bool)
        
        #Master source catalog
        hdul4 = fits.open(fname_pointsource_candidates)
        stardat = Table(hdul4[2].data)
        stardat['x'] = stardat['X_IMAGE']-1
        stardat['y'] = stardat['Y_IMAGE']-1
        del stardat['VIGNET']
        
        for h in [hdul1, hdul2, hdul3, hdul4]:
            h.close()
            
        
        print('Number of stars in {}: {}'.format(filtername, len(stardat)))
        print('Generating PSFs for all stars with {} cores...'.format(ncpu))
        func_i = partial(star_func, stardat=stardat, seg=seg, im=im, im_mask=im_mask, cutout_size=cutout_size)        
        
        pool = WorkerPool(n_jobs=ncpu)
        epsf_starlist = pool.map(func_i, range(len(stardat)), progress_bar=True, progress_bar_style='rich')
        pool.join()
        
        # for i in range(len(stardat)):
        #     idnum = stardat[i]['NUMBER']
        #     mask_i = ((seg != 0) & (seg != idnum)) | im_mask
        
        #     im_i = im.copy()
        #     im_i[mask_i] = 0
        #     cutout = Cutout2D(im_i, (stardat[i]['x'], stardat[i]['y']), cutout_size, mode='partial', fill_value=0.)

        #     #Get and subtract local median bkg
        #     mean, median, std = sigma_clipped_stats(cutout.data, sigma=2, maxiters=10)
        #     x_new, y_new = centroid_sources(cutout.data-median, (cutout_size-1)/2, (cutout_size-1)/2, 
        #                                     box_size=3, centroid_func=centroid_com)
            
        #     epsf_starlist.append(
        #         EPSFStar(cutout.data, cutout_center=(x_new,y_new), origin=cutout.origin_original)
        #     )
        
    else:
        stardat = Table.read(stardir + prefix + '_stars.ipac', format='ascii.ipac')
        
        
        for i in range(len(stardat)):
            ind = stardat[i]['Index']
            
            with fits.open(stardir + '{}.fits'.format(ind), memmap=False) as hdul:
                im_star = hdul[0].data
                
                x_new = stardat[i]['x_cutout']
                y_new = stardat[i]['y_cutout']
                
                epsf_starlist.append(
                    EPSFStar(im_star, cutout_center=(x_new, y_new))
                )
                del im_star
                del hdul[0].data
                
            if i % 100 == 0:
                gc.collect()
            
        

    stars_new = EPSFStars(epsf_starlist)
    epsf_builder = EPSFBuilder(oversampling=nsamp, smoothing_kernel='quadratic', norm_radius=5,
                                recentering_boxsize=(3,3), shape=shape, maxiters=20)
    epsf, fitted_stars = epsf_builder(stars_new)
    
    
    hdu = fits.PrimaryHDU(epsf.data)
    hdu.header['NSTARS_USED'] = fitted_stars.n_good_stars
    hdu.header['NSTARS'] = fitted_stars.n_stars
    hdu.header['N_SAMP'] = nsamp
    hdu.writeto(outdir + prefix + '_photutils_PSF_{}.fits'.format(oversampling), overwrite=True)
    del hdu

    print('Saving stars...')
    if (save_stars) and (oversampling == os_save_star):
        if not os.path.exists(stardir):
            os.mkdir(stardir)
            
        results = []
        good_stars = fitted_stars.all_good_stars  
        
        for i in range(len(good_stars)):
            x_og = good_stars[i].center[0][0]
            y_og = good_stars[i].center[0][1]
            mask = np.sqrt( (x_og-stardat['x'])**2 + (y_og-stardat['y'])**2 ) < 2
            
            if len(stardat[mask]) == 0:
                print(good_stars[i].center)
                continue
            
            snr = stardat[mask][0]['SNR_WIN']
            row = [i, good_stars[i].center[0][0], good_stars[i].center[0][1],
                   good_stars[i].cutout_center[0][0], good_stars[i].cutout_center[1][0], snr]
            results.append(row)
            
            
            
            hdu = fits.PrimaryHDU(good_stars[i].data)
            hdu.header['CRPIX1'] = round(good_stars[i].cutout_center[0][0] + 1, 3)
            hdu.header['CRPIX2'] = round(good_stars[i].cutout_center[1][0] + 1, 3)
            
            for key in ['CRVAL1', 'CRVAL2', 'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2', 'CDELT1', 'CDELT2', 'PC1_1', 'PC1_2',
                        'PC2_1', 'PC2_2']:
                hdu.header[key] = header[key]
                
            hdu.header['EXPTIME'] = header['EXPTIME']
            hdu.header['GAIN'] = header['GAIN']
            hdu.headerp['NCOMBINE'] = header['NCOMBINE']
            
            hdu.writeto(stardir + '{}.fits'.format(i), overwrite=True)
            del hdu
            
        resdat = Table(rows=results, names=['Index', 'x_ori', 'y_ori', 'x_cutout', 'y_cutout', 'SNR_WIN'])
        resdat['x_ori'].format = '%.3f'
        resdat['y_ori'].format = '%.3f'
        resdat['x_cutout'].format = '%.3f'
        resdat['y_cutout'].format = '%.3f'
        resdat['SNR_WIN'].format = '%.1f'
        resdat.write(stardir + '{}_stars.ipac'.format(prefix), format='ascii.ipac', overwrite=True)
        
    return
            
            
            
############################################################################################################
############################################################################################################
# Fit PSF

def fit_psf_and_update_header(fname_psf, fwhm_estimate, ps, centers=None):
    
    if centers is not None:
        
        if '_v2' in fname_psf:
            psf_header = fits.getheader(fname_psf.replace('_v2', '_c'))
        elif '_v' in fname_psf:
            psf_header = fits.getheader(fname_psf.replace('_v', '_c'))
    
        window_size = psf_header['win_size']
        
        with fits.open(fname_psf, mode='update') as hdu:
            
            for i in range(len(centers)):
                x0 = centers[i]['x']
                y0 = centers[i]['y']
                param = estimate_psf_fwhm(hdu[i].data, x0, y0, fwhm_estimate, ps, True, window_size)
                
                for key in param.keys():
                    value = param[key]
                    hdu[i].header[key] = value
                    
    else:
        with fits.open(fname_psf, mode='update') as hdu:
            param = estimate_psf_fwhm(hdu[0].data, 0, 0, fwhm_estimate, ps)
            
            for key in param.keys():
                value = param[key]
                hdu[0].header[key] = value
                
    return

############################################################################################################
############################################################################################################
# Estimate PSF

def psf_polynomial_2(x, y, a, b, c, d, e, f):
    return a + b*x + c*y + d*x**2 + e*x*y + f*y**2

def psf_polynomial_3(x, y, a, b, c, d, e, f, g, h, i, j):
    return a + b*x + c*x**2 + d*x**3 + e*y + f*x*y + g*x**2*y + h*y**2 + i*x*y**2 + j*x**3


def estimate_psf_fwhm(im_psf, x0, y0, fwhm_estimate, ps, fix_window_size=False, window_size=None):
    
    #Normalize PSF to a sum of 1e6 for the fit
    if np.nansum(im_psf) != 0:
        im_psf[im_psf < -1e29] = 0
        im_psf /= (np.nansum(im_psf) * 1e-6)
        
        im_err = np.sqrt(im_psf)
        
        im_wht = 1/im_err
        im_wht[np.isnan(im_wht)] = 0
        
        shape = im_psf.shape
        x_cent = int(shape[1]/2)
        y_cent = int(shape[0]/2)
        
        
        
        if not fix_window_size:
            fwhm_est_int = int(fwhm_estimate/2) + 1
            
            im_psf_fit = im_psf[y_cent-fwhm_est_int:y_cent+fwhm_est_int+1, 
                                x_cent-fwhm_est_int:x_cent+fwhm_est_int+1]
            
            im_wht_fit = im_wht[y_cent-fwhm_est_int:y_cent+fwhm_est_int+1,
                                x_cent-fwhm_est_int:x_cent+fwhm_est_int+1]
            
            #Fit 2D gaussian
            out1 = models.Gaussian2D(amplitude=im_psf_fit[fwhm_est_int, fwhm_est_int],
                                     x_mean=fwhm_est_int, y_mean=fwhm_est_int,
                                     x_stddev=fwhm_est_int/2.355, y_stddev=fwhm_est_int/2.355) + \
                    models.Polynomial2D(degree=1)
                    
            y, x = np.mgrid[:fwhm_est_int*2 +1, :fwhm_est_int*2 +1]
            fitter = fitting.LevMarLSQFitter()
            res1 = fitter(out1, x, y, im_psf_fit)

            
            largest_fwhm = np.max([res1.y_stddev_0.value*2.355, res1.x_stddev_0.value*2.355])
            fwhm_major = round(largest_fwhm*ps, 5)
            
            
            
            
            # Get better estimate of FWHM
            fwhm_est_int = np.max([ int(fwhm_major/ps/2)+1, fwhm_est_int ])
            im_psf_fit = im_psf[y_cent-fwhm_est_int:y_cent+fwhm_est_int+1,
                                x_cent-fwhm_est_int:x_cent+fwhm_est_int+1]      
            im_whit_fit = im_wht[y_cent-fwhm_est_int:y_cent+fwhm_est_int+1,
                                x_cent-fwhm_est_int:x_cent+fwhm_est_int+1] 
            
            #Fit 2D gaussian
            out2 = models.Gaussian2D(amplitude=im_psf_fit[fwhm_est_int, fwhm_est_int],
                                        x_mean=fwhm_est_int, y_mean=fwhm_est_int,
                                        x_stddev=fwhm_major/ps/2.355, y_stddev=fwhm_major/ps/2.355) + \
                    models.Polynomial2D(degree=1)
                    
            y, x = np.mgrid[:fwhm_est_int*2 +1, :fwhm_est_int*2 +1]
            fitter = fitting.LevMarLSQFitter()
            res2 = fitter(out2, x, y, im_psf_fit)
            
            
            largest_fwhm = np.max([res2.y_stddev_0.value*2.355, res2.x_stddev_0.value*2.355])
            smallest_fwhm = np.min([res2.y_stddev_0.value*2.355, res2.x_stddev_0.value*2.355])
            fwhm_major = round(largest_fwhm*ps, 5)
            fwhm_minor = round(smallest_fwhm*ps, 5)
            theta = round(res2.theta_0.value, 2)
            
            if theta < 0:
                theta = theta % np.pi
            if theta > np.pi:
                theta = theta % np.pi
                
            theta = round(theta, 2)
            
            x_0 = round(res2.x_mean_0.value, 3)
            y_0 = round(res2.y_mean_0.value, 3)
            window_size = int(fwhm_est_int*2 + 1)
            
            
            
            
        else:
            half_size = int((window_size-1)/2)
            im_psf_fit = im_psf[y_cent-half_size:y_cent+half_size+1,
                                x_cent-half_size:x_cent+half_size+1]
            im_wht_fit = im_wht[y_cent-half_size:y_cent+half_size+1,
                                x_cent-half_size:x_cent+half_size+1]
            im_wht_fit /= im_wht_fit[half_size, half_size]
            
            #Fit 2D gaussian
            out = models.Gaussian2D(amplitude=im_psf_fit[half_size, half_size],
                                    x_mean=half_size, y_mean=half_size,
                                    x_stddev=fwhm_estimate/2.355, y_stddev=fwhm_estimate/2.355)
            
            y, x = np.mgrid[:window_size, :window_size]
            fitter = fitting.LevMarLSQFitter()
            res = fitter(out, x, y, im_psf_fit)
            
            try:
                ystd = res.y_stddev_0.value
                xstd = res.x_stddev_0.value
                theta = res.theta_0.value
                xmean = res.x_mean_0.value
                ymean = res.y_mean_0.value
            except:
                ystd = res.y_stddev.value
                xstd = res.x_stddev.value
                theta = res.theta.value
                xmean = res.x_mean.value
                ymean = res.y_mean.value
            
            
            largest_fwhm = np.max([ystd*2.355, xstd*2.355])
            smallest_fwhm = np.min([ystd*2.355, xstd*2.355])
            fwhm_major = round(largest_fwhm*ps, 5)
            fwhm_minor = round(smallest_fwhm*ps, 5)
            theta = round(theta, 2)
            
            while theta < 0:
                theta += np.pi
            while theta > np.pi:
                theta -= np.pi
                
            theta = round(theta, 2)
            
            x_0 = round(xmean, 3)
            y_0 = round(ymean, 3)
            
        dict_out = {'x':x0, 'y':y0, 'FWHM_major':fwhm_major, 'FWHM_minor':fwhm_minor, 'theta':theta, 
                    'x_0':x_0, 'y_0':y_0, 'ps':ps, 'win_size':window_size}
    
    else:
        dict_out = {'x':x0, 'y':y0}
        
    return dict_out    
                
        
        

def get_spatially_constant_psf(fname_psf, fname_out, filtername, oversampling, fwhm_px=None, ps_out=.03):    
    ps = ps_out / oversampling  # arcsec/pixel
    
    if filtername in NIRCam_filter_FWHM['SW'].keys():
        rough_FWHM = NIRCam_filter_FWHM_new['SW'][filtername]*oversampling * (.03/ps_out)
    elif filtername in NIRCam_filter_FWHM['LW'].keys():
        rough_FWHM = NIRCam_filter_FWHM_new['LW'][filtername] * oversampling * 2 * (.03/ps_out)
    elif filtername in EUCLID.keys():
        
        if fwhm_px is None:
            logger.error('fwhm_px is None. Please provide a value for the FWHM in pixels for Euclid images.')
            raise ValueError('fwhm_px is None. Please provide a value for the FWHM in pixels for Euclid images.')
        
        rough_FWHM = fwhm_px * oversampling * 3.3333333 * (.03/ps_out)

    with fits.open(fname_psf) as hdu:
        header = hdu[1].header
        img = hdu[1].data[0][0][0]
        
        param = estimate_psf_fwhm(img, 0, 0, rough_FWHM, ps)
        
        temp_hdu = fits.PrimaryHDU(img, header=header)
        for key in param.keys():
            value = param[key]
            temp_hdu.header[key] = value
        temp_hdu.writeto(fname_out, overwrite=True)
        
    return


def get_varpsf_corner_positions(fname_mask):
    mask = fits.open(fname_mask)
    mask_img = mask[0].data
    mask.close()
    yshape, xshape = mask_img.shape
    left = (0, 0)
    bottom = (yshape-1, 0)
    right = (yshape-1, xshape-1)
    top = (0, xshape-1)

    corners = {'left': left, 'bottom': bottom, 'right': right, 'top': top}
    return corners


def get_varpsf_center_positions(corners):
    center1 = (corners['left'][0]+(corners['right'][0]-corners['left'][0])/12, corners['left'][1]+(corners['right'][1]-corners['left'][1])/12)
    center2 = (center1[0]+(corners['right'][0]-corners['left'][0])/6, center1[1])
    center3 = (center2[0]+(corners['right'][0]-corners['left'][0])/6, center1[1])
    center4 = (center3[0]+(corners['right'][0]-corners['left'][0])/6, center1[1])
    center5 = (center4[0]+(corners['right'][0]-corners['left'][0])/6, center1[1])
    center6 = (center5[0]+(corners['right'][0]-corners['left'][0])/6, center1[1])
    center7 = (center1[0], center1[1]+(corners['top'][1]-corners['bottom'][1])/6)
    center8 = (center2[0], center7[1])
    center9 = (center3[0], center7[1])
    center10 = (center4[0], center7[1])
    center11 = (center5[0], center7[1])
    center12 = (center6[0], center7[1])
    center13 = (center1[0], center7[1]+(corners['top'][1]-corners['bottom'][1])/6)
    center14 = (center2[0], center13[1])
    center15 = (center3[0], center13[1])
    center16 = (center4[0], center13[1])
    center17 = (center5[0], center13[1])
    center18 = (center6[0], center13[1])
    center19 = (center1[0], center13[1]+(corners['top'][1]-corners['bottom'][1])/6)
    center20 = (center2[0], center19[1])
    center21 = (center3[0], center19[1])
    center22 = (center4[0], center19[1])
    center23 = (center5[0], center19[1])
    center24 = (center6[0], center19[1])
    center25 = (center1[0], center19[1]+(corners['top'][1]-corners['bottom'][1])/6)
    center26 = (center2[0], center25[1])
    center27 = (center3[0], center25[1])
    center28 = (center4[0], center25[1])
    center29 = (center5[0], center25[1])
    center30 = (center6[0], center25[1])
    center31 = (center1[0], center25[1]+(corners['top'][1]-corners['bottom'][1])/6)
    center32 = (center2[0], center31[1])
    center33 = (center3[0], center31[1])
    center34 = (center4[0], center31[1])
    center35 = (center5[0], center31[1])
    center36 = (center6[0], center31[1])
    centers ={'center1': center1, 'center2': center2, 'center3': center3, 'center4': center4,
              'center5': center5, 'center6': center6, 'center7': center7, 'center8': center8,
              'center9': center9, 'center10': center10, 'center11': center11, 'center12': center12,
              'center13': center13, 'center14': center14, 'center15': center15, 'center16': center16,
              'center17': center17, 'center18': center18, 'center19': center19, 'center20': center20,
              'center21': center21, 'center22': center22, 'center23': center23, 'center24': center24, 
              'center25': center25, 'center26': center26, 'center27': center27, 'center28': center28,
              'center29': center29, 'center30': center30, 'center31': center31, 'center32': center32, 
              'center33': center33, 'center34': center34, 'center35': center35, 'center36': center36}
    results = []
    for key in centers.keys():
        results.append([int(key[6:]), centers[key][0], centers[key][1]])
    center_t = Table(rows=results, names=('center', 'x', 'y'))
    center_t['x'].format = '%.3f'
    center_t['y'].format = '%.3f'

    return center_t



def get_spatially_var_psf(x, y, fname_psf, fname_out, band, oversampling, win_size=None, ps_out=.03):
    """
    Obtain spatial variable PSF at the input position
    :param x: int, list
    x position in image with index from 0
    :param y: int, list
    y position in image with index from 0
    :param PSF_file: str
    input PSF file from PSFEx
    :param out_file: str
    file name of output PSF
    :param band: str
    filter name
    :return: None
    """
    if not os.path.exists(fname_out):
        PSF_parm_list = []
        ps = ps_out / oversampling  # pixelscale "/pixel
        if band in NIRCam_filter_FWHM['SW'].keys():
            rough_FWHM = NIRCam_filter_FWHM_new['SW'][band]*oversampling * (.03/ps_out)
        elif band in NIRCam_filter_FWHM['LW'].keys():
            rough_FWHM = NIRCam_filter_FWHM_new['LW'][band] * oversampling * 2 * (.03/ps_out)
        elif band in EUCLID.keys():
            if fwhm_px is None:
                logger.error('fwhm_px is None. Please provide a value for the FWHM in pixels for Euclid images.')
                raise ValueError('fwhm_px is None. Please provide a value for the FWHM in pixels for Euclid images.')

            rough_FWHM = fwhm_px * oversampling * 3.3333333 * (.03/ps_out)
            
            
        if win_size is None:
            if '_v2' in fname_out:
                constant_header = fits.getheader(fname_out.replace('_v2', '_c'))
            elif '_v' in fname_out:
                constant_header = fits.getheader(fname_out.replace('_v', '_c'))
            else:
                constant_header = fits.getheader(fname_out.replace('_obj', '_c'))
            win_size = constant_header['win_size']
        else:
            win_size = win_size
            

        with fits.open(fname_psf) as hdu:
            header = hdu[1].header
            POLZERO1 = header['POLZERO1']
            POLSCAL1 = header['POLSCAL1']
            POLZERO2 = header['POLZERO2']
            POLSCAL2 = header['POLSCAL2']
            POLDEG1 = hdu[1].header['POLDEG1']
            img = hdu[1].data[0][0]

            X = (x+1 - POLZERO1) / POLSCAL1
            Y = (y+1 - POLZERO2) / POLSCAL2

            if type(X) == float:
                if POLDEG1==3:
                    output_PSF = PSF_poly_3rd(X, Y, img[0], img[1], img[2], img[3], img[4],
                                              img[5], img[6], img[7], img[8], img[9])
                elif POLDEG1 == 2:
                    output_PSF = PSF_poly_2nd(X, Y,img[0],img[1],img[2], img[3],img[4],img[5])
                elif POLDEG1 == 1:
                    output_PSF =img[0] +img[1] * X +img[2] * Y
                param = estimate_psf_fwhm(output_PSF, x, y, rough_FWHM, ps, True, win_size)
                PSF_parm_list.append(param)
                temp_hdu = fits.PrimaryHDU(output_PSF)
                temp_hdu.header['POLDEG1'] = POLDEG1
                for key in param.keys():
                    value = param[key]
                    temp_hdu.header[key] = value
                temp_hdu.writeto(fname_out, overwrite=True)
            else:
                HDU = fits.HDUList()
                for each in range(len(X)):
                    X0 = X[each]
                    Y0 = Y[each]
                    x0 = x[each]
                    y0 = y[each]
                    if POLDEG1 == 3:
                        output_PSF = PSF_poly_3rd(X0, Y0, img[0], img[1], img[2], img[3], img[4],
                                                  img[5], img[6], img[7], img[8], img[9])
                    elif POLDEG1 == 2:
                        output_PSF = PSF_poly_2nd(X0, Y0, img[0], img[1], img[2], img[3], img[4], img[5])
                    elif POLDEG1 == 1:
                        output_PSF = img[0] + img[1] * X0 + img[2] * Y0
                    if each == 0:
                        temp_hdu = fits.PrimaryHDU(output_PSF)
                    else:
                        temp_hdu = fits.ImageHDU(output_PSF)

                    param = estimate_psf_fwhm(output_PSF, x0, y0, rough_FWHM, ps, True, win_size)
                    PSF_parm_list.append(param)
                    temp_hdu.header['POLDEG1'] = POLDEG1
                    for key in param.keys():
                        value = param[key]
                        temp_hdu.header[key] = value
                    HDU.append(temp_hdu)
                HDU.writeto(fname_out, overwrite=True)
            return PSF_parm_list



def check_psf_bkg(outdir, filtername, prefix):
    
    for psf_type in ['SWarp', 'photutils']:
        fname_psf = outdir + prefix + '_{}_PSF_2.fits'.format(psf_type)
        
        with fits.open(fname_psf, mode='update') as hdu:
            dat = hdu[0].data
            
            arr = np.concatenate([dat[5:10,:].flat, dat[:,5:10].flat, dat[-10:-5,:].flat, dat[:,-10:-5].flat])
            mean, median, std = sigma_clipped_stats(arr, sigma=2, maxiters=10, cenfunc=np.nanmedian, stdfunc=np.nanstd)
    
            dat -= median
            dat[:5,:] = 0
            dat[:,:5] = 0
            dat[-5:,:] = 0
            dat[:,-5:] = 0
            
            sumval = np.sum(dat)
            dat /= sumval
            
    return

#JWST NIRCam
def get_mag_zp(fname_data, fname_out, logger):
    logger.info('Getting MAG_ZP and EXPTIME from {}'.format(fname_data))

    with fits.open(fname_data) as hdu:
        header = hdu[0].header

        PIXAR_SR = header['PIXAR_SR']  # Nominal pixel area in steradians

        mag_zp = -2.5 * np.log10(PIXAR_SR * 1e6) + 8.9
        header['MAG_ZP'] = mag_zp

        if 'XPOSURE' in header.keys():
            EXPTIME = header['XPOSURE']  # /divide_factor
        elif 'TEXPTIME' in header.keys():
            EXPTIME = header['TEXPTIME']
        
        header['EXPTIME'] = EXPTIME
        hdu.writeto(fname_out, overwrite=True)

    logger.info('MAG_ZP: {:.3f}'.format(mag_zp))
    logger.info('EXPTIME: {:.3f}'.format(EXPTIME))

    return

############################################################################################################
############################################################################################################
# Select stars

def identify_stars(fname_sexcat, SNR_WIN, CLASS_STAR):
    print('Selecting stars from {}'.format(fname_sexcat))
    fname_out = fname_sexcat.replace('.cat', '_select.cat')
    
    with fits.open(fname_sexcat) as hdu:
        results = hdu[2].data
        
        mask = (results['SNR_WIN']>SNR_WIN)&(results['IMAFLAGS_ISO']==0)&(results['CLASS_STAR']>CLASS_STAR)&(results['ELONGATION']<1.5)&(results['FLAGS']<2)
        results = results[mask]

        hdu[2].data = results
        hdu.writeto(fname_out, overwrite=True)
        print('Number of stars: {}'.format(len(results)))
        print('Saved pre-selected stars to {}'.format(fname_out))
        
    return fname_out


def stack_stars(outdir, stardir, filtername, prefix):
    fnames_starlist = glob.glob(stardir + '*.fits')
    stardat = Table.read(stardir + prefix + '_stars.ipac', format='ascii.ipac')
    
    header_i = fits.getheader(fnames_starlist[0])
    wcs_i = wcs.WCS(header_i)
    
    if filtername in NIRCam_filter_FWHM['SW'].keys():
        wcs_i.wcs.crpix = (int(101), int(101))
        wcs_i.wcs.cdelt = (8.33333333333333e-06/3, 8.33333333333333e-06/3)
        wcs_i.NAXIS = (201, 201)
        shape = (201, 201)
    else:
        wcs_i.wcs.crpix = (int(134), int(134))
        wcs_i.wcs.cdelt = (8.33333333333333e-06/2, 8.33333333333333e-06/2)
        wcs_i.NAXIS = (267, 267)
        shape = (267, 267)
        
        
    array, footprint = reproject_and_coadd(fnames_starlist, wcs_i, shape_out=shape,
                                           reproject_function=reproject_exact,
                                           combine_function='sum')
    
    hdu = fits.PrimaryHDU(array/np.sum(array))
    hdu.writeto(outdir + prefix + '_stack_PSF.fits', overwrite=True)
    
    return