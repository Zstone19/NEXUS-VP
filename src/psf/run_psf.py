import os
import glob
import subprocess
import shutil
import logging

import numpy as np
from astropy.io import fits
import psf_methods as psfm
import psf_utils as psfu
import psf_euclid as psfe



def run_pipeline_psf(indir, maindir, default_dir, filtername_refgrid,
                     run_psfex=False, run_photutils=False, run_swarp=False, 
                     oversamp_vals=[1,2], nsig_sex=5., ncpu=1):    
    
    os.makedirs(maindir, exist_ok=True)
    _, logger_all = psfu.setup_logger('psf', maindir + 'psf.log')
    
    #Get files
    logger_all.info('Getting files...')
    fnames_in = glob.glob(indir + '*_data*')
    
    if len(fnames_in) == 0:
        logger_all.error('No data files found in {}'.format(indir))
        return
    else:
        logger_all.info('Found {} data files in {}'.format(len(fnames_in), indir))
        for f in fnames_in:
            logger_all.info('\t\t' + f)
        
        for f in fnames_in:
            if not os.path.exists(f.replace('_data', '_err')):
                logger_all.error('No error file found associated with {}'.format(f))
                raise FileNotFoundError('No error file found associated with {}'.format(f))    
        
            if ( not os.path.exists(f.replace('_data', '_wht')) ) and ( not os.path.exists(f.replace('_data', '_mask')) ):
                logger_all.error('No wht/mask file found associated with {}'.format(f))
                raise FileNotFoundError('No wht/mask file found associated with {}'.format(f))
    
    

    
    #Get filters
    logger_all.info('Getting filters...')
    filters_all = []
    for fname in fnames_in:
        filters_all.append(psfu.get_filter(fname))
        
    filters_unique = np.unique(filters_all)
    logger_all.info('Found {} filters:'.format(len(filters_unique)))
    for f in filters_unique:
        logger_all.info('\t\t' + f)

        
    #Make subdirectories
    logger_all.info('Making input, output, stars subdirectories ...')
    os.makedirs(maindir + '/input/', exist_ok=True)
    os.makedirs(maindir + '/output/', exist_ok=True)
    os.makedirs(maindir + '/stars/', exist_ok=True)
    os.makedirs(maindir + '/paramfiles/', exist_ok=True)
    
    #Make filter-specific out/star subdirs
    logger_all.info('Making filter-specific output, stars subdirectories ...')
    for filt in filters_unique:
        os.makedirs(maindir + 'output/' + filt + '/', exist_ok=True)
        os.makedirs(maindir + 'stars/' + filt + '/', exist_ok=True)
    
    
    logger_all.info('Making logs...')
    loggers = []
    prefixes = []
    for i in range(len(filters_all)):
        prefix = os.path.basename(fnames_in[i]).split('.')[0].split('_')[:-1]
        prefix = '_'.join(prefix)
        prefixes.append(prefix)
        
        logfile = maindir + 'output/' + filters_all[i] + '/' + prefix + '.log'
        _, logger = psfu.setup_logger('psf.{}'.format(prefix), logfile)
        loggers.append(logger)
    

    #Copy input files
    logger_all.info('Copying input files ...')
    for hdlr in logger_all.handlers[:]:  # remove all old handlers
        logger_all.removeHandler(hdlr)
    
    for i in range(len(filters_all)):
        data_exist = os.path.exists(  maindir + '/input/' + prefixes[i] + '_data.fits' )
        err_exist = os.path.exists(  maindir + '/input/' + prefixes[i] + '_err.fits' )
        mask_exist = os.path.exists(  maindir + '/input/' + prefixes[i] + '_mask.fits' )
        
        if data_exist and err_exist and mask_exist:
            loggers[i].info('All input files related to {} already exist in {}'.format(fnames_in[i], maindir + '/input/'))
            loggers[i].info('Skipping copying input files')
            continue
        
        
        shutil.copy(fnames_in[i], maindir + '/input/')
        loggers[i].info('Copied {} to {}'.format(fnames_in[i], maindir + 'input/'))
        
        shutil.copy(fnames_in[i].replace('_data', '_err'), maindir + 'input/')
        loggers[i].info('Copied {} to {}'.format(fnames_in[i].replace('_data', '_err'), maindir + 'input/'))
        
        #Make mask file if it doesn't exist
        mask_fname_in = fnames_in[i].replace('_data', '_mask')
                
        if not os.path.exists(mask_fname_in):
            loggers[i].info("Mask file doesn't exist in {}".format(indir))
            mask_fname_out = maindir + 'input/' + os.path.basename(mask_fname_in)    
            
            if filters_all[i][0].upper() == 'F':
                loggers[i].info("Creating mask file at {} using {}".format(mask_fname_out, fnames_in[i].replace('_data', '_wht')))
                
                with fits.open(fnames_in[i].replace('_data', '_wht'), mammap=True) as hdul:
                    wht = hdul[0].data
                    mask = np.zeros_like(wht)
                    
                    filter_mask = (wht == 0)
                    mask[filter_mask] = 1
                    mask = mask.astype(np.int16)
                    
                    hdu = fits.PrimaryHDU(mask)
                    hdu.writeto(mask_fname_out, overwrite=True)
                    loggers[i].info("Mask file created at {}".format(mask_fname_out))
                    
            elif 'NIR' in filters_all[i]:
                loggers[i].info("Creating mask file at {} using {}".format(mask_fname_out, fnames_in[i].replace('_data', '_flag')))
        
                im_mask = psfe.get_mask_image(fnames_in[i].replace('_data', '_flag'))
                hdu = fits.PrimaryHDU(im_mask)
                hdu.writeto(mask_fname_out, overwrite=True)
                loggers[i].info("Mask file created at {}".format(mask_fname_out))
        
    
        else:
            shutil.copy(mask_fname_in, maindir + 'input/')
            loggers[i].info('Copied {} to {}'.format(mask_fname_in, maindir + 'input/'))



        #Make sure all files are uncompressed
        fname_data = maindir + 'input/' + os.path.basename(fnames_in[i])
        if '.gz' in fname_data:
            loggers[i].info('Data file compressed, uncompressing...')
            out = subprocess.run(['gunzip', fname_data], check=True, capture_output=True).stdout
            loggers[i].info('Uncompressed data file'.format(fname_data))
            
            fname_data = fname_data.replace('.gz', '')
            

            
        fname_wht = maindir + 'input/' + os.path.basename(fnames_in[i].replace('_data', '_err'))
        if '.gz' in fname_wht:
            loggers[i].info('Weight file compressed, uncompressing...')
            out = subprocess.run(['gunzip', fname_wht], check=True, capture_output=True).stdout
            loggers[i].info('Uncompressed weight file'.format(fname_wht))
            
        fname_mask = maindir + 'input/' + os.path.basename(fnames_in[i].replace('_data', '_mask'))
        if '.gz' in fname_mask:
            loggers[i].info('Mask file compressed, uncompressing...')
            out = subprocess.run(['gunzip', fname_mask], check=True, capture_output=True).stdout
            loggers[i].info('Uncompressed mask file'.format(fname_mask))
            
            
            
        #Make sure data file has MAG_ZP
        with fits.open(fname_data) as hdul:
            if 'MAG_ZP' not in hdul[0].header:
                loggers[i].info('MAG_ZP not in header of data file')
                loggers[i].info('Adding MAG_ZP to header of data file')
                
                if filters_all[i][0].upper() == 'F':
                    psfu.get_mag_zp(fname_data, fname_data, loggers[i])
                elif 'NIR' in filters_all[i]:

                    with fits.open(fname_data) as hdu:
                        header = hdu[0].header

                        mag_zp = header['MAGZERO']
                        header['MAG_ZP'] = mag_zp
                        
                        hdu.header = header
                        hdu.writeto(fname_data, overwrite=True)

                    loggers[i].info('MAG_ZP: {:.3f}'.format(mag_zp))
                    
                loggers[i].info('MAG_ZP added to header of data file')
                
        #If Euclid data, make sure psf file exists
        if 'NIR' in filters_all[i]:
            
            if not os.path.exists(fnames_in[i].replace('_data.fits', '.psf')):
                loggers[i].error('No PSF file found for {}'.format(fnames_in[i]))
                raise FileNotFoundError('No PSF file found for {}'.format(fnames_in[i]))

            shutil.copy(fnames_in[i].replace('_data.fits', '.psf'), maindir + 'input/')
            loggers[i].info('Copied {} to {}'.format(fnames_in[i].replace('_data.fits', '.psf'), maindir + 'input/'))

    ############################################################################################################
    # Run SExtractor and Identify Stars
    ############################################################################################################
    
    _, logger_all = psfu.setup_logger('psf', maindir + 'psf.log')
    
    #Get SExtractor default config file
    shutil.copy(default_dir + 'default.sex', maindir + 'paramfiles/')    
    logger_all.info('Copied {} to {}'.format(default_dir + 'default.sex', maindir + 'paramfiles/default.sex'))
    
    shutil.copy(default_dir + 'default_LW.param', maindir + 'paramfiles/')
    logger_all.info('Copied {} to {}'.format(default_dir + 'default_LW.param', maindir + 'paramfiles/default_LW.param'))
    
    shutil.copy(default_dir + 'default_SW.param', maindir + 'paramfiles/')
    logger_all.info('Copied {} to {}'.format(default_dir + 'default_SW.param', maindir + 'paramfiles/default_SW.param'))
    
    shutil.copy(default_dir + 'default_euclid.param', maindir + 'paramfiles/')
    logger_all.info('Copied {} to {}'.format(default_dir + 'default_euclid.param', maindir + 'paramfiles/default_euclid.param'))
    
    shutil.copy(default_dir + 'gauss_4.0_7x7.conv', maindir + 'paramfiles/')
    logger_all.info('Copied {} to {}'.format(default_dir + 'gauss_4.0_7x7.conv', maindir + 'paramfiles/gauss_4.0_7x7.conv'))
        
    shutil.copy(default_dir + 'default.nnw', maindir + 'paramfiles/')
    logger_all.info('Copied {} to {}'.format(default_dir + 'default.nnw', maindir + 'paramfiles/default.nnw'))
        
    #Run SExtractor for all
    logger_all.info('Running SExtractor for all...')
    for hdlr in logger_all.handlers[:]:  # remove all old handlers
        logger_all.removeHandler(hdlr)
    
    _ = psfm.run_sextractor_all(maindir, filtername_refgrid, nsig=nsig_sex, ncpu=ncpu)
    
    _, logger_all = psfu.setup_logger('psf', maindir + 'psf.log')
    logger_all.info('SExtractor runs complete')
    for hdlr in logger_all.handlers[:]:  # remove all old handlers
        logger_all.removeHandler(hdlr)
    
    ############################################################################################################
    # Get Point Source Catalog (Master and Band-Specific)
    ############################################################################################################
    
    _, logger_all = psfu.setup_logger('psf', maindir + 'psf.log')
    logger_all.info('Obtaining point source catalogs...')
    for hdlr in logger_all.handlers[:]:  # remove all old handlers
        logger_all.removeHandler(hdlr)
    
    _ = psfm.get_master_pointsource_catalog(maindir)
    
    _, logger_all = psfu.setup_logger('psf', maindir + 'psf.log')
    logger_all.info('Point source catalogs obtained')
    for hdlr in logger_all.handlers[:]:  # remove all old handlers
        logger_all.removeHandler(hdlr)
    
    ############################################################################################################
    # Run PSFEx
    ############################################################################################################
    
    if run_psfex:
        _, logger_all = psfu.setup_logger('psf', maindir + 'psf.log')
        
        #Get PSFEx default config file
        shutil.copy(default_dir + 'default_c.psfex', maindir + 'paramfiles/default_c.psfex')    
        logger_all.info('Copied {} to {}'.format(default_dir + 'default_c.psfex', maindir + 'paramfiles/default_c.psfex'))
        
        shutil.copy(default_dir + 'default_v.psfex', maindir + 'paramfiles/default_v.psfex')    
        logger_all.info('Copied {} to {}'.format(default_dir + 'default_v.psfex', maindir + 'paramfiles/default_v.psfex'))
        
        shutil.copy(default_dir + 'default_v2.psfex', maindir + 'paramfiles/default_v2.psfex')    
        logger_all.info('Copied {} to {}'.format(default_dir + 'default_v2.psfex', maindir + 'paramfiles/default_v2.psfex'))
        
        #Run PSFEx for all
        logger_all.info('Running PSFEx for all...')
        for hdlr in logger_all.handlers[:]:  # remove all old handlers
            logger_all.removeHandler(hdlr)
        
        _ = psfm.run_psfex_all(maindir, filtername_refgrid, oversamp_vals, ncpu=ncpu)
        
        _, logger_all = psfu.setup_logger('psf', maindir + 'psf.log')
        logger_all.info('PSFEx runs complete')




        #Fit PSF and update PSF file headers w/ fit params
        logger_all.info('Fitting PSFs and updating PSF file headers...')
        for hdlr in logger_all.handlers[:]:
            logger_all.removeHandler(hdlr)
            
        if filtername_refgrid in psfm.NIRCam_filter_FWHM['SW']:
            ps_out = .03
        elif filtername_refgrid in psfm.NIRCam_filter_FWHM['LW']:
            ps_out = .03
        elif filtername_refgrid in psfm.EUCLID:
            ps_out = .1            
        
        for i in range(len(filters_all)):
            outdir = maindir + '/output/' + filters_all[i] + '/'
            
            if filters_all[i][0].upper() == 'F':
                fwhm_px = None
            elif 'NIR' in filters_all[i]:
                fname_psf = fnames_in[i].replace('_data.fits', '.psf')
                fwhm_px = fits.open(fname_psf)[0].header['FWHM']
                        
            loggers[i].info('Fitting PSF and updating PSF file headers for PSFEx')
            _ = psfu.update_psfex_psf_param(outdir, filters_all[i], prefixes[i], oversamp_vals, fwhm_px, ps_out)
            loggers[i].info('PSF fitting and header updating complete for PSFEx')
            
        _, logger_all = psfu.setup_logger('psf', maindir + 'psf.log')
        logger_all.info('PSF fitting and header updating complete')
        for hdlr in logger_all.handlers[:]:
            logger_all.removeHandler(hdlr)
    
    ############################################################################################################
    # Run photutils
    ############################################################################################################

    if run_photutils:
    
        _ = psfm.run_photutils_all(maindir, [2], True, False, 2, ncpu=ncpu)
        
        _ = psfm.run_photutils_all(maindir, [1], False, False, 1, ncpu=ncpu)
        _ = psfm.run_photutils_all(maindir, [2], True, False, 1, ncpu=ncpu)
        _ = psfm.run_photutils_all(maindir, [1], False, True, 1, ncpu=ncpu)
        _ = psfm.run_photutils_all(maindir, [2], False, True, 1, ncpu=ncpu)
    
    ############################################################################################################
    # Stack stars
    ############################################################################################################
    
    if run_photutils:
            
        outdirs = []
        stardirs = []
        prefixes = []
        for i in range(len(filters_all)):
            list_of_str = os.path.basename(fnames[i]).split('.')[0].split('_')[:-1]
            prefixes.append('_'.join(list_of_str))   
            
            outdirs.append(maindir + '/output/' + filt + '/')
            stardirs.append(maindir + '/stars/' + filt + '/')
        
        for i in range(len(filters_all)):
            _ = psfu.stack_stars(outdirs[i], stardirs[i], filters_all[i], prefixes[i])
        
        
    ############################################################################################################
    # Run SWarp
    ############################################################################################################
    
    if run_photutils and run_swarp:
        _ = psfm.run_swarp_all(maindir, [1], False, ncpu=ncpu)
        _ = psfm.run_swarp_all(maindir, [1], True, ncpu=ncpu)
        _ = psfm.run_swarp_all(maindir, [2], False, ncpu=ncpu)
        _ = psfm.run_swarp_all(maindir, [2], True, ncpu=ncpu)
    
    ############################################################################################################
    # Check PSF background
    ############################################################################################################
 
    if run_photutils and run_swarp:    
        for i in range(len(filters_all)):
            psfu.check_psf_bkg(outdirs[i], filters_all[i], prefixes[i])
        
    ############################################################################################################
    
    return
    