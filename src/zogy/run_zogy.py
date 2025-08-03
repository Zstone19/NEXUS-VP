import os
import shutil
import subprocess
import logging
from functools import partial

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

import toml
from mpire import WorkerPool
from tqdm import tqdm

import zogy_utils as zogyu
import zogy_cutout as zogyc
from zogy_preprocess import cross_conv, subtract_sky



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


def setup_logger_empty(name, log_file=None, level=logging.INFO):
    
    formatter = logging.Formatter('%(message)s')
    
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
        
        
        
        
    
def zogy_loop(i, ncutout, 
              cutout_name, maindir, ref_name, sci_name, logfile, 
              conv_ref, conv_sci, skysub, use_var, match_gain, n_stamps, max_iter):
        
    maindir_i = maindir + 'output_{}/'.format(cutout_name)
    logfile_i = maindir_i + 'zogy.log' 
    
    _, logger = setup_logger('NVP.zogy.zogy', logfile)
    logger.info('Running PyZOGY for cutout {}/{}: {}'.format(i+1, ncutout, cutout_name))
    reset_logger(logger)
    
    
    
    _, logger_i = setup_logger('NVP.zogy.zogy', logfile_i)

    zogyu.run_zogy(maindir_i, ref_name, sci_name, logger_i,
                    conv_ref, conv_sci, skysub, use_var, match_gain, n_stamps, max_iter)
    
    reset_logger(logger_i)
    
    return




def run_zogy_pipeline(config_filename):

    #Read TOML file
    config = toml.load(config_filename)
    general = config['inputs']
    preprocess = config['preprocess']
    cutout = config['cutout']
    zogy = config['zogy']
    
    #Cutout params
    cutout_together = (not cutout['mask_separate'])  #See if we want to do all pre-processing separate
    cutout_run = cutout['run']                       #Or together and then cutout after
    cutout_shape = cutout['shape']
    cutout_npx = cutout['npx']
    cutout_fname = cutout['filename']    #If cutout_fname is None, will split the image into cutouts of 
                                        # (cutout_npx x cutout_npx) evenly across the image
    cutout_dither = cutout['dither']
    subset = cutout['subset']
    npx_min = cutout['npx_min']         #Minimum shape of cutout (in px)
    nz_thresh = cutout['nz_thresh']     #Threshold (as a percentage of all pixels in cutout) need to be non-zero
    
    if cutout_fname == '':
        cutout_fname = None
    if cutout_dither == '':
        cutout_dither = None
    if subset == '':
        subset = None
    else:
        subset = np.array(subset, dtype=int)
    
    #Get general parameters
    maindir = general['maindir']
    paramdir = general['paramdir']
    ref_name = general['ref_name']
    sci_name = general['sci_name']
    filtername_ref = general['filtername_ref']
    filtername_sci = general['filtername_sci']
    filtername_grid = general['filtername_grid']
    logfile = general['logfile']
    ncpu = general['ncpu']
    
    if logfile == '':
        logfile = None
    
    if (logfile is not None) and (os.path.basename(logfile) == logfile):
        logfile = maindir + logfile
        
        
    #Get preprocessing params
    skysub = preprocess['skysub']
    mask_type = preprocess['mask_type']
    
    conv_sci = preprocess['conv_sci']
    conv_ref = preprocess['conv_ref']
    crossconv = (conv_ref or conv_sci)
    
    ##############################################################
    # Set up logger
    
    _, logger = setup_logger('NVP.zogy.startup', logfile)
    
    txt = """Starting PyZOGY
--------------------------------
Main directory: {}
REF name: {}
SCI name: {}
REF filter: {}
SCI filter: {}
Number of CPUs: {}
--------------------------------
Subtract Sky: {}
Sky Mask Type: {}
Convolve REF: {}
Convolve SCI: {}
--------------------------------""".format(maindir, ref_name, sci_name, filtername_ref, filtername_sci, ncpu,
                                           skysub, mask_type, conv_ref, conv_sci)
    logger.info(txt)
    
    ##############################################################
    # Make directories
    
    logger.info('Making directories')
    os.makedirs(maindir, exist_ok=True)
    os.makedirs(maindir + 'input/', exist_ok=True)
    os.makedirs(maindir + 'noise/', exist_ok=True)
    os.makedirs(maindir + 'psf/', exist_ok=True)
    
    if not ( cutout_run and (not cutout_together) ):
        os.makedirs(maindir + 'mask/', exist_ok=True)
        os.makedirs(maindir + 'output/', exist_ok=True)
    
    ##############################################################
    # Make sure files exist
    logger.info('Checking for input files')
    
    #REF
    if not os.path.exists(maindir + 'input/{}.fits'.format(ref_name)):
        logger.error('\t Input REF file does not exist')
        raise FileNotFoundError('Input REF file does not exist')
    
    #SCI
    if not os.path.exists(maindir + 'input/{}.fits'.format(sci_name)):
        logger.error('\t Input SCI file does not exist')
        raise FileNotFoundError('Input SCI file does not exist')
    
    #REF maskin
    if not os.path.exists(maindir + 'input/{}.maskin.fits'.format(ref_name)):
        logger.error('\t Input REF mask file does not exist')
        raise FileNotFoundError('Input REF mask file does not exist')
    
    #SCI maskin
    if not os.path.exists(maindir + 'input/{}.maskin.fits'.format(sci_name)):
        logger.error('\t Input SCI mask file does not exist')
        raise FileNotFoundError('Input SCI mask file does not exist')
    
    #REF PSF
    if not os.path.exists(maindir + 'psf/{}.psf.fits'.format(ref_name)):
        logger.error('\t REF PSF file does not exist')
        raise FileNotFoundError('Input REF PSF file does not exist')
    
    #SCI PSF
    if not os.path.exists(maindir + 'psf/{}.psf.fits'.format(sci_name)):
        logger.error('\t SCI PSF file does not exist')
        raise FileNotFoundError('Input SCI PSF file does not exist')
    
    #REF noise
    if not os.path.exists(maindir + 'noise/{}.noise.fits'.format(ref_name)):
        logger.error('\t REF noise file does not exist')
        raise FileNotFoundError('Input REF noise file does not exist')
    
    #SCI noise
    if not os.path.exists(maindir + 'noise/{}.noise.fits'.format(sci_name)):
        logger.error('\t SCI noise file does not exist')
        raise FileNotFoundError('Input SCI noise file does not exist')

    reset_logger(logger)
    
    ################################################################
    #Get cutouts

    if cutout_run and (not cutout_together):
        _, logger = setup_logger('NVP.zogy.cutout', logfile)
        logger.info('Saving image cutouts')
        
        if cutout_fname is not None:        
            if not os.path.exists(cutout_fname):
                logger.error('\t Cutout file does not exist')
                raise FileNotFoundError('Cutout file does not exist')
            
            logger.info('\t Cutout shape: {}x{}'.format(cutout_shape[0], cutout_shape[1]))
        
            cutout_names, ras, decs = np.loadtxt(cutout_fname, dtype=object, unpack=True, usecols=[0,1,2])
            cutout_names = np.atleast_1d(cutout_names)
            ras = np.atleast_1d(ras)
            decs = np.atleast_1d(decs)

            cutout_names = cutout_names.astype(str)
            ras = ras.astype(float)
            decs = decs.astype(float)
            ncutout = len(cutout_names)
            
            logger.info('\t Found {} cutouts'.format(len(cutout_names)))
        
            if subset is not None:
                logger.info('\t Only using a subset of {} cutouts: {}'.format(len(subset)) )
                
                cutout_names = cutout_names[subset]
                ras = ras[subset]
                decs = decs[subset]
                ncutout = len(cutout_names)
        
            dirs_exist = np.zeros(ncutout, dtype=bool)
            for i in range(ncutout):
                cutout_name = cutout_names[i]
                maindir_i = maindir + 'output_{}/'.format(cutout_name)
                
                if os.path.exists(maindir_i):
                    dirs_exist[i] = True        
        
            if np.sum(dirs_exist) > 0:   
                logger.error('NOT IMPLEMENTED YET')
                raise NotImplementedError    
                #func = partial(sfftc.make_cutout_subdir_separate, maindir=maindir, ref_name=ref_name, sci_name=sci_name, cutout_shape=cutout_shape)
                
                logger.info('Splitting images into cutouts')
                pool = WorkerPool(n_jobs=ncpu)
                pool.map(func, zip(cutout_names, ras, decs), iterable_len=ncutout,
                        progress_bar=True, progress_bar_style='rich')     
                
                logger.info('Finished splitting images')
                
            else:
                logger.info('Cutout directories already exist')
                logger.info('Skipping cutout creation')
                
            reset_logger(logger)
            
        else:
            if cutout_npx is None:
                logger.error('\t Cutout size not specified')
                raise ValueError('Cutout size not specified')
            
            cutout_npx = int(cutout_npx)
            cutout_fname = maindir + 'cutout_info.txt'

            
            if os.path.exists(cutout_fname):
                logger.info('\t Cutout file already exists')
                logger.info('\t Skipping cutout creation')
            else:                
                logger.info('\t Splitting full image into {}x{} segments'.format(cutout_npx, cutout_npx))
                zogyc.make_split_subdir_separate(cutout_npx, maindir, ref_name, sci_name, dither=cutout_dither, npx_min=npx_min, nz_thresh=nz_thresh)      
                logger.info('\t Cutout file: {}'.format(cutout_fname))
     
            _, _, _, _, _, _, cutout_names = np.loadtxt(cutout_fname, dtype=object, unpack=True, skiprows=1)                        
            cutout_names = cutout_names.astype(str)
            ncutout = len(cutout_names)
            
            logger.info('\t {} cutouts'.format(ncutout))
            
            if subset is not None:
                logger.info('\t Only using a subset of {} cutouts'.format(len(subset)) )
                
                for i in range(ncutout):
                    if i in subset:
                        continue
                    else:
                        dir_i = maindir + 'output_{}/'.format(cutout_names[i])                    

                        if os.path.exists(dir_i):
                            shutil.rmtree(dir_i)
                
                cutout_names = cutout_names[subset]
                ncutout = len(cutout_names)            
            
            logger.info('Finished splitting images')        
        
        
        
        if skysub:
            _, logger = setup_logger('NVP.zogy.skysub', logfile)
            logger.info('Subtracting sky from cutouts')
            reset_logger(logger)
            
            for i in range(ncutout):
                maindir_i = maindir + 'output_{}/'.format(cutout_names[i])
                logfile_i = maindir_i + 'zogy.log'
                
                _, logger = setup_logger('NVP.zogy.skysub', logfile)
                logger.info('Subtracting sky from cutout {}/{}: {}'.format(i+1, ncutout, cutout_names[i]))
                reset_logger(logger)
                
                _, logger_i = setup_logger('NVP.zogy.skysub', logfile_i)
                subtract_sky(maindir_i, ref_name, sci_name, logger_i, mask_type=mask_type, ncpu=ncpu)
                reset_logger(logger_i)
            
            
    elif skysub:
        _, logger = setup_logger('NVP.zogy.skysub', logfile)
        logger.info('Subtracting sky from images')
        subtract_sky(maindir, ref_name, sci_name, logger, mask_type=mask_type, ncpu=ncpu)
        reset_logger(logger)
        
        
    
    ##############################################################
    #Cross-convolve images
    
    if crossconv:
    
        if cutout_run and (not cutout_together):
            
            for i in range(ncutout):
                maindir_i = maindir + 'output_{}/'.format(cutout_names[i])
                logfile_i = maindir_i + 'zogy.log'
                
                _, logger = setup_logger('NVP.zogy.crossconv', logfile)
                logger.info('Cross-convolving images for cutout {}/{}: {}'.format(i+1, ncutout, cutout_names[i]))
                reset_logger(logger)
                
                _, logger_i = setup_logger('NVP.zogy.crossconv', logfile_i)
                logger_i.info('Cross-convolving images')
                cross_conv(maindir_i, ref_name, sci_name, conv_ref, conv_sci, skysub, logger_i)
                logger_i.info('Finished cross-convolving images')
                reset_logger(logger_i)    

        else:
            _, logger = setup_logger('NVP.zogy.crossconv', logfile)
            
            logger.info('Cross-convolving images')
            cross_conv(maindir, ref_name, sci_name, conv_ref, conv_sci, skysub, logger)
            logger.info('Finished cross-convolving images')
            
            reset_logger(logger)
    
    ##############################################################
    # Get image cutouts
    
    if cutout_run and cutout_together:    
        _, logger = setup_logger('NVP.zogy.cutout', logfile)
        logger.info('Saving image cutouts')
        logger.info('\t Cutout shape: {}x{}'.format(cutout_shape[0], cutout_shape[1]))
        
        
        if cutout_fname is not None:
            if not os.path.exists(cutout_fname):
                logger.error('\t Cutout file does not exist')
                raise FileNotFoundError('Cutout file does not exist')
            
            cutout_names, ras, decs = np.loadtxt(cutout_fname, dtype=object, unpack=True, usecols=[0,1,2])
            cutout_names = np.atleast_1d(cutout_names)
            ras = np.atleast_1d(ras)
            decs = np.atleast_1d(decs)

            cutout_names = cutout_names.astype(str)
            ras = ras.astype(float)
            decs = decs.astype(float)
            ncutout = len(cutout_names)
            
            logger.info('\t Found {} cutouts'.format(len(cutout_names)))
            
            if subset is not None:
                logger.info('\t Only using a subset of {} cutouts: {}'.format(len(subset)) )
                
                cutout_names = cutout_names[subset]
                ras = ras[subset]
                decs = decs[subset]
                ncutout = len(cutout_names)
            
    
            logger.error('NOT IMPLEMENTED YET')
            raise NotImplementedError
            # func = partial(make_cutout_subdir_together, maindir=maindir, ref_name=ref_name, sci_name=sci_name, cutout_shape=cutout_shape, skysub=skysub, crossconv=crossconv)
            
            logger.info('Splitting images into cutouts')
            pool = WorkerPool(n_jobs=ncpu)
            pool.map(func, zip(cutout_names, ras, decs), iterable_len=ncutout,
                    progress_bar=True, progress_bar_style='rich')  
            pool.join()
            
        else:
            if cutout_npx is None:
                logger.error('\t Cutout size not specified')
                raise ValueError('Cutout size not specified')
            
            cutout_npx = int(cutout_npx)

            logger.error('NOT IMPLEMENTED YET')
            raise NotImplementedError

            logger.info('\t Splitting full image into {}x{} segments'.format(cutout_npx, cutout_npx))
            # sfftc.make_split_subdir_together(cutout_npx, maindir, ref_name, sci_name, skysub, conv_ref, conv_sci, dither=cutout_dither)           
            
            cutout_fname = maindir + 'cutout_info.txt'
            _, _, _, _,  _, _, cutout_names = np.loadtxt(cutout_fname, dtype=object, unpack=True, skiprows=1)
            cutout_names = cutout_names.astype(str)
            ncutout = len(cutout_names)
            
            logger.info('\t Created {} cutouts'.format(ncutout))
            logger.info('\t Cutout file: {}'.format(cutout_fname))
            
            
            if subset is not None:
                logger.info('\t Only using a subset of {} cutouts: {}'.format(len(subset)) )
                
                cutout_names = cutout_names[subset]
                ncutout = len(cutout_names)

        logger.info('Finished splitting images')
        reset_logger(logger)
    
    ##############################################################
    #Run ZOGY
    
    run = zogy['general']['run']
    
    if run:        
        _, logger = setup_logger('NVP.zogy.zogy', logfile)
                
        max_iter = zogy['general']['max_iter']
        use_var = zogy['general']['use_var']
        n_stamps = zogy['general']['n_stamps']
        match_gain = zogy['general']['match_gain']
        txt = """Running PyZOGY
--------------------------------
Maximum Number of Iterations: {}
Use Variance Image: {}
Match Gain: {}
Number of Stamps: {}
Number of CPUs: {}
--------------------------------""".format(max_iter, use_var, match_gain, n_stamps, ncpu)
        logger.info(txt)
        
        
        logger.info('Starting PyZOGY run')
        reset_logger(logger)
    
        
        if cutout_run:
            
            inputs = []
            for i in range(ncutout):
                inputs.append((
                    i, ncutout, cutout_names[i], maindir, ref_name, sci_name, logfile, 
                    conv_ref, conv_sci, skysub, use_var, match_gain, n_stamps, max_iter
                ))
  
            pool = WorkerPool(n_jobs=ncpu)
            pool.map(zogy_loop, iter(inputs), iterable_len=ncutout,
                    progress_bar=True, progress_bar_style='rich')
            pool.stop_and_join()
        
        else:
            _, logger = setup_logger('NVP.zogy.zogy', logfile)
            logger.info('Running PyZOGY')
            reset_logger(logger)

            _, logger = setup_logger('NVP.zogy.zogy', logfile)

            zogyu.run_zogy(maindir, ref_name, sci_name, logger,
                                  conv_ref, conv_sci, skysub, use_var, match_gain, n_stamps, max_iter)
            
            reset_logger(logger)
        
        _, logger = setup_logger('NVP.zogy.zogy', logfile)
        logger.info('Finished running PyZOGY')
        reset_logger(logger)


    
    _, logger = setup_logger('NVP.zogy.finish', logfile)
    logger.info('Finished PyZOGY')
    reset_logger(logger)
    
    return
    