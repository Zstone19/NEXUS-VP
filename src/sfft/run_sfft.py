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
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve_fft

import toml
from mpire import WorkerPool
from tqdm import tqdm

from sfft_preprocess import cross_conv, subtract_sky
import sfft_mask as sfftm
import sfft_utils as sfftu
import sfft_cutout as sfftc



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




def check_num_obj_maskedcc(maindir, ref_name, sci_name, logger, conv_ref=False, conv_sci=False):
    
    if conv_ref:
        fname_ref_maskedcc = maindir + 'output/{}.crossconvd.masked.fits'.format(ref_name)
    else:
        fname_ref_maskedcc = maindir + 'output/{}.masked.fits'.format(ref_name)
        
    if conv_sci:
        fname_sci_maskedcc = maindir + 'output/{}.crossconvd.masked.fits'.format(sci_name)
    else:
        fname_sci_maskedcc = maindir + 'output/{}.masked.fits'.format(sci_name)
    
    fname_ref_segmap = maindir + 'mask/{}_sexseg.fits'.format(ref_name)
    fname_sci_segmap = maindir + 'mask/{}_sexseg.fits'.format(sci_name)
    
    if os.path.exists(fname_ref_maskedcc) and os.path.exists(fname_sci_maskedcc):
        im_ref_mcc = fits.open(fname_ref_maskedcc)[0].data
        im_sci_mcc = fits.open(fname_sci_maskedcc)[0].data
        
        im_ref_segmap = fits.open(fname_ref_segmap)[0].data
        
        mask = (im_ref_mcc > 0) & (im_sci_mcc > 0)
        obj_unique = np.unique(im_ref_segmap[mask])
        obj_unique = obj_unique[obj_unique > 0]
        
        nobj = len(obj_unique)
        logger.info('\t Masked image contains {} objects'.format(nobj))
        
        if nobj < 5:
            logger.warning('\t Masked image contains less than 5 objects')
            logger.warning('\t SFFT may fail due to imsufficient sampling')
            logger.warning('\t You may want to extend the size of the image to include more stars')

    return
        
        
        
        
    
    





def run_sfft(config_filename):

    #Read TOML file
    config = toml.load(config_filename)
    general = config['inputs']
    preprocess = config['preprocess']
    cutout = config['cutout']
    sfft = config['sfft']
    
    #Cutout params
    cutout_together = (not cutout['mask_separate'])  #See if we want to do all pre-processing separate
    cutout_run = cutout['run']                       #Or together and then cutout after
    cutout_shape = cutout['shape']
    cutout_npx = cutout['npx']
    cutout_fname = cutout['filename']    #If cutout_fname is None, will split the image into cutouts of 
                                        # (cutout_npx x cutout_npx) evenly across the image
    cutout_dither = cutout['dither']
    subset = cutout['subset']
    pp_separate = cutout['preprocess_separate']   #If True, will do all pre-processing separately for each cutout
    
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
    use_gpu = general['use_gpu']
    
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
    
    sat_ref = preprocess['saturation_ref']
    sat_sci = preprocess['saturation_sci']
    
    ##############################################################
    # Set up logger
    
    _, logger = setup_logger('NVP.sfft.startup', logfile)
    
    txt = """Starting SFFT
--------------------------------
Main directory: {}
REF name: {}
SCI name: {}
REF filter: {}
SCI filter: {}
Number of CPUs: {}
Use GPU: {}
--------------------------------
Subtract Sky: {}
Sky Mask Type: {}
Convolve REF: {}
Convolve SCI: {}
Saturation REF: {:.2e}
Saturation SCI: {:.2e}
--------------------------------""".format(maindir, ref_name, sci_name, filtername_ref, filtername_sci, ncpu, use_gpu,
                                           skysub, mask_type, conv_ref, conv_sci, sat_ref, sat_sci)
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
    
    ##############################################################
    # Make sure paramfiles exist
    
    logger.info('Checking for parameter files')
    
    if not os.path.exists(paramdir + 'default_LW.param'):
        logger.error('\t SExtractor LW parameter file does not exist')
        raise FileNotFoundError('SExtractor LW parameter file does not exist')
    
    if not os.path.exists(paramdir + 'default_SW.param'):
        logger.error('\t SExtractor SW parameter file does not exist')
        raise FileNotFoundError('SExtractor SW parameter file does not exist')
    
    if not os.path.exists(paramdir + 'default.sex'):
        logger.error('\t SExtractor configuration file does not exist')
        raise FileNotFoundError('SExtractor configuration file does not exist')
    
    if not os.path.exists(paramdir + 'default.nnw'):
        logger.error('\t SExtractor neural network file does not exist')
        raise FileNotFoundError('SExtractor neural network file does not exist')
    
    if not os.path.exists(paramdir + 'gauss_4.0_7x7.conv'):
        logger.error('\t SExtractor convolution file does not exist')
        raise FileNotFoundError('SExtractor convolution file does not exist')

    reset_logger(logger)
    
    ################################################################
    #Pre-process / get cutouts
    
    npx_boundary = sfft['general']['GKerHW']
    
    if skysub:
        if pp_separate:
            pass
        else:
            _, logger = setup_logger('NVP.sfft.skysub', logfile)
            logger.info('Subtracting sky from images')
            subtract_sky(maindir, ref_name, sci_name, logger, mask_type=mask_type, ncpu=ncpu)
            reset_logger(logger)
            
            
    if crossconv:
        if pp_separate:
            pass
        else:
            _, logger = setup_logger('NVP.sfft.crossconv', logfile)
            
            logger.info('Cross-convolving images')
            cross_conv(maindir, ref_name, sci_name, conv_ref, conv_sci, skysub, logger)
            logger.info('Finished cross-convolving images')
            
            reset_logger(logger)
        
    

    if cutout_run and (not cutout_together):
        _, logger = setup_logger('NVP.sfft.cutout', logfile)
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
        
            if np.sum(dirs_exist) < 0:        
                func = partial(sfftc.make_cutout_subdir_separate, maindir=maindir, ref_name=ref_name, sci_name=sci_name, cutout_shape=cutout_shape)
                
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
                sfftc.make_split_subdir_separate(cutout_npx, maindir, ref_name, sci_name, npx_boundary=npx_boundary, subset=subset, dither=cutout_dither, 
                                                 pp_separate=pp_separate, skysub=skysub, conv_ref=conv_ref, conv_sci=conv_sci)      
                logger.info('\t Cutout file: {}'.format(cutout_fname))
     
            ras, decs, _, _, n0_vals, n1_vals, fracs_nnz, cutout_names = np.loadtxt(cutout_fname, dtype=object, unpack=True, skiprows=1) 
            ras = np.atleast_1d(ras).astype(float)
            decs = np.atleast_1d(decs).astype(float)     
            n0_vals = np.atleast_1d(n0_vals).astype(int)
            n1_vals = np.atleast_1d(n1_vals).astype(int)                
            cutout_names = np.atleast_1d(cutout_names).astype(str)
            fracs_nnz = np.atleast_1d(fracs_nnz).astype(float)
            ncutout = len(cutout_names)
            
            logger.info('\t {} cutouts'.format(ncutout))
            
            if (subset is not None) and (ncutout != len(subset)):
                logger.info('\t Only using a subset of {} cutouts'.format(len(subset)) )
                
                # for i in range(ncutout):
                #     if i in subset:
                #         continue
                    # else:
                    #     dir_i = maindir + 'output_{}/'.format(cutout_names[i])                    

                        # if os.path.exists(dir_i):
                        #     shutil.rmtree(dir_i)
                
                cutout_names = cutout_names[subset]
                ncutout = len(cutout_names)            
            
            logger.info('Finished splitting images')        
        
        
        
        if skysub and pp_separate:
            _, logger = setup_logger('NVP.sfft.skysub', logfile)
            logger.info('Subtracting sky from cutouts')
            reset_logger(logger)
            
            for i in range(ncutout):
                maindir_i = maindir + 'output_{}/'.format(cutout_names[i])
                logfile_i = maindir_i + 'sfft.log'
                
                _, logger = setup_logger('NVP.sfft.skysub', logfile)
                logger.info('Subtracting sky from cutout {}/{}: {}'.format(i+1, ncutout, cutout_names[i]))
                reset_logger(logger)
                
                _, logger_i = setup_logger('NVP.sfft.skysub', logfile_i)
                subtract_sky(maindir_i, ref_name, sci_name, logger_i, mask_type=mask_type, ncpu=ncpu)
                reset_logger(logger_i)
        
    
    ##############################################################
    #Cross-convolve images
    
    if crossconv:
    
        if cutout_run and (not cutout_together) and pp_separate:
            
            for i in range(ncutout):
                maindir_i = maindir + 'output_{}/'.format(cutout_names[i])
                logfile_i = maindir_i + 'sfft.log'
                
                _, logger = setup_logger('NVP.sfft.crossconv', logfile)
                logger.info('Cross-convolving images for cutout {}/{}: {}'.format(i+1, ncutout, cutout_names[i]))
                reset_logger(logger)
                
                _, logger_i = setup_logger('NVP.sfft.crossconv', logfile_i)
                logger_i.info('Cross-convolving images')
                cross_conv(maindir_i, ref_name, sci_name, conv_ref, conv_sci, skysub, logger_i)
                logger_i.info('Finished cross-convolving images')
                reset_logger(logger_i)    
    
    ##############################################################
    #Make SFFT mask
    
    logfile = maindir + 'sfft.log'
    _, logger = setup_logger('NVP.sfft.makemask', logfile)
    logger.info('Running SExtractor on whole image')
    sfftm.initial_global_fit(maindir, paramdir, 
                            ref_name, sci_name, filtername_ref, filtername_sci, filtername_grid, 
                            skysub, logger, ncpu=ncpu)
    logger.info('Finished running SExtractor on whole image')
    reset_logger(logger)

    #Get global background stddev in cross-convolved images
    if cutout_run and (not cutout_together) and pp_separate:
        bkgstd_ref_global = np.inf
        bkgstd_sci_global = np.inf    
    else:
        
        #Check if all masks are made already
        mask_exist = np.zeros(ncutout, dtype=bool)
        for i in range(ncutout):
            cutout_name = cutout_names[i]
            maindir_i = maindir + 'output_{}/'.format(cutout_name)
            

            fname = maindir_i + 'mask/{}.mask4sfft.fits'.format(ref_name)                
            if os.path.exists(fname):
                mask_exist[i] = True
                
        if np.sum(~mask_exist) > 0:        
            _, logger = setup_logger('NVP.sfft.makemask', logfile)
            logger.info('Getting global background stddev in cross-convolved images')
            
            if conv_ref:
                fname_ref_cc = maindir + 'output/{}.crossconvd.fits'.format(ref_name)
            else:
                if skysub:
                    fname_ref_cc = maindir + 'output/{}.skysub.fits'.format(ref_name)
                else:
                    fname_ref_cc = maindir + 'output/{}.fits'.format(ref_name)
                
            if conv_sci:
                fname_sci_cc = maindir + 'output/{}.crossconvd.fits'.format(sci_name)
            else:
                if skysub:
                    fname_sci_cc = maindir + 'output/{}.skysub.fits'.format(sci_name)
                else:
                    fname_sci_cc = maindir + 'output/{}.fits'.format(sci_name)
    
            
            with fits.open(fname_ref_cc) as hdul:
                im_ref_cc = hdul[0].data.copy()
            with fits.open(fname_sci_cc) as hdul:
                im_sci_cc = hdul[0].data.copy()
                                
            fname_mref = maindir + 'input/{}.maskin.fits'.format(ref_name)
            fname_msci = maindir + 'input/{}.maskin.fits'.format(sci_name)
            with fits.open(fname_mref) as hdul:
                mask_r = hdul[0].data.astype(bool)
            with fits.open(fname_msci) as hdul:
                mask_s = hdul[0].data.astype(bool)
                
            mask_all = mask_r | mask_s

            fname_sexseg_r = maindir + 'mask/{}_sexseg.fits'.format(ref_name)
            fname_sexseg_s = maindir + 'mask/{}_sexseg.fits'.format(sci_name)
            with fits.open(fname_sexseg_r) as hdul:
                segmap_r = hdul[0].data.astype(int)
            with fits.open(fname_sexseg_s) as hdul:
                segmap_s = hdul[0].data.astype(int)
                            
            
            _, _, bkgstd_ref_global = sigma_clipped_stats(im_ref_cc, sigma=3.0, maxiters=None, mask=(mask_all | (segmap_r > 0)) )
            logger.info('Global background stddev in cross-convolved REF image: {:.2e}'.format(bkgstd_ref_global))

            _, _, bkgstd_sci_global = sigma_clipped_stats(im_sci_cc, sigma=3.0, maxiters=None, mask=(mask_all | (segmap_s > 0)) )
            logger.info('Global background stddev in cross-convolved SCI image: {:.2e}'.format(bkgstd_sci_global)) 
            
            del segmap_r, segmap_s
            del mask_r, mask_s, mask_all
            del im_ref_cc, im_sci_cc
            
            reset_logger(logger)
            
        else:
            bkgstd_ref_global = np.inf
            bkgstd_sci_global = np.inf
        
    
    if cutout_run and (not cutout_together):

        for i in range(ncutout):
            maindir_i = maindir + 'output_{}/'.format(cutout_names[i])
            logfile_i = maindir_i + 'sfft.log'
            
            _, logger = setup_logger('NVP.sfft.makemask', logfile)
            logger.info('Making SFFT mask for cutout {}/{}: {}'.format(i+1, ncutout, cutout_names[i]))
            reset_logger(logger)
            
            _, logger_i = setup_logger('NVP.sfft.makemask', logfile_i)
            
            logger_i.info('Making SFFT mask')
            sfftm.make_mask(maindir_i, paramdir, ref_name, sci_name, filtername_ref, filtername_sci, filtername_grid, 
                            skysub, conv_ref, conv_sci, logger_i, sat_ref, sat_sci, 
                            bkgstd_ref_global, bkgstd_sci_global, 
                            ra=ras[i], dec=decs[i], npx_side=(n0_vals[i], n1_vals[i]), ncpu=ncpu)
            logger_i.info('Finished making SFFT mask')
            
            reset_logger(logger_i)
    
    else:
        _, logger = setup_logger('NVP.sfft.makemask', logfile)
        
        logger.info('Making SFFT mask')
        sfftm.make_mask(maindir, paramdir, ref_name, sci_name, filtername_ref, filtername_sci, filtername_grid, 
                        skysub, conv_ref, conv_sci, logger, sat_ref, sat_sci, 
                        bkgstd_ref_global, bkgstd_sci_global,
                        ra=None, dec=None, npx_side=None, ncpu=ncpu)
        logger.info('Finished making SFFT mask')
        
        reset_logger(logger)
    
    ##############################################################
    #Mask images (separate pp, or full image pre-process)
    
    if cutout_run and (not cutout_together):

        #Mask
        for i in range(ncutout):
            maindir_i = maindir + 'output_{}/'.format(cutout_names[i])
            logfile_i = maindir_i + 'sfft.log'
            
            _, logger = setup_logger('NVP.sfft.mask', logfile)
            logger.info('Masking images for cutout {}/{}: {}'.format(i+1, ncutout, cutout_names[i]))
            reset_logger(logger)
            
            _, logger_i = setup_logger('NVP.sfft.mask', logfile_i)
            
            logger_i.info('Masking images')
            sfftu.mask_images(maindir_i, ref_name, sci_name, logger_i, skysub, conv_ref, conv_sci)
            logger_i.info('Finished masking images')
            
            reset_logger(logger_i)
            
            
        #Check number of objects in masked cross-convolved image
        for i in range(ncutout):
            cutout_name = cutout_names[i]
            maindir_i = maindir + 'output_{}/'.format(cutout_name)
            logfile_i = maindir_i + 'sfft.log' 
            
            _, logger = setup_logger('NVP.sfft.mask', logfile)
            logger.info('Checking number of objects in masked image for cutout {}/{}: {}'.format(i+1, ncutout, cutout_name))
            reset_logger(logger)
            
            _, logger_i = setup_logger('NVP.sfft.mask', logfile_i)
            logger_i.info('Checking number of objects in masked image')
            check_num_obj_maskedcc(maindir_i, ref_name, sci_name, logger_i, conv_ref, conv_sci)
            reset_logger(logger_i)
    

    else:    
        _, logger = setup_logger('NVP.sfft.mask', logfile)
        
        logger.info('Masking images')
        sfftu.mask_images(maindir, ref_name, sci_name, logger, skysub, conv_ref, conv_sci)
        logger.info('Finished masking images')
        
        reset_logger(logger)
    
    ##############################################################
    # Get image cutouts
    
    if cutout_run and cutout_together:    
        _, logger = setup_logger('NVP.sfft.cutout', logfile)
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
            
    
            func = partial(make_cutout_subdir_together, maindir=maindir, ref_name=ref_name, sci_name=sci_name, cutout_shape=cutout_shape, skysub=skysub, crossconv=crossconv)
            
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
            cutout_fname = maindir + 'cutout_info.txt'


            if not os.path.exists(cutout_fname):
                logger.info('\t Splitting full image into {}x{} segments'.format(cutout_npx, cutout_npx))
                sfftc.make_split_subdir_together(cutout_npx, maindir, ref_name, sci_name, npx_boundary=npx_boundary, subset=subset, dither=cutout_dither, 
                                                skysub=skysub, conv_ref=conv_ref, conv_sci=conv_sci)           
            
            _, _, _, _, _, _, fracs_nnz, cutout_names = np.loadtxt(cutout_fname, dtype=object, unpack=True, skiprows=1)                        
            cutout_names = np.atleast_1d(cutout_names).astype(str)
            fracs_nnz = np.atleast_1d(fracs_nnz).astype(float)
            ncutout = len(cutout_names)
            
            logger.info('\t Created {} cutouts'.format(ncutout))
            logger.info('\t Cutout file: {}'.format(cutout_fname))
            
            
            if (subset is not None) and (ncutout != len(subset)):
                logger.info('\t Only using a subset of {} cutouts'.format(len(subset)) )
                
                cutout_names = cutout_names[subset]
                ncutout = len(cutout_names)   
        
        
        #Check number of objects in masked cross-convolved image
        for i in range(ncutout):
            cutout_name = cutout_names[i]
            maindir_i = maindir + 'output_{}/'.format(cutout_name)
            logfile_i = maindir_i + 'sfft.log' 
            
            _, logger = setup_logger('NVP.sfft.cutout', logfile)
            logger.info('Checking number of objects in masked image for cutout {}/{}: {}'.format(i+1, ncutout, cutout_name))
            reset_logger(logger)
            
            _, logger_i = setup_logger('NVP.sfft.cutout', logfile_i)
            logger_i.info('Checking number of objects in masked image')
            check_num_obj_maskedcc(maindir_i, ref_name, sci_name, logger_i, conv_ref, conv_sci)
            reset_logger(logger_i)


        logger.info('Finished splitting images')
        reset_logger(logger)
    
    ##############################################################
    #Run SFFT
    
    run = sfft['general']['run']

    if run:        
        _, logger = setup_logger('NVP.sfft.sfft', logfile)
        
        force_conv = sfft['general']['ForceConv']
        gkerhw = sfft['general']['GKerHW']
        minimize_memory_usage = sfft['general']['minimize_memory_usage']
        max_threads = sfft['general']['max_threads_per_block']
        
        if use_gpu:
            ngpu = sfft['general']['ngpu']
        else:
            ngpu = 0
        
        kernel_type1 = sfft['kernel']['type']
        kernel_type2 = sfft['kernel']['type_lownnz']
        kernel_deg1 = sfft['kernel']['degree']
        kernel_deg2 = sfft['kernel']['degree_lownnz']
        nknot_x = sfft['kernel']['nknot_x']
        nknot_y = sfft['kernel']['nknot_y']
        
        separate_scaling = sfft['phot_scale']['separate_scaling']
        scaling_type = sfft['phot_scale']['type']
        scaling_deg = sfft['phot_scale']['degree']
        
        bkg_type = sfft['bkg']['type']
        bkg_deg = sfft['bkg']['degree']
        
        regularize_kernel = sfft['regularization']['regularize_kernel']
        ignore_laplacian_kercent = sfft['regularization']['ignore_laplacian_kercent']
        xy_regularize_fname = sfft['regularization']['xy_regularize_fname']
        lambda_val = sfft['regularization']['lambda']
        
        if filtername_grid in sfftm.NIRCam_filter_FWHM_new['SW'].keys():
            channel = 'SW'
            # gkerhw = 11
            grid = 300
        elif filtername_grid in sfftm.NIRCam_filter_FWHM_new['LW'].keys():
            channel = 'LW'
            # gkerhw = 5
            grid = 150
        elif filtername_grid in sfftm.EUCLID.keys():
            channel = 'euclid'
            # gkerhw = 3
            grid = 100
        
        
        txt = """Running SFFT
--------------------------------
ForceConv: {}
Channel: {}
GKerHW: {}
Number Spline Knots: {}, {}
Kernel type: {}
Kernel degree: {}
Separate scaling: {}
Scaling type: {}
Scaling degree: {}
Background type: {}
Background degree: {}
Regularize kernel: {}
Ignore Laplacian kercent: {}
XY Regularize file: {}
Lambda value: {:.2e}
Minimize memory usage: {}
Max threads per block: {}
Number of CPUs: {}
Use GPU: {}
Number of GPUs: {}
--------------------------------""".format(force_conv, channel, gkerhw, nknot_x, nknot_y, kernel_type1, kernel_deg1, separate_scaling,
                scaling_type, scaling_deg, bkg_type, bkg_deg, regularize_kernel,
                ignore_laplacian_kercent, xy_regularize_fname, lambda_val, 
                minimize_memory_usage, max_threads, ncpu, use_gpu, ngpu)
        logger.info(txt)
        
        
        logger.info('Starting SFFT run')
        reset_logger(logger)
    
        
        if cutout_run:
            for i in range(ncutout):
                cutout_name = cutout_names[i]
                maindir_i = maindir + 'output_{}/'.format(cutout_name)
                logfile_i = maindir_i + 'sfft.log' 
                
                _, logger = setup_logger('NVP.sfft.sfft', logfile)
                logger.info('Running SFFT for cutout {}/{}: {}'.format(i+1, ncutout, cutout_name))
                reset_logger(logger)
                
                
                
                _, logger_i = setup_logger('NVP.sfft.sfft', logfile_i)
                
                if fracs_nnz[i] < .75:
                    logger_i.info('Cutout {} has less than 75% non-zero pixels'.format(cutout_name))
                    logger_i.info('Using low-nnz kernel type: {}'.format(kernel_type2))
                    logger_i.info('Using low-nnz kernel degree: {}'.format(kernel_deg2))
                    kernel_type = kernel_type2
                    kernel_deg = kernel_deg2
                else:
                    kernel_type = kernel_type1
                    kernel_deg = kernel_deg1
                    
                if (kernel_type == scaling_type) and (kernel_deg <= scaling_deg):
                    separate_scaling_in = False
                else:
                    separate_scaling_in = separate_scaling
                    

                sfftu.run_sfft_bspline(maindir_i, ref_name, sci_name, logger_i,
                                    channel, force_conv, gkerhw,
                                    kernel_type, kernel_deg, nknot_x, nknot_y, separate_scaling_in, 
                                    scaling_type, scaling_deg, bkg_type, bkg_deg, 
                                    regularize_kernel, ignore_laplacian_kercent, xy_regularize_fname, lambda_val,
                                    max_threads, minimize_memory_usage, ncpu, use_gpu, ngpu, 
                                    skysub, conv_ref, conv_sci)
                
                reset_logger(logger_i)


        else:
            _, logger = setup_logger('NVP.sfft.sfft', logfile)
            logger.info('Running SFFT')
            reset_logger(logger)

            _, logger = setup_logger('NVP.sfft.sfft', logfile)
            
            if (kernel_type1 == scaling_type) and (kernel_deg <= scaling_deg):
                separate_scaling_in = False
            else:
                separate_scaling_in = separate_scaling

            sfftu.run_sfft_bspline(maindir, ref_name, sci_name, logger,
                                channel, force_conv, gkerhw,
                                kernel_type1, kernel_deg1, nknot_x, nknot_y, separate_scaling_in, 
                                scaling_type, scaling_deg, bkg_type, bkg_deg, 
                                regularize_kernel, ignore_laplacian_kercent, xy_regularize_fname, lambda_val,
                                max_threads, minimize_memory_usage, ncpu, use_gpu, ngpu,
                                skysub, conv_ref, conv_sci)
            
            reset_logger(logger)
        
        _, logger = setup_logger('NVP.sfft.sfft', logfile)
        logger.info('Finished running SFFT')
        reset_logger(logger)
        
    ##############################################################
    # Decorrelate noise
    
    run = sfft['decorr']['run']

    if run:
        _, logger = setup_logger('NVP.sfft.decorr', logfile)
        
        tsize_ratio = sfft['decorr']['tilesize_ratio']
        mcmc_nsamp = sfft['decorr']['mcmc_nsamp']
        
        logger.info('Running noise decorrelation and differential SNR')
        logger.info('\t Tilesize ratio: {:.3f}'.format(tsize_ratio))
        logger.info('\t Number of MCMC samples: {}'.format(mcmc_nsamp))
        logger.info('\t Number of CPUs: {}'.format(ncpu))
        
        reset_logger(logger)
                
                
        if cutout_run:
            for i in range(ncutout):
                cutout_name = cutout_names[i]
                maindir_i = maindir + 'output_{}/'.format(cutout_name)
                logfile_i = maindir_i + 'sfft.log' 
                
                _, logger = setup_logger('NVP.sfft.decorr', logfile)
                logger.info('Running noise decorrelation for cutout {}/{}: {}'.format(i+1, ncutout, cutout_name))
                reset_logger(logger)
                
                _, logger_i = setup_logger('NVP.sfft.decorr', logfile_i)
                
                logger_i.info('Decorrelating noise and getting SNR map')
                sfftu.decorrelate_noise_get_snr(maindir_i, ref_name, sci_name, conv_ref, conv_sci, skysub, logger_i, tsize_ratio, mcmc_nsamp, ncpu, use_gpu)
                reset_logger(logger)


                
        else:
            _, logger = setup_logger('NVP.sfft.decorr', logfile)
            
            logger.info('Decorrelating noise')
            sfftu.decorrelate_noise(maindir, ref_name, sci_name, conv_ref, conv_sci, skysub, logger, tsize_ratio, mcmc_nsamp, ncpu, use_gpu)
            reset_logger(logger)
        
        _, logger = setup_logger('NVP.sfft.decorr', logfile)
        logger.info('Finished noise decorrelation')        
        reset_logger(logger)

    
    ##############################################################
    # Get differential SNR map
        
    # run = sfft['decorr']['run_snr']
    
    # if run:
    #     _, logger = setup_logger('NVP.sfft.diffsnr', logfile)
        
    #     mcmc_nsamp = sfft['decorr']['mcmc_nsamp']
        
    #     logger.info('Getting differential SNR map')
    #     logger.info('\t Number of MCMC samples: {}'.format(mcmc_nsamp))
    #     logger.info('\t Number of CPUs: {}'.format(ncpu))            
        
    #     if cutout_run:
    #         for i in range(ncutout):
    #             cutout_name = cutout_names[i]
    #             maindir_i = maindir + 'output_{}/'.format(cutout_name)
    #             logfile_i = maindir_i + 'sfft.log' 
                
    #             _, logger = setup_logger('NVP.sfft.diffsnr', logfile)
    #             logger.info('Getting differential SNR map for cutout {}/{}: {}'.format(i+1, ncutout, cutout_name))
    #             reset_logger(logger)
                
    #             _, logger_i = setup_logger('NVP.sfft.diffsnr', logfile_i)
    #             sfftu.get_differential_snr(maindir_i, ref_name, sci_name, conv_ref, conv_sci, logger_i, mcmc_nsamp, ncpu)
    #             reset_logger(logger_i)

        
    #     else:
    #         _, logger = setup_logger('NVP.sfft.diffsnr', logfile)
            
    #         logger.info('Getting differential SNR map')
    #         sfftu.get_differential_snr(maindir, ref_name, sci_name, conv_ref, conv_sci, logger, mcmc_nsamp, ncpu)
    #         reset_logger(logger)
        
    #     _, logger = setup_logger('NVP.sfft.diffsnr', logfile)
    #     logger.info('Finished getting differential SNR map')
    #     reset_logger(logger)

    
    ##############################################################
    # Get statistics 

    run = sfft['decorr']['run_stats']

    if run:
        if cutout_run:
            for i in range(ncutout):
                cutout_name = cutout_names[i]
                maindir_i = maindir + 'output_{}/'.format(cutout_name)
                logfile_i = maindir_i + 'sfft.log' 
                
                _, logger = setup_logger('NVP.sfft.stats', logfile)
                logger.info('Getting statistics on signal for cutout {}/{}: {}'.format(i+1, ncutout, cutout_name))
                reset_logger(logger)
                
                _, logger_i = setup_logger('NVP.sfft.stats', logfile_i)
            
                logger_i.info('Getting statistics on signal')
                sfftu.get_statistics(maindir_i, ref_name, sci_name, skysub, logger_i, npx_boundary=npx_boundary)
                logger_i.info('Finished getting statistics on signal')
                reset_logger(logger_i)            
        
        else:       
            _, logger = setup_logger('NVP.sfft.stats', logfile)

        
            logger.info('Getting statistics on signal')
            sfftu.get_statistics(maindir, ref_name, sci_name, skysub, logger, npx_boundary=npx_boundary)
            reset_logger(logger)

        _, logger = setup_logger('NVP.sfft.stats', logfile)
        logger.info('Finished getting statistics on signal')
        reset_logger(logger)
            
    ##############################################################
    
    _, logger = setup_logger('NVP.sfft.finish', logfile)
    logger.info('Finished SFFT')
    reset_logger(logger)
    
    return
    