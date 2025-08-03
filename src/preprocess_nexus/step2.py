import os
import shutil
import glob
import subprocess
import logging
import datetime
from mpire import WorkerPool

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel, convolve_fft

from stdatamodels.jwst.datamodels import dqflags
from stdatamodels import util
from jwst import datamodels, outlier_detection

from photutils.background import Background2D, SExtractorBackground
from destriping import measure_striping, get_source_mask




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





def get_imname(f):
    list_of_str = os.path.basename(f).split('.')[0].split('_')
    return '_'.join(list_of_str[:4])

def compress_file(fname_in):
    _ = subprocess.run(['gzip', '--best', fname_in], check=True, capture_output=True).stdout
    return

def decompress_file(fname_in):
    _ = subprocess.run(['gunzip', '-k', fname_in], check=True, capture_output=True).stdout
    return

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def mk_stpipe_log_cfg(output_dir, logdir, imname):
    """
    Create a configuration file with the name log_name, where
    the pipeline will write all output.
    Args:
        outpur_dir: str, path of the output directory
        log_name: str, name of the log to record screen output
    Returns:
        nothing
    """
    config = configparser.ConfigParser()
    config.add_section("*")
    config.set("*", "level", "DEBUG")
    config.set("*", "handler", 'append:' + logdir + imname + '.log')
    pipe_log_config = output_dir + "pipeline-log-{}.cfg".format(imname)
    config.write(open(pipe_log_config, "w"))
    
    return


def setup_crds_logger(log_file):
    
    log = logging.getLogger('CRDS')
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler2 = logging.FileHandler(log_file)
    handler2.setFormatter(formatter)
    log.setLevel(logging.DEBUG)
    
    if not len(log.handlers):
        log.addHandler(handler2)

    return handler2, log

def reset_logger(logger):
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    return


########################################################################################################################################################################
# Step 2

# 1. Run Image2Pipeline with default parameters on each image

# 2. Generate source mask in smoothed image
#  - Smooth w/ Gaussian (FWHM=3px) with dilation radius of 10px (5px) for SW (LW) filters
#  - Get background with photutils SExtractorBackground, using 256x256px box size and 3x3 boxes
#  - Threshold at 2sigma above background, remove background
#  - Repeat twice

# 3. Remove 1/f noise (vert+horiz stripes)

# 4. Subtract wisps
#  - Use A3, A4, B3, B4 detectors
#  - Get wisp templates by stacking 2σ-clipped, source emission masked-images in each detector and each band
#  - Subtract template from each exposure without scaling

# 5. Destripe SW images
#  - Get template by median stacking data/error images from all the affected SW exposures after masking bright sources and a 2σ-clipping
#  - Best-fit scaling factor is multiplied to the template before subtracting the striping features in individual exposures



def run_image2_indiv(maindir, indir, fname_in, imname):
    logdir = maindir + 'preprocess/logs/'
    tmpdir = maindir + 'preprocess/tmp/'


    handler, logger = setup_logger('NVP.Image2Pipeline', logdir + imname + '.log')
    mk_stpipe_log_cfg(tmpdir, logdir, imname)    
    

    if tmpdir not in fname_in:
        #Move file to tmp
        fname_in_tmp = tmpdir + imname + '_image2_input.fits.gz'
        shutil.copy(fname_in, fname_in_tmp)
        logger.info('Moved unzipped file to temp directory {}'.format(fname_in))
    else:
        fname_in_tmp = fname_in


    if '.gz' in fname_in:        
        #Decompress file
        logger.info('Unzipping file {}'.format(fname_in_tmp))
        decompress_file(fname_in_tmp)
        fname_in_unzip = fname_in_tmp[:-3]
        logger.info('Unzipped file {}'.format(fname_in))
        
        #Remove temp file
        os.remove(fname_in_tmp)
        logger.info('Deleted temp file {}'.format(fname_in_tmp))
    else:
        fname_in_unzip = fname_in_tmp




    from jwst.pipeline import Image2Pipeline
    pipe_success = False
    
    #Disable logging from stpipe
    log = logging.getLogger('stpipe')
    log.disabled = True
    
    #Change CRDS logger to output to log file
    handler2, log = setup_crds_logger(logdir + imname + '.log')
    
    
    logger.info('Running Image2Pipeline on {}'.format(imname))
    try:
        res = Image2Pipeline.call(fname_in, output_dir=tmpdir, 
                                     logcfg=tmpdir+"pipeline-log-{}.cfg".format(imname), 
                                     save_results=True)
        pipe_success = True
        logger.info('Image2Pipeline finished'.format(imname))
    except Exception:
        logger.error('Image2Pipeline FAILED FOR {}'.format(imname))
        pipe_crash_msg = traceback.print_exc()
        
    if not pipe_success:
        crashfile = open(outdir + imname + '_pipecrash.txt', 'w')
        print(pipe_crash_msg, file=crashfile)
        
        
    #Delete unzipped file
    os.remove(fname_in_unzip)
    logger.info('Deleted temp unzipped file {}'.format(fname_in_unzip))

    #Delete unneeded output files    
    fname_out = tmpdir + imname + '_i2d.fits'
    os.remove(fname_out)
    logger.info('Removed output file {}'.format(fname_out))
    
    #Delete logger config file
    os.remove(tmpdir + "pipeline-log-{}.cfg".format(imname))

    #Make sure logger stops printing    
    reser_logger(logger)
    reset_logger(log)
    
    return



def destripe_indiv(maindir, indir, imname, logger):
    
    logdir = maindir + 'preprocess/logs/'
    tmpdir = maindir + 'preprocess/tmp/'
    outdir = maindir + 'preprocess/step2/'
    fname_in = indir + imname + '_rate.fits'
    
    fname_mask = outdir + imname + '_sourcemask.fits'
    fname_bkg = outdir + imname + '_bkg.fits'
    fname_v = outdir + imname + '_v.fits'
    fname_h = outdir + imname + '_h.fits'
    fname_out = tmpdir + imname + '_destriped.fits'
    measure_destriping(fname_in, fname_out, fname_mask, fname_bkg,
                     fname_v, fname_h, logger, 
                     thresh=None, apply_flat=True, mask_sources=True, 
                     save_patterns=True)


    #Compress auxiliary files
    compress_file(fname_mask)
    compress_file(fname_bkg)
    compress_file(fname_v)
    compress_file(fname_h)
    
    #Move auxiliary files
    shutil.move(fname_mask + '.gz', outdir)
    shutil.move(fname_bkg + '.gz', outdir)
    shutil.move(fname_v + '.gz', outdir)
    shutil.move(fname_h + '.gz', outdir)

    return    


def make_wisp_templates(maindir, outdir):
    tmpdir = maindir + 'preprocess/tmp/'
    logdir = maindir + 'preprocess/logs/'
    step2dir = maindir + 'preprocess/step2/'
    outdir = maindir + 'preprocess/wisps/'

    #Get all images with wisps
    fnames_all = glob.glob(tmpdir + '*_destriped.fits')
    relevant_detectors = ['a3', 'a4', 'b4', 'b3']

    fnames_wisp = []
    for fname in fnames_all:
        
        for det in relevant_detectors:
            if filt in fname:
                fnames_wisp.append(fname)
                break


    #Break up by filter
    fnames_wisp_dict = {}
    imnames_wisp_dict = {}
    for fname in fnames_wisp:
        filt = fits.getheader(fname)['FILTER']
        
        if filt not in fnames_wisp_dict:
            fnames_wisp_dict[filt] = []
            imnames_wisp_dict[filt] = []

        fnames_wisp_dict[filt].append(fname)
        imnames_wisp_dict[filt].append(get_imname(fname))


    #Stack images
    for filt in fnames_wisp_dict:
        ims = []
        
        #Skip if the template exists already
        if os.path.exists(outdir + filt + '_wisp_template.fits'):
            continue

        
        fnames = fnames_wisp_dict[filt]
        imnames = imnames_wisp_dict[filt]
        for i in range(len(fnames)):
            im = fits.getdata(fname, ext='SCI')
            mask = fits.getdata(step2dir + imname + '_sourcemask.fits.gz')
            
            im[mask] = np.nan         
            
            #2sigma clip
            mean, median, std = sigma_clipped_stats(im, sigma=2)
            im[im > median + 2*std] = np.nan
            
            ims.append(im)

        im_stack = np.nanmedian(ims, axis=0)

        #Save
        fname_out = outdir + filt + '_wisp_template.fits'
        hdu = fits.PrimaryHDU(im_stack)
        hdu.writeto(fname_out, overwrite=True)

    return


def subtract_wisps(maindir, imname, logger):
    logdir = maindir + 'preprocess/logs/'
    tmpdir = maindir + 'preprocess/tmp/'
    wispdir = maindir + 'preprocess/wisps/'
    
    fname_in = tmpdir + imname + '_destriped.fits'
    fname_out = tmpdir + imname + '_dewisped.fits'
    
    logger.info('Subtracting wisp from {}'.format(imname))
    
    filt = fits.getheader(fname_in)['FILTER']
    fname_wisp = wispdir + filt + '_wisp_template.fits'
    logger.info('Using wisp template {}'.format(fname_wisp))
    
    im_wisp = fits.getdata(fname_wisp)

    with datamodels.open(fname_in) as im:
        im_dq = im.dq
        good_mask = (im_dq == 0)
    
        im.data[~good_mask] -= im_wisp[~good_mask]
        
        #Add history
        time = datetime.now()
        stepdescription = 'NVP: Subtract wisp template %s'%time.strftime('%Y-%m-%d %H:%M:%S')
        substr = util.create_history_entry(stepdescription)
        im.history.append(substr)

        #Save
        im.save(fname_out)
        
    logger.info('Finished dewisping {}'.format(imname))
    logger.info('Saved dewisped image to {}'.format(fname_out))
        
    #Delete input file
    os.remove(fname_in)
    logger.info('Removed input file {}'.format(fname_in))
    
    return


def run_bkg_subtract_indiv(maindir, imname, logger, suffix='bkg'):
    logdir = maindir + 'preprocess/logs/'
    tmpdir = maindir + 'preprocess/tmp/'
    outdir = maindir + 'preprocess/step2/'
    fname_in = tmpdir + imname + '_cal.fits'
    
    im = datamodels.open(fname_in)
    im_dat = np.array(im.data)
    im_dq = im.dq
    good_mask = (im_dq == 0)
    im.close()
    
    
    if 'long' in imname:
        radius = 5
    else:
        radius = 10
    
    
    #Run once
    logger.info('Running background subtraction on {}'.format(imname))
    source_mask1 = get_source_mask(im_dat, ~good_mask, nsig=2, kernel_fwhm=3, dilation_radius=radius)
    source_mask_tot = source_mask1 | (~good_mask)
    
    estimator = SExtractorBackground()
    bkg1 = Background2D(im_dat, box_size=(256, 256), mask=source_mask_tot, bkg_estimator=estimator, exclude_percentile=30) #Changed 10->30




    #Run again
    logger.info('Running background subtraction again on {}'.format(imname))
    source_mask = get_source_mask(im_dat-bkg1.background, ~good_mask, nsig=2, kernel_fwhm=3, dilation_radius=radius)
    source_mask_tot = source_mask | (~good_mask)
    bkg2 = Background2D(im_dat, box_size=(256, 256), mask=source_mask_tot, bkg_estimator=estimator, exclude_percentile=30)
    
    
    
    
    #Save background as an auxiliary image
    header = fits.open(fname_in)[0].header
    hdu1 = fits.PrimaryHDU(header=header)
    hdu2 = fits.ImageHDU(bkg2.background)
    hdu3 = fits.ImageHDU(bkg2.background_rms)
    
    hdu2.name = 'BKG'
    hdu3.name = 'RMS'

    hdu2.header['SIZE'] = 256
    hdu2.header['BKG_EST'] = 'SExtractorBackground'
    
    
    hdul = fits.HDUList([hdu1, hdu2, hdu3])
    hdul.writeto(outdir + imname + '_{}.fits'.format(suffix), overwrite=True)
    logger.info('Saved background to {}'.format(outdir + imname + '_{}.fits'.format(suffix)))
    
    #Compress auxiliary files
    compress_file(outdir + imname + '_{}.fits'.format(suffix))
    logger.info('Compressed background file {}'.format(outdir + imname + '_{}.fits'.format(suffix)))
    
    
    
    #Save background-subtracted image
    with datamodels.open(fname_in) as im_in:
        
        #Subtract background
        im_in.data = im_dat - bkg2.background

        #Set pixels with error=0 to 0
        mask_err0 = (im_in.err==0) & (~np.isnan(im_in.data))
        im_in.data[mask_err0] = 0.
        
        #Add history
        time = datetime.datetime.now()
        stepdescription = 'NVP: Subtract 2D background using photutils %s'%time.strftime('%Y-%m-%d %H:%M:%S')
        substr = util.create_history_entry(stepdescription)
        im_in.history.append(substr)

        #Save
        im_in.save(tmpdir + imname + '_subbkg.fits')
        logger.info('Saved background-subtracted image to {}'.format(tmpdir + imname + '_subbkg.fits'))

    return

########################################################################################################################################################################
########################################################################################################################################################################

def run_step2_all(maindir, indir, im2=True, destripe=True, wisp=True, bkg=True, move=True, ncpu=1):    
    
    tmpdir = maindir + 'preprocess/tmp/'
    wispdir = maindir + 'preprocess/wisps/'
    logdir = maindir + 'preprocess/logs/'
    outdir = maindir + 'preprocess/step2/'
    
    ########################################################################################################
    #Run image2pipeline
    
    if im2:    
        fnames_in = glob.glob(indir + '*.fits')
        imnames_in = [get_imname(f) for f in fnames_in]    
        loggers_in = [setup_logger('NVP.Image2Pipeline', logdir + imname + '.log') for imname in imnames_in]
        
        inputs = [(maindir, indir, fname_in, imname, logger) for fname_in, imname, logger in zip(fnames_in, imnames_in, loggers_in)]
        pool = WorkerPool(ncpu)
        pool.map(run_image2_indiv, inputs,  progress_bar=True, progress_bar_style='rich')
        pool.join()
    
    
    ########################################################################################################
    #Destripe
    
    if destripe:
        fnames_in = glob.glob(tmpdir + '*_cal.fits')
        imnames_in = [get_imname(f) for f in fnames_in]
        loggers_in = [setup_logger('NVP.Destripe', logdir + imname + '.log') for imname in imnames_in]
        
        inputs = [(maindir, indir, imname, logger) for imname, logger in zip(imnames_in, loggers_in)]
        pool = WorkerPool(ncpu)
        pool.map(destripe_indiv, inputs,  progress_bar=True, progress_bar_style='rich')
        pool.join()
    
    ########################################################################################################
    #Make wisp templates
    
    if wisp:
        make_wisp_templates(maindir, wispdir)
        
        #Subtract wisps
        if not destripe:
            fnames_in = glob.glob(tmpdir + '*_cal.fits')
            imnames_in = [get_imname(f) for f in fnames_in]
            
            for i in range(len(imnames_in)):
                fi = tmpdir + imnames_in[i] + '_destriped.fits'
                shutil.copy(fnames_in[i], fi)
            

        fnames_in = glob.glob(tmpdir + '*_destriped.fits')    
        imnames_in = [get_imname(f) for f in fnames_in]
        loggers_in = [setup_logger('NVP.SubtractWisps', logdir + imname + '.log') for imname in imnames_in]
        
        inputs = [(maindir, imname, logger) for imname, logger in zip(imnames_in, loggers_in)]
        pool = WorkerPool(ncpu)
        pool.map(subtract_wisps, inputs,  progress_bar=True, progress_bar_style='rich')
        pool.join()
    
    
    ########################################################################################################
    #Background subtraction
    
    if bkg:
        fnames_in = glob.glob(tmpdir + '*_cal.fits')
        imnames_in = [get_imname(f) for f in fnames_in]
        loggers_in = [setup_logger('NVP.BkgSubtract', logdir + imname + '.log') for imname in imnames_in]
        
        inputs = [(maindir, imname, logger) for imname, logger in zip(imnames_in, loggers_in)]
        pool = WorkerPool(ncpu)
        pool.map(run_bkg_subtract_indiv, inputs,  progress_bar=True, progress_bar_style='rich')
        pool.join()
    
    
    ########################################################################################################
    #Move output

    if move:
        fnames_out = glob.glob(tmpdir + '*_subbkg.fits')
        imnames_out = [get_imname(f) for f in fnames_out]
        fnames_step2 = [outdir + imname + '_step2.fits' for imname in imnames_out]
        
        print('Moving output files to {}'.format(outdir))
        for i in range(len(fnames_out)):
            shutil.move(fnames_out[i], fnames_step2[i])

        # pool.map(shutil.copy, zip(fnames_out, fnames_step2), 
        #         progress_bar=True, progress_bar_style='rich')
        # pool.join()

        print('Compressing output files')    
        pool = WorkerPool(ncpu)
        pool.map(compress_file, fnames_step2,
                progress_bar=True, progress_bar_style='rich')
        pool.join()
            
    return
    
