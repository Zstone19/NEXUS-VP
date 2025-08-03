import os
import glob
import configparser
import traceback
import subprocess
import logging
import shutil
import multiprocessing as mp
from mpire import WorkerPool
import numpy as np
from astropy.io import fits




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


def mk_stpipe_log_cfg(output_dir, log_name):
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
    config.set("*", "handler", "file:" + log_name)
    config.set("*", "level", "INFO")
    pipe_log_config = os.path.join(output_dir, "pipeline-log.cfg")
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

def compress_file(fname_in):
    out = subprocess.run(['gzip', '--best', fname_in], check=True, capture_output=True).stdout
    return out

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





def run_resample_indiv(maindir, fname_epoch, filtername, epoch):
    indir = maindir + 'preprocess/skysub/'
    outdir = maindir + 'mosaic/{}/resamp/'.format(filtername)
    tmpdir = maindir + 'mosaic/tmp/'
    logdir = maindir + 'mosaic/logs/'
    
    imname_all, _, _, epoch_all, filter_all = np.loadtxt(fname_epoch, delimiter=',', dtype=str, unpack=True)
    epoch_all = epoch_all.astype(int)
    
    mask = (epoch_all == epoch) & (filter_all == filtername)
    imnames = imname_all[mask]
    fnames_in_file = [indir + f + '_skysub.fits.gz' for f in imnames]
        
    if len(fnames_in_file) == 0:
        return
    
    fnames_in = []
    for f in fnames_in_file:
        if not os.path.exists(f):
            logger.error('File {} does not exist. Excluding from analysis'.format(f))
        else:
            fnames_in.append(f)
    
    outname = 'epoch{:02d}_{}'.format(epoch, filtername)
    
    #Make directory for output
    os.makedirs(outdir, exist_ok=True)
    
    #Logger
    handler, logger = setup_logger('NVP.Resamp', logdir + outname + '.log')
    mk_stpipe_log_cfg(tmpdir, logdir, outname)
    
    logger.info('Found {} suitable files for resampling'.format(len(fnames_in)))
    for f in fnames_in:
        logger.info('\t\t {}'.format(f))
    
    
    from jwst.resample.resample_step import ResampleStep
    pipe_success = False
    
    #Disable logging from stpipe
    log = logging.getLogger('stpipe')
    log.disabled = True
    
    #Change CRDS logger to output to log file
    handler2, log = setup_crds_logger(logdir + outname + '.log')


    logger.info('Running ResampleStep for epoch{:02d} {}'.format(epoch, filtername))
    try:
        res = ResampleStep.call(fnames_in, output_dir=outdir, 
                                logcfg=tmpdir+"pipeline-log-{}.cfg".format(outname), 
                                save_results=True,
                                suffix='{}_resamp'.format(outname))
        pipe_success = True
        logger.info('ResampleStep finished for epoch{:02d} {}'.format(epoch, filtername))
    except Exception:
        logger.error('ResampleStep FAILED FOR EPOCH{:02d} {}'.format(epoch, filtername))
        pipe_crash_msg = traceback.print_exc()
        
    if not pipe_success:
        crashfile = open(outdir + outname + '_pipecrash.txt', 'w')
        print(pipe_crash_msg, file=crashfile)
    
    
    #Rename output files
    fname_out1 = outdir + 'step_' + outname + '_resamp.fits'
    os.rename(fname_out1, outdir + outname + '_resamp.fits')
    logger.info('Renamed output file to {}'.format(fname_out1))
    
    #Compress output file
    logger.info('Compressing output file')
    msg = compress_file(outdir + outname + '_resamp.fits')
    logger.info(msg)
    logger.info('Compressed output file to {}'.format(outdir + outname + '_outname.fits'))
        
    #Delete logger config file
    os.remove(tmpdir + "pipeline-log-{}.cfg".format(outname))
    
    #Make sure logger stops printing
    logger.removeHandler(handler)
    log.removeHandler(handler2)
    
    return



def run_resample_all(maindir, fname_epoch, ncpu=1):
    # For NEXUS:
    # wide_filters = ['F070W', 'F090W', 'F115W', 'F150W', 'F200W', 'F356W', 'F444W', 'F770W', 'F1000W', 'F1280W']
    # deep_filters = ['F210M', 'F200W', 'F360M', 'F444W', 'F1000W']


    _, _, _, epoch_all, filter_all = np.loadtxt(fname_epoch, delimiter=',', dtype=str, unpack=True)
    epoch_all = epoch_all.astype(int)
    
    epoch_unique = np.unique(epoch_all)
    filter_unique = np.unique(filter_all)
    
    for f in filter_unique:
        os.makedirs(maindir + 'mosaic/{}/resamp/'.format(f), exist_ok=True)    

    inputs = []
    for f in filter_unique:
        for e in epoch_unique:

            nfname = len( np.argwhere((epoch_all == e) & (filter_all == f)).flatten() )
            if nfname == 0:
                continue

            inputs.append((maindir, fname_epoch, f, e))
    
    

    print('Found {} unique epochs'.format(len(epoch_unique)))
    print('Found {} unique filters'.format(len(filter_unique)))
    print('Running Resample for {} epochs/bands with {} cores'.format(len(inputs), ncpu))
    
    pool = WorkerPool(n_jobs=ncpu)
    pool.map(run_resample_indiv, inputs, 
             progress_bar=True, progress_bar_style='rich')
    pool.join()
    
    # run_resample_indiv(*inputs[0])
        
    print('Resample finished for all exposures/bands')
    return