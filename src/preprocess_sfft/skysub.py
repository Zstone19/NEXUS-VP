import os
import glob
import configparser
import traceback
import logging
import shutil
import subprocess
import multiprocess as mp
from mpire import WorkerPool
import multiprocessing_logging as mplog


def get_imname(f):
    list_of_str = os.path.basename(f).split('.')[0].split('_')
    return '_'.join(list_of_str[:4])


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
    _ = subprocess.run(['gzip', '--best', fname_in], check=True, capture_output=True).stdout
    return

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




def change_all_fnames(maindir):
    indir = maindir + 'preprocess/image2/'
    outdir = maindir + 'preprocess/skysub/'
    
    raw_fnames = glob.glob(indir + '*_cal.fits')    
    imnames = [get_imname(f) for f in raw_fnames]
    
    for imname in imnames:
        fname_in = outdir + 'step_SkyMatchStep_0_' + imname + '.fits'
        fname_out = outdir + imname + '_skysub.fits'
        
        os.rename(fname_in, fname_out)
        

    return


def run_skymatch_indiv(maindir, imname):
    indir = maindir + 'preprocess/image2/'
    outdir = maindir + 'preprocess/skysub/'
    tmpdir = maindir + 'preprocess/tmp/'
    logdir = maindir + 'preprocess/logs/'
    fname_in = indir + imname + '_cal.fits.gz'
    

    handler, logger = setup_logger('NVP.SkySubtract', logdir + imname + '.log')
    mk_stpipe_log_cfg(tmpdir, logdir, imname)    
    
    
    #Copy input file to temp
    fname_in_tmp = tmpdir + imname + '_cal.fits.gz'
    shutil.copy(fname_in, fname_in_tmp)
    logger.info('Moved unzipped file to temp directory {}'.format(fname_in_tmp))
    
    #Decompress temp input file 
    logger.info('Unzipping file {}'.format(fname_in_tmp))
    _ = subprocess.run(['gunzip', '-k', fname_in_tmp], check=True, capture_output=True).stdout
    fname_in_unzip = fname_in_tmp[:-3]
    logger.info('Unzipped file {}'.format(fname_in))
    
    #Remove temp file
    os.remove(fname_in_tmp)
    logger.info('Deleted temp file {}'.format(fname_in_tmp))
    
    
    from jwst.skymatch import SkyMatchStep
    pipe_success = False
    
    
    #Disable logging from stpipe
    log = logging.getLogger('stpipe')
    log.disabled = True
    
    #Change CRDS logger to output to log file
    handler2, log = setup_crds_logger(logdir + imname + '.log')
    
    logger.info('Running SkyMatchStep on {}'.format(imname))
    try:
        res = SkyMatchStep.call([fname_in_unzip], output_dir=tmpdir, suffix=imname,
                                subtract=True, skymethod='local',
                                logcfg=tmpdir+"pipeline-log-{}.cfg".format(imname), 
                                save_results=True)
        pipe_success = True
        logger.info('SkyMatchStep finished for {}'.format(imname))
    except Exception:
        logger.error('SkyMatchStep FAILED FOR {}'.format(imname))
        pipe_crash_msg = traceback.print_exc()
        
    if not pipe_success:
        crashfile = open(outdir + imname + '_pipecrash.txt', 'w')
        print(pipe_crash_msg, file=crashfile)
        
        
        
    #Delete unzipped file
    os.remove(fname_in_unzip)
    logger.info('Deleted temp unzipped file {}'.format(fname_in_unzip))
    
    #Rename output files
    fname_out1 = tmpdir + 'step_SkyMatchStep_0_' + imname + '.fits'
    os.rename(fname_out1, tmpdir + imname + '_skysub.fits')
    logger.info('Moved output file {}'.format(fname_out1))
    
    #Compress output file
    compress_file(tmpdir + imname + '_skysub.fits')
    logger.info('Compressed output file {}'.format(tmpdir + imname + '_skysub.fits'))
    
    #Move back to outdir
    os.rename(tmpdir + imname + '_skysub.fits.gz', outdir + imname + '_skysub.fits.gz')
    logger.info('Moved compressed output file to image2/ {}'.format(outdir + imname + '_skysub.fits.gz'))
    
    #Delete logger config file
    os.remove(tmpdir + "pipeline-log-{}.cfg".format(imname))
    
    #Make sure logger stops printing
    logger.removeHandler(handler)
    log.removeHandler(handler2)
    
    return


def run_skymatch_all(maindir, ncpu=1):
    indir = maindir + 'preprocess/image2/'
    outdir = maindir + 'preprocess/skysub/'
    os.makedirs(outdir, exist_ok=True)
    
    raw_fnames = glob.glob(indir + '*_cal.fits.gz')    
    imnames = [get_imname(f) for f in raw_fnames]
    
    print('Running SkyMatch for {} images with {} cores'.format(len(imnames), ncpu))
    pool = WorkerPool(n_jobs=ncpu)
    pool.map(run_skymatch_indiv, [(maindir, imname) for imname in imnames],
             progress_bar=True, progress_bar_style='rich')
    pool.join()
            
    print('SkyMatch finished for all images')    
    return
