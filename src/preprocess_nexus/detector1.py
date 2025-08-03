import os
import glob
import configparser
import traceback
import shutil
import logging
import subprocess
from mpire import WorkerPool



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






def run_detector1_indiv(rawdir, maindir, imname):
    indir = rawdir
    outdir = maindir + 'preprocess/detector1/'
    tmpdir = maindir + 'preprocess/tmp/'
    logdir = maindir + 'preprocess/logs/'
    fname_in = indir + imname + '_uncal.fits.gz' 
    

    handler, logger = setup_logger('NVP.Detector1Pipeline', logdir + imname + '.log')
    mk_stpipe_log_cfg(tmpdir, logdir, imname)    
    
    
    #Move file to tmp
    fname_in_tmp = tmpdir + imname + '_uncal.fits.gz'
    shutil.copy(fname_in, fname_in_tmp)
    logger.info('Moved unzipped file to temp directory {}'.format(fname_in))
    
    #Decompress file
    logger.info('Unzipping file {}'.format(fname_in_tmp))
    decompress_file(fname_in_tmp)
    fname_in_unzip = fname_in_tmp[:-3]
    logger.info('Unzipped file {}'.format(fname_in))
    
    #Remove temp file
    os.remove(fname_in_tmp)
    logger.info('Deleted temp file {}'.format(fname_in_tmp))
    

    
    from jwst.pipeline import Detector1Pipeline
    pipe_success = False
    
    #Disable logging from stpipe
    log = logging.getLogger('stpipe')
    log.disabled = True
    
    #Change CRDS logger to output to log file
    handler2, log = setup_crds_logger(logdir + imname + '.log')
    
    
    logger.info('Running Detector1Pipeline on {}'.format(imname))
    try:
        res = Detector1Pipeline.call(fname_in, output_dir=tmpdir, 
                                     logcfg=tmpdir+"pipeline-log-{}.cfg".format(imname), 
                                     save_results=True,
                                     steps={'jump': {'expand_large_events': True}})
        pipe_success = True
        logger.info('Detector1Pipeline finished'.format(imname))
    except Exception:
        logger.error('Detector1Pipeline FAILED FOR {}'.format(imname))
        pipe_crash_msg = traceback.print_exc()
        
    if not pipe_success:
        crashfile = open(outdir + imname + '_pipecrash.txt', 'w')
        print(pipe_crash_msg, file=crashfile)
        
        
    #Delete unzipped file
    os.remove(fname_in_unzip)
    logger.info('Deleted temp unzipped file {}'.format(fname_in_unzip))
        
        
    #Delete unneeded output files    
    fname_out1 = tmpdir + imname + '_rateints.fits'
    os.remove(fname_out1)
    logger.info('Removed output file {}'.format(fname_out1))
    
    fname_out2 = tmpdir + imname + '_ramp.fits'
    os.remove(fname_out2)
    logger.info('Removed output file {}'.format(fname_out2))
    
    
    
    
    
    #Compress output file
    compress_file(tmpdir + imname + '_rate.fits')
    logger.info('Compressed output file {}'.format(tmpdir + imname + '_rate.fits'))
    
    #Move back to outdir
    os.rename(tmpdir + imname + '_rate.fits.gz', outdir + imname + '_rate.fits.gz')
    logger.info('Moved compressed output file to detector1/ {}'.format(outdir + imname + '_rate.fits.gz'))

    #Delete logger config file
    os.remove(tmpdir + "pipeline-log-{}.cfg".format(imname))

    #Make sure logger stops printing    
    reser_logger(logger)
    reset_logger(log)

    return




def run_detector1_all(rawdir, maindir, ncpu=1):
    indir = rawdir
    outdir = maindir + 'preprocess/detector1/'
    logdir = maindir + 'preprocess/logs/'
    tmpdir = maindir + 'preprocess/tmp/'
    
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(tmpdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    
    raw_fnames = glob.glob(indir + '*.fits.gz')    
    imnames = [get_imname(f) for fname in raw_fnames]


    print('Running Detector1Pipeline on {} images with {} cores'.format(len(imnames), ncpu))
    pool = WorkerPool(n_jobs=ncpu)
    pool.map(run_image2_indiv, [(maindir, imnames[i]) for i in range(len(imnames))],
                 progress_bar=True, progress_bar_style='rich')
    pool.join()
            
            
    print('Detector1Pipeline finished for all images')
    return
