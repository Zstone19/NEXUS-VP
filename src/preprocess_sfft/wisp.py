import os
import glob
import shutil
import logging
import subprocess
import multiprocessing as mp
from subtract_wisp import process_files

def get_imname(f):
    list_of_str = os.path.basename(f).split('.')[0].split('_')
    return '_'.join(list_of_str[:4])

def compress_file(fname_in):
    _ = subprocess.run(['gzip', '--best', fname_in], check=True, capture_output=True).stdout
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


def run_wisp(indir, maindir, wisp_dir, save_model=True, ncpu=1):
    outdir = maindir + 'preprocess/wisp/'
    logdir = maindir + 'preprocess/logs/'
    
    fnames_in = glob.glob(indir + '*_rate.fits.gz')    
    imnames = [get_imname(fname) for fname in fnames_in]
    loggers = [setup_logger('NVP', logdir + imname + '.log') for imname in imnames]
    
    kwargs = {'sub_wisp': True, 'scale_wisp': True, 'create_segmap': False,
              'factor_max': 4., 'correct_rows': False, 'correct_cols': False,
              'save_data': True, 'save_model': save_model,
              'plot': True, 'show_plot': False,
              'suffix': '_wisp', 'wisp_dir': wisp_dir}
    
    _ = process_files(fnames_in, loggers, outdir, nproc=ncpu, **kwargs)
    
    return


def run_wisp_all(indir, maindir, wisp_dir, save_model=True, ncpu=1):
    os.makedirs(maindir + 'preprocess/wisp/', exist_ok=True)
    run_wisp(indir, maindir, wisp_dir,save_model=save_model, ncpu=ncpu)
    return
