import os
import glob
import configparser
import traceback
import multiprocessing as mp



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



def run_detector1_indiv(rawdir, maindir, imname):
    indir = rawdir
    outdir = maindir + 'detector1/'
    fname_in = indir + imname + '.fits' 
    
    os.makedirs(outdir, exist_ok=True)
    
    log_name = outdir + imname + '_detector1'
    mk_stpipe_log_cfg(outdir, imname + '_detector1')
    
    
    from jwst.pipeline import Detector1Pipeline
    pipe_success = False

    try:
        res = Detector1Pipeline.call(fname_in, output_dir=outdir, 
                                     logcfg="pipeline-log.cfg", save_results=True)
        pipe_success = True
        print('Detector1Pipeline finished for {}'.format(imname))
    except Exception:
        print('\t Detector1Pipeline FAILED FOR {}'.format(imname))
        pipe_crash_msg = traceback.print_exc()
        
    if not pipe_success:
        crashfile = open(log_name + '_pipecrash.txt', 'w')
        print(pipe_crash_msg, file=crashfile)
    
    return




def run_detector1_all(rawdir, maindir, ncpu=1):
    indir = rawdir
    
    raw_fnames = glob.glob(indir + '*.fits')    
    imnames = [os.path.basename(fname).split('.')[0] for fname in raw_fnames]


    print('Running Detector1Pipeline on {} images with {} cores'.format(len(imnames), ncpu))
    if ncpu > 1:
        pool = mp.Pool(ncpu)
        pool.starmap(run_detector1_indiv, [(maindir, imname) for imname in imnames])
        pool.close()
        pool.join()
    else:
        for imname in imnames:
            run_detector1_indiv(maindir, imname)
            
            
    print('Detector1Pipeline finished for all images')
    return
