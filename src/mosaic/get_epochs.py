import glob
import os

import numpy as np
from astropy.io import fits



def get_mjd_from_fname(fname):
    return float(fits.open(fname)[0].header['EXPMID'])

def get_filter_from_fname(fname):
    return fits.open(fname)[0].header['FILTER']

def get_imname(f):
    list_of_str = os.path.basename(f).split('.')[0].split('_')
    return '_'.join(list_of_str[:4])

def get_epoch_file(maindir, fname_out):
    fnames_all = glob.glob(maindir + 'preprocess/skysub/*_skysub.fits.gz')
    imnames_all = [ get_imname(f) for f in fnames_all ]
    filters_all = [ get_filter_from_fname(f) for f in fnames_all ]
    
    mjd_all = np.array([ get_mjd_from_fname(f) for f in fnames_all ])
    mjd_rounded = np.round(mjd_all, 1).astype(int)
    
    mjd_unique = np.unique(mjd_rounded)
    sort_ind = np.argsort(mjd_unique)
    mjd_unique = mjd_unique[sort_ind]
    
    epochs_all = []
    for i in range(len(mjd_rounded)):
        ind = np.argwhere(mjd_rounded[i] == mjd_unique).flatten()[0]
        epochs_all.append(ind+1)
 
    with open(fname_out, 'w+') as f:
        f.write('#imname,mjd,rounded_mjd,epoch,filter\n')
        
        for i in range(len(fnames_all)):
            f.write(imnames_all[i] + ',' + str(mjd_all[i]) + ',' + str(mjd_rounded[i]) + ',' + str(epochs_all[i]) + ',' + filters_all[i] + '\n')
            
    return