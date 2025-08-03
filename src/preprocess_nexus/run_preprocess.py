import os
import glob
import shutil
import logging

import numpy as np
from astropy.io import fits

import detector1 as d1
import step2 as s2

def run_pipeline(maindir, indir, ncpu=1):
    
    os.makedirs(maindir, exist_ok=True)
    os.makedirs(maindir + 'preprocess/')
    os.makedirs(maindir + 'preprocess/detector1/')
    os.makedirs(maindir + 'preprocess/logs/')
    os.makedirs(maindir + 'preprocess/tmp/')
    os.makedirs(maindir + 'preprocess/step2/')
    os.makedirs(maindir + 'preprocess/wisps')
    
    
    #############################
    #Step 1
    d1.run_detector1_all(indir, maindir, ncpu)
    
    #############################
    #Step 2
    
    
    
    