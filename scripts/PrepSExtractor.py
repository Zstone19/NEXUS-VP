import os
import shutil
import glob
import subprocess

import numpy as np
from astropy.io import fits


maindir = '/data6/stone28/nexus/'

sexdir_og_sfft = maindir + 'nuclear_variability_wide01_deep01/sfftsource_combined_nexus_wide01_deep01/'
sexdir_og_sub = maindir + 'nuclear_variability_wide01_deep01/subsource_combined_nexus_wide01_deep01/'

names = ['wide', 'deep']
epochs = ['01', '02']
bands = ['F200W', 'F444W']
prefix1 = '{}{}'.format(names[0], epochs[0])
prefix2 = '{}{}'.format(names[1], epochs[1])
name_prefix = '{}{}_{}{}'.format(names[0], epochs[0], names[1], epochs[1])

sexdir_sfft = maindir + 'sfftsource_combined_nexus_{}/'.format(name_prefix)
sexdir_sub = maindir + 'subsource_combined_nexus_{}/'.format(name_prefix)
stackdir = maindir + 'NEXUS/stacked_cropped_{}/'.format(name_prefix)

###########
#Make new dirs
###########
shutil.copytree(sexdir_og_sub, sexdir_sub, dirs_exist_ok=True)
shutil.copytree(sexdir_og_sfft, sexdir_sfft, dirs_exist_ok=True)

for f in glob.glob(sexdir_sub + 'output_pos/*'):
    os.remove(f)
for f in glob.glob(sexdir_sfft + 'output_pos/*'):
    os.remove(f)
for f in glob.glob(sexdir_sub + 'ims/*'):
    os.remove(f)
for f in glob.glob(sexdir_sfft + 'ims/*'):
    os.remove(f)


###########
#Prepare sextractor config files
###########
for b in bands:
    #SUB
    with open(sexdir_sub + 'nexus_{}_subdiff_stack.sex'.format(b.lower()), 'r') as f:    
        lines = f.readlines()
    
    lines[4]  = "CATALOG_NAME     {}\n".format(sexdir_sub + 'output_pos/nexus_{}_subdiff_stack.cat'.format(b))
    lines[27] = "WEIGHT_IMAGE     {},{}ims/DIFF_F200W.noise.fits\n" .format(stackdir + 'nexus_{}_{}.weight.fits'.format(name_prefix, sexdir_sub, b))
    lines[67] = "CHECKIMAGE_NAME  {}\n".format(sexdir_sub + 'output_pos/nexus_{}_subdiff_stack.seg'.format(b))

    with open(sexdir_sub + 'nexus_{}_subdiff_stack.sex'.format(b.lower()), 'w') as f:
        f.writelines(lines)
        

    #SFFT
    with open(sexdir_sfft + 'nexus_{}_sfftdiff_stack.sex'.format(b.lower()), 'r') as f:    
        lines = f.readlines()
    
    lines[4]  = "CATALOG_NAME     {}\n".format(sexdir_sfft + 'output_pos/nexus_{}_sfftdiff_stack.cat'.format(b))
    lines[27] = "WEIGHT_IMAGE     {},{}ims/DIFF_F200W.noise.fits\n" .format(stackdir + 'nexus_{}_{}.weight.fits'.format(name_prefix, sexdir_sfft, b))
    lines[67] = "CHECKIMAGE_NAME  {}\n".format(sexdir_sfft + 'output_pos/nexus_{}_sfftdiff_stack.seg'.format(b))

    with open(sexdir_sfft + 'nexus_{}_sfftdiff_stack.sex'.format(b.lower()), 'w') as f:
        f.writelines(lines)



###########
#Copy image files to new dirs
###########

for b in bands:

    ##############################
    #Define file names

    #INPUT
    indir = maindir + 'zogy_nexus_{}_{}/'.format(name_prefix, b)
    fname_ref = indir + 'input/nexus_{}_{}.fits'.format(prefix1, b)
    fname_sci = indir + 'input/nexus_{}_{}.fits'.format(prefix2, b)
    fname_ref_noise = indir + 'noise/nexus_{}_{}.noise.fits'.format(prefix1, b)
    fname_sci_noise = indir + 'noise/nexus_{}_{}.noise.fits'.format(prefix2, b)
    fname_ref_mask = indir + 'input/nexus_{}_{}.maskin.fits'.format(prefix1, b)
    fname_sci_mask = indir + 'input/nexus_{}_{}.maskin.fits'.format(prefix2, b)

    #SUB
    fname_subdiff = maindir + 'zogy_nexus_{}_{}/output/nexus_{}_{}.subdiff.fits'.format(name_prefix, b, prefix2, b)
    fname_subdiff_noise = maindir + 'zogy_nexus_{}_{}/output/nexus_{}_{}.noise.fits'.format(name_prefix, b, prefix2, b)
    fname_subdiff_mask = maindir + 'zogy_nexus_{}_{}/output/nexus_{}_{}.mask.fits'.format(name_prefix, b, prefix2, b)

    #SFFT
    fname_sfftdiff = maindir + 'sfft_nexus_{}_{}/output/nexus_{}_{}.sfftdiff.decorr.combined.fits'.format(name_prefix, b, prefix2, b)
    fname_sfftdiff_noise = maindir + 'sfft_nexus_{}_{}/output/nexus_{}_{}.sfftdiff.decorr.noise.combined.fits'.format(name_prefix, b, prefix2, b)
    fname_sfftdiff_mask = maindir + 'sfft_nexus_{}_{}/output/nexus_{}_{}.sfftdiff.decorr.mask.combined.fits'.format(name_prefix, b, prefix2, b)

    # ##############################
    # #For REF/SCI image detection

    # maindir_input = maindir + 'inputsource_combined_nexus_{}{}_{}{}/ims/'.format(name_prefix)
    # os.makedirs(maindir_input, exist_ok=True)

    # shutil.copy(fname_ref, maindir_input + 'REF_{}.fits'.format(b))
    # shutil.copy(fname_sci, maindir_input + 'SCI_{}.fits'.format(b))
    # shutil.copy(fname_ref_noise, maindir_input + 'REF_{}.noise.fits'.format(b))
    # shutil.copy(fname_sci_noise, maindir_input + 'SCI_{}.noise.fits'.format(b))
    # shutil.copy(fname_ref_mask, maindir_input + 'REF_{}.mask.fits'.format(b))
    # shutil.copy(fname_sci_mask, maindir_input + 'SCI_{}.mask.fits'.format(b))

    ##############################
    #For SUB/SFFT image detection

    maindir_sub = maindir + 'subsource_combined_nexus_{}/ims/'.format(name_prefix)
    maindir_sfft = maindir + 'sfftsource_combined_nexus_{}/ims/'.format(name_prefix)
    os.makedirs(maindir_sub, exist_ok=True)
    os.makedirs(maindir_sfft, exist_ok=True)

    with fits.open(fname_subdiff) as hdul:
        subdat = hdul[0].data
    with fits.open(fname_sfftdiff) as hdul:
        sfftdat = hdul[0].data
        
    mask = (subdat == 0) | np.isnan(subdat)



    #Save SUB
    with fits.open(fname_subdiff) as hdul:
        im = hdul[0].data
        im[mask] = 0.
        hdr = hdul[0].header
        fits.writeto(maindir_sub + 'DIFF_{}.fits'.format(b), im, hdr, overwrite=True)
        fits.writeto(maindir_sub + 'DIFF_{}.neg.fits'.format(b), -1.*im, hdr, overwrite=True)
    with fits.open(fname_subdiff_noise) as hdul:
        im = hdul[0].data
        im[mask] = 0.
        hdr = hdul[0].header
        fits.writeto(maindir_sub + 'DIFF_{}.noise.fits'.format(b), im, hdr, overwrite=True)
    with fits.open(fname_subdiff_mask) as hdul:
        im = hdul[0].data
        im[mask] = 1
        hdr = hdul[0].header
        fits.writeto(maindir_sub + 'DIFF_{}.mask.fits'.format(b), im.astype(np.int16), hdr, overwrite=True)
    
    
    #Save SFFT
    with fits.open(fname_sfftdiff) as hdul:
        im = hdul[0].data
        im[mask] = 0.
        hdr = hdul[0].header
        fits.writeto(maindir_sfft + 'DIFF_{}.fits'.format(b), im, hdr, overwrite=True)
        fits.writeto(maindir_sfft + 'DIFF_{}.neg.fits'.format(b), -1.*im, hdr, overwrite=True)
    with fits.open(fname_sfftdiff_noise) as hdul:
        im = hdul[0].data
        im[mask] = 0.
        hdr = hdul[0].header
        fits.writeto(maindir_sfft + 'DIFF_{}.noise.fits'.format(b), im, hdr, overwrite=True)
    with fits.open(fname_sfftdiff_mask) as hdul:
        im = hdul[0].data
        im[mask] = 1
        hdr = hdul[0].header
        fits.writeto(maindir_sfft + 'DIFF_{}.mask.fits'.format(b), im, hdr, overwrite=True)


###########
# Run SExtractor
###########
for b in bands:
    #SUB
    cmd = ['sex']
    cmd += [stackdir + '{}/nexus_{}_stacked_{}.fits,{}ims/DIFF_{}.fits'.format(b.lower(), name_prefix, b, sexdir_sub, b)]
    cmd += ['-c'] 
    cmd += ['nexus_{}_subdiff_stack.sex'.format(b.lower())]
    subprocess.run(cmd, check=True)
    
    #SFFT
    cmd = ['sex']
    cmd += [stackdir + '{}/nexus_{}_stacked_{}.fits,{}ims/DIFF_{}.fits'.format(b.lower(), name_prefix, b, sexdir_sfft, b)]
    cmd += ['-c'] 
    cmd += ['nexus_{}_sfftdiff_stack.sex'.format(b.lower())]
    subprocess.run(cmd, check=True)