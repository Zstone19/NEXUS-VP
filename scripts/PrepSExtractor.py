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
epochs = ['01', '03']
bands = ['F200W', 'F444W']
prefix1 = '{}{}'.format(names[0], epochs[0])
prefix2 = '{}{}'.format(names[1], epochs[1])
name_prefix = '{}{}_{}{}'.format(names[0], epochs[0], names[1], epochs[1])

sexdir_sfft = maindir + 'sfftsource_combined_nexus_{}/'.format(name_prefix)
sexdir_sub = maindir + 'subsource_combined_nexus_{}/'.format(name_prefix)
sexdir_in = maindir + 'inputsource_combined_nexus_{}/'.format(name_prefix)
stackdir = maindir + 'NEXUS/stacked_cropped_{}/'.format(name_prefix)

###########
#Make new dirs
###########
shutil.copytree(sexdir_og_sub, sexdir_sub, dirs_exist_ok=True)
shutil.copytree(sexdir_og_sfft, sexdir_sfft, dirs_exist_ok=True)
shutil.copytree(sexdir_og_sfft, sexdir_in, dirs_exist_ok=True)

for f in glob.glob(sexdir_sub + 'output_pos/*'):
    os.remove(f)
for f in glob.glob(sexdir_sfft + 'output_pos/*'):
    os.remove(f)
for f in glob.glob(sexdir_sub + 'ims/*'):
    os.remove(f)
for f in glob.glob(sexdir_sfft + 'ims/*'):
    os.remove(f)
    
for f in glob.glob(sexdir_in + 'output_pos/*'):
    os.remove(f)
for f in glob.glob(sexdir_in + 'ims/*'):
    os.remove(f)
for f in glob.glob(sexdir_in + '*.sex'):
    os.remove(f)

###########
#Prepare sextractor config files
###########
for b in bands:
    #SUB
    with open(sexdir_sub + 'nexus_{}_subdiff_stack.sex'.format(b.lower()), 'r') as f:    
        lines = f.readlines()
    
    lines[4]  = "CATALOG_NAME     {}\n".format(sexdir_sub + 'output_pos/nexus_{}_subdiff_stack.cat'.format(b))
    lines[7]  = "PARAMETERS_NAME  {}dual_sex.outparam  # Will be overwritten anyway! name of the file containing catalog contents\n".format(sexdir_sub)
    lines[16] = "FILTER_NAME      {}gauss_4.0_7x7.conv   # name of the file containing the filter\n".format(sexdir_sub)
    lines[27] = "WEIGHT_IMAGE     {},{}ims/DIFF_{}.noise.fits\n" .format(stackdir + '{}/nexus_{}_stacked_{}.weight.fits'.format(b.lower(), name_prefix, b), sexdir_sub, b)
    lines[31] = "FLAG_IMAGE       {}ims/DIFF_{}.mask.fits      # filename for an input FLAG-image\n".format(sexdir_sub, b)
    lines[53] = "STARNNW_NAME     {}default.nnw    # Neural-Network_Weight table filename\n".format(sexdir_sub)
    lines[67] = "CHECKIMAGE_NAME  {}\n".format(sexdir_sub + 'output_pos/nexus_{}_subdiff_stack.seg'.format(b))

    with open(sexdir_sub + 'nexus_{}_subdiff_stack.sex'.format(b.lower()), 'w') as f:
        f.writelines(lines)
        

    #SFFT
    with open(sexdir_sfft + 'nexus_{}_sfftdiff_stack.sex'.format(b.lower()), 'r') as f:    
        lines = f.readlines()
    
    lines[4]  = "CATALOG_NAME     {}\n".format(sexdir_sfft + 'output_pos/nexus_{}_sfftdiff_stack.cat'.format(b))
    lines[7]  = "PARAMETERS_NAME  {}dual_sex.outparam  # Will be overwritten anyway! name of the file containing catalog contents\n".format(sexdir_sfft)
    lines[16] = "FILTER_NAME      {}gauss_4.0_7x7.conv   # name of the file containing the filter\n".format(sexdir_sfft)
    lines[27] = "WEIGHT_IMAGE     {},{}ims/DIFF_{}.noise.fits\n" .format(stackdir + '{}/nexus_{}_stacked_{}.weight.fits'.format(b.lower(), name_prefix, b), sexdir_sfft, b)
    lines[31] = "FLAG_IMAGE       {}ims/DIFF_{}.mask.fits      # filename for an input FLAG-image\n".format(sexdir_sfft, b)
    lines[53] = "STARNNW_NAME     {}default.nnw    # Neural-Network_Weight table filename\n".format(sexdir_sfft)
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

    ##############################
    #For REF/SCI image detection

    maindir_input = sexdir_in + 'ims/'
    os.makedirs(maindir_input, exist_ok=True)

    shutil.copy(fname_ref, maindir_input + 'REF_{}.fits'.format(b))
    shutil.copy(fname_sci, maindir_input + 'SCI_{}.fits'.format(b))
    shutil.copy(fname_ref_noise, maindir_input + 'REF_{}.noise.fits'.format(b))
    shutil.copy(fname_sci_noise, maindir_input + 'SCI_{}.noise.fits'.format(b))
    shutil.copy(fname_ref_mask, maindir_input + 'REF_{}.mask.fits'.format(b))
    shutil.copy(fname_sci_mask, maindir_input + 'SCI_{}.mask.fits'.format(b))

    ##############################
    #For SUB/SFFT image detection

    maindir_sub = sexdir_sub + 'ims/'
    maindir_sfft = sexdir_sfft + 'ims/'
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
# Copy SExtractor config for input images
###########
for b in bands:
    sexdir_og = '/data4/jwst/nexus/reduced_data/SEXtractor/configs/'
    inds = [6, 9, 21, 37, 43, 60, 69, 87, 49, 68]
    
    #REF
    fname_in = sexdir_og + 'nexus_wide_ep01_NEXUS-Tile-0_{}.sex'.format(b.lower())
    fname_out_r = maindir + 'inputsource_combined_nexus_{}/nexus_{}_ref_stack.sex'.format(name_prefix, b.lower())
    shutil.copy(fname_in, fname_out_r)
    
    #SCI
    fname_out_s = maindir + 'inputsource_combined_nexus_{}/nexus_{}_sci_stack.sex'.format(name_prefix, b.lower())
    shutil.copy(fname_in, fname_out_s)
    
    
    #Adjust REF config file
    with open(fname_out_r, 'r') as f:    
        lines = f.readlines()
    
    lines[inds[0]] = "CATALOG_NAME     {}\n".format(sexdir_in + 'output_pos/nexus_{}_ref_stack.cat'.format(b))
    lines[inds[1]] = "PARAMETERS_NAME  {}dual_sex.outparam  # Will be overwritten anyway! name of the file containing catalog contents\n".format(sexdir_in)
    lines[inds[2]] = "FILTER_NAME      {}gauss_4.0_7x7.conv   # name of the file containing the filter\n".format(sexdir_in)
    lines[inds[3]] = "WEIGHT_IMAGE     {},{}ims/REF_{}.noise.fits\n" .format(stackdir + '{}/nexus_{}_stacked_{}.weight.fits'.format(b.lower(), name_prefix, b), sexdir_in, b)
    lines[inds[4]] = "FLAG_IMAGE       {}ims/REF_{}.mask.fits      # filename for an input FLAG-image\n".format(sexdir_in, b)
    lines[inds[5]] = "MAG_ZEROPOINT    23.9            # magnitude zero-point\n"
    lines[inds[6]] = "STARNNW_NAME     {}default.nnw    # Neural-Network_Weight table filename\n".format(sexdir_in)
    lines[inds[7]] = "CHECKIMAGE_NAME  {}\n".format(sexdir_in + 'output_pos/nexus_{}_subdiff_stack.seg'.format(b))

    if b == 'F200W':
        lines[inds[8]] = "PHOT_APERTURES   6.666666666666667,10.0,16.666666666666668  # MAG_APER aperture diameter(s) in pixels\n"
        lines[inds[9]] = "SEEING_FWHM      0.075           # stellar FWHM in arcsec\n"
    elif b == 'F444W':
        lines[inds[8]] = "PHOT_APERTURES   3.3333333333333335,5.0,8.333333333333334 # MAG_APER aperture diameter(s) in pixels\n"
        lines[inds[9]] = "SEEING_FWHM      0.158           # stellar FWHM in arcsec\n"

    with open(fname_out_r, 'w') as f:
        f.writelines(lines)
        
        
        
    #Adjust SCI config file
    with open(fname_out_s, 'r') as f:    
        lines = f.readlines()
    
    lines[inds[0]] = "CATALOG_NAME     {}\n".format(sexdir_in + 'output_pos/nexus_{}_sci_stack.cat'.format(b))
    lines[inds[1]] = "PARAMETERS_NAME  {}dual_sex.outparam  # Will be overwritten anyway! name of the file containing catalog contents\n".format(sexdir_in)
    lines[inds[2]] = "FILTER_NAME      {}gauss_4.0_7x7.conv   # name of the file containing the filter\n".format(sexdir_in)
    lines[inds[3]] = "WEIGHT_IMAGE     {},{}ims/SCI_{}.noise.fits\n" .format(stackdir + '{}/nexus_{}_stacked_{}.weight.fits'.format(b.lower(), name_prefix, b), sexdir_in, b)
    lines[inds[4]] = "FLAG_IMAGE       {}ims/SCI_{}.mask.fits      # filename for an input FLAG-image\n".format(sexdir_in, b)
    lines[inds[5]] = "MAG_ZEROPOINT    23.9            # magnitude zero-point\n"
    lines[inds[6]] = "STARNNW_NAME     {}default.nnw    # Neural-Network_Weight table filename\n".format(sexdir_in)
    lines[inds[7]] = "CHECKIMAGE_NAME  {}\n".format(sexdir_in + 'output_pos/nexus_{}_subdiff_stack.seg'.format(b))

    if b == 'F200W':
        lines[inds[8]] = "PHOT_APERTURES   6.666666666666667,10.0,16.666666666666668  # MAG_APER aperture diameter(s) in pixels\n"
        lines[inds[9]] = "SEEING_FWHM      0.075           # stellar FWHM in arcsec\n"
    elif b == 'F444W':
        lines[inds[8]] = "PHOT_APERTURES   3.3333333333333335,5.0,8.333333333333334 # MAG_APER aperture diameter(s) in pixels\n"
        lines[inds[9]] = "SEEING_FWHM      0.158           # stellar FWHM in arcsec\n"

    with open(fname_out_s, 'w') as f:
        f.writelines(lines)
    
    

###########
# Run SExtractor
###########
for b in bands:
    #SUB
    cmd = ['sex']
    cmd += [stackdir + '{}/nexus_{}_stacked_{}.fits,{}ims/DIFF_{}.fits'.format(b.lower(), name_prefix, b, sexdir_sub, b)]
    cmd += ['-c'] 
    cmd += [sexdir_sub + 'nexus_{}_subdiff_stack.sex'.format(b.lower())]
    subprocess.run(cmd, check=True)
    
    #SFFT
    cmd = ['sex']
    cmd += [stackdir + '{}/nexus_{}_stacked_{}.fits,{}ims/DIFF_{}.fits'.format(b.lower(), name_prefix, b, sexdir_sfft, b)]
    cmd += ['-c'] 
    cmd += [sexdir_sfft + 'nexus_{}_sfftdiff_stack.sex'.format(b.lower())]
    subprocess.run(cmd, check=True)

    #REF
    cmd = ['sex']
    cmd += [stackdir + '{}/nexus_{}_stacked_{}.fits,{}ims/REF_{}.fits'.format(b.lower(), name_prefix, b, sexdir_in, b)]
    cmd += ['-c'] 
    cmd += [sexdir_in + 'nexus_{}_ref_stack.sex'.format(b.lower())]
    subprocess.run(cmd, check=True)
    
    #SCI
    cmd = ['sex']
    cmd += [stackdir + '{}/nexus_{}_stacked_{}.fits,{}ims/SCI_{}.fits'.format(b.lower(), name_prefix, b, sexdir_in, b)]
    cmd += ['-c'] 
    cmd += [sexdir_in + 'nexus_{}_sci_stack.sex'.format(b.lower())]
    subprocess.run(cmd, check=True)
