import os
import shutil

import numpy as np
from astropy.io import fits


maindir = '/data6/stone28/nexus/'

names = ['wide', 'deep']
epochs = ['01', '02']
bands = ['F200W', 'F444W']
prefix1 = '{}{}'.format(names[0], epochs[0])
prefix2 = '{}{}'.format(names[1], epochs[1])
name_prefix = '{}{}_{}{}'.format(names[0], epochs[0], names[1], epochs[1])

os.makedirs(maindir + 'subsource_combined_nexus_{}/'.format(name_prefix), exist_ok=True)
os.makedirs(maindir + 'subsource_combined_nexus_{}/ims/'.format(name_prefix), exist_ok=True)
os.makedirs(maindir + 'sfftsource_combined_nexus_{}/'.format(name_prefix), exist_ok=True)
os.makedirs(maindir + 'sfftsource_combined_nexus_{}/ims/'.format(name_prefix), exist_ok=True)

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
