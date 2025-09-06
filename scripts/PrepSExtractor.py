import os
import shutil

import numpy as np
from astropy.io import fits


maindir = '/data6/stone28/nexus/'

names = ['wide', 'deep']
epochs = ['01', '01']
bands = ['F200W', 'F444W']
prefix1 = '{}{}'.format(names[0], epochs[0])
prefix2 = '{}{}'.format(names[1], epochs[1])
name_prefix = '{}{}_{}{}'.format(names[0], epochs[0], names[1], epochs[1])

##############################
#Define file names


#F200W (INPUT)
indir = maindir + 'zogy_nexus_{}_{}/'.format(name_prefix, bands[0])
fname_ref_f200w = indir + 'input/nexus_{}_{}.fits'.format(prefix1, bands[0])
fname_sci_f200w = indir + 'input/nexus_{}_{}.fits'.format(prefix2, bands[0])
fname_ref_f200w_noise = indir + 'noise/nexus_{}_{}.noise.fits'.format(prefix1, bands[0])
fname_sci_f200w_noise = indir + 'noise/nexus_{}_{}.noise.fits'.format(prefix2, bands[0])
fname_ref_f200w_mask = indir + 'input/nexus_{}_{}.maskin.fits'.format(prefix1, bands[0])
fname_sci_f200w_mask = indir + 'input/nexus_{}_{}.maskin.fits'.format(prefix2, bands[0])

#F444W (INPUT)
indir = maindir + 'zogy_nexus_{}_{}/'.format(name_prefix, bands[1])
fname_ref_f444w = indir + 'input/nexus_{}_{}.fits'.format(prefix1, bands[1])
fname_sci_f444w = indir + 'input/nexus_{}_{}.fits'.format(prefix2, bands[1])
fname_ref_f444w_noise = indir + 'noise/nexus_{}_{}.noise.fits'.format(prefix1, bands[1])
fname_sci_f444w_noise = indir + 'noise/nexus_{}_{}.noise.fits'.format(prefix2, bands[1])
fname_ref_f444w_mask = indir + 'input/nexus_{}_{}.maskin.fits'.format(prefix1, bands[1])
fname_sci_f444w_mask = indir + 'input/nexus_{}_{}.maskin.fits'.format(prefix2, bands[1])


#F200W (SUB)
fname_subdiff_f200w = maindir + 'zogy_nexus_{}_{}/output/nexus_{}_{}.subdiff.fits'.format(name_prefix, bands[0], prefix2, bands[0])
fname_subdiff_f200w_noise = maindir + 'zogy_nexus_{}_{}/output/nexus_{}_{}.noise.fits'.format(name_prefix, bands[0], prefix2, bands[0])
fname_subdiff_f200w_mask = maindir + 'zogy_nexus_{}_{}/output/nexus_{}_{}.mask.fits'.format(name_prefix, bands[0], prefix2, bands[0])

#F444W (SUB)
fname_subdiff_f444w = maindir + 'zogy_nexus_{}_F{}output/nexus_{}_{}.subdiff.fits'.format(name_prefix, bands[1], prefix2, bands[1])
fname_subdiff_f444w_noise = maindir + 'zogy_nexus_{}_{}/output/nexus_{}_{}.noise.fits'.format(name_prefix, bands[1], prefix2, bands[1])
fname_subdiff_f444w_mask = maindir + 'zogy_nexus_{}_{}/output/nexus_{}_{}.mask.fits'.format(name_prefix, bands[1], prefix2, bands[1])

#F200W (SFFT)
fname_sfftdiff_f200w = maindir + 'sfft_nexus_{}_{}/output/nexus_{}_{}.sfftdiff.decorr.combined.fits'.format(name_prefix, bands[0], prefix2, bands[0])
fname_sfftdiff_f200w_noise = maindir + 'sfft_nexus_{}_{}/output/nexus_{}_{}.sfftdiff.decorr.noise.combined.fits'.format(name_prefix, bands[0], prefix2, bands[0])
fname_sfftdiff_f200w_mask = maindir + 'sfft_nexus_{}_{}/output/nexus_{}_{}.sfftdiff.decorr.mask.combined.fits'.format(name_prefix, bands[0], prefix2, bands[0])

#F444W (SFFT)
fname_sfftdiff_f444w = maindir + 'sfft_nexus_{}_{}/output/nexus_{}_{}.sfftdiff.decorr.combined.fits'.format(name_prefix, bands[1], prefix2, bands[1])
fname_sfftdiff_f444w_noise = maindir + 'sfft_nexus_{}_{}/output/nexus_{}_{}.sfftdiff.decorr.noise.combined.fits'.format(name_prefix, bands[1], prefix2, bands[1])
fname_sfftdiff_f444w_mask = maindir + 'sfft_nexus_{}_{}/output/nexus_{}_{}.sfftdiff.decorr.mask.combined.fits'.format(name_prefix, bands[1], prefix2, bands[1])

##############################
#For REF/SCI image detection

maindir_input = maindir + 'inputsource_combined_nexus_{}{}_{}{}/ims/'.format(name_prefix)
os.makedirs(maindir_input, exist_ok=True)

shutil.copy(fname_ref_f200w, maindir_input + 'REF_F200W.fits')
shutil.copy(fname_sci_f200w, maindir_input + 'SCI_F200W.fits')
shutil.copy(fname_ref_f200w_noise, maindir_input + 'REF_F200W.noise.fits')
shutil.copy(fname_sci_f200w_noise, maindir_input + 'SCI_F200W.noise.fits')
shutil.copy(fname_ref_f200w_mask, maindir_input + 'REF_F200W.mask.fits')
shutil.copy(fname_sci_f200w_mask, maindir_input + 'SCI_F200W.mask.fits')

shutil.copy(fname_ref_f444w, maindir_input + 'REF_F444W.fits')
shutil.copy(fname_sci_f444w, maindir_input + 'SCI_F444W.fits')
shutil.copy(fname_ref_f444w_noise, maindir_input + 'REF_F444W.noise.fits')
shutil.copy(fname_sci_f444w_noise, maindir_input + 'SCI_F444W.noise.fits')
shutil.copy(fname_ref_f444w_mask, maindir_input + 'REF_F444W.mask.fits')
shutil.copy(fname_sci_f444w_mask, maindir_input + 'SCI_F444W.mask.fits')

##############################
#For SUB/SFFT image detection

maindir_sub = maindir + 'subsource_combined_nexus_{}/ims/'.format(name_prefix)
maindir_sfft = maindir + 'sfftsource_combined_nexus_{}/ims/'.format(name_prefix)
os.makedirs(maindir_sub, exist_ok=True)
os.makedirs(maindir_sfft, exist_ok=True)

with fits.open(fname_subdiff_f200w) as hdul:
    subdat_f200w = hdul[0].data
with fits.open(fname_subdiff_f444w) as hdul:
    subdat_f444w = hdul[0].data
with fits.open(fname_sfftdiff_f200w) as hdul:
    sfftdat_f200w = hdul[0].data
with fits.open(fname_sfftdiff_f444w) as hdul:
    sfftdat_f444w = hdul[0].data
    
mask_f200w = (subdat_f200w == 0) | np.isnan(subdat_f200w)
mask_f444w = (subdat_f444w == 0) | np.isnan(subdat_f444w)



#Save SUB
with fits.open(fname_subdiff_f200w) as hdul:
    im = hdul[0].data
    im[mask_f200w] = 0.
    hdr = hdul[0].header
    fits.writeto(maindir_sub + 'DIFF_F200W.fits', im, hdr, overwrite=True)
    fits.writeto(maindir_sub + 'DIFF_F200W.neg.fits', -1.*im, hdr, overwrite=True)
with fits.open(fname_subdiff_f200w_noise) as hdul:
    im = hdul[0].data
    im[mask_f200w] = 0.
    hdr = hdul[0].header
    fits.writeto(maindir_sub + 'DIFF_F200W.noise.fits', im, hdr, overwrite=True)
with fits.open(fname_subdiff_f200w_mask) as hdul:
    im = hdul[0].data
    im[mask_f200w] = 1
    hdr = hdul[0].header
    fits.writeto(maindir_sub + 'DIFF_F200W.mask.fits', im.astype(np.int16), hdr, overwrite=True)


with fits.open(fname_subdiff_f444w) as hdul:
    im = hdul[0].data
    im[mask_f444w] = 0.
    hdr = hdul[0].header
    fits.writeto(maindir_sub + 'DIFF_F444W.fits', im, hdr, overwrite=True)
    fits.writeto(maindir_sub + 'DIFF_F444W.neg.fits', -1.*im, hdr, overwrite=True)
with fits.open(fname_subdiff_f444w_noise) as hdul:
    im = hdul[0].data
    im[mask_f444w] = 0.
    hdr = hdul[0].header
    fits.writeto(maindir_sub + 'DIFF_F444W.noise.fits', im, hdr, overwrite=True)
with fits.open(fname_subdiff_f444w_mask) as hdul:
    im = hdul[0].data
    im[mask_f444w] = 1
    hdr = hdul[0].header
    fits.writeto(maindir_sub + 'DIFF_F444W.mask.fits', im.astype(np.int16), hdr, overwrite=True)
    
    
    
    
#Save SFFT
with fits.open(fname_sfftdiff_f200w) as hdul:
    im = hdul[0].data
    im[mask_f200w] = 0.
    hdr = hdul[0].header
    fits.writeto(maindir_sfft + 'DIFF_F200W.fits', im, hdr, overwrite=True)
    fits.writeto(maindir_sfft + 'DIFF_F200W.neg.fits', -1.*im, hdr, overwrite=True)
with fits.open(fname_sfftdiff_f200w_noise) as hdul:
    im = hdul[0].data
    im[mask_f200w] = 0.
    hdr = hdul[0].header
    fits.writeto(maindir_sfft + 'DIFF_F200W.noise.fits', im, hdr, overwrite=True)
with fits.open(fname_sfftdiff_f200w_mask) as hdul:
    im = hdul[0].data
    im[mask_f200w] = 1
    hdr = hdul[0].header
    fits.writeto(maindir_sfft + 'DIFF_F200W.mask.fits', im, hdr, overwrite=True)
    
    
with fits.open(fname_sfftdiff_f444w) as hdul:
    im = hdul[0].data
    im[mask_f444w] = 0.
    hdr = hdul[0].header
    fits.writeto(maindir_sfft + 'DIFF_F444W.fits', im, hdr, overwrite=True)
    fits.writeto(maindir_sfft + 'DIFF_F444W.neg.fits', -1.*im, hdr, overwrite=True)
with fits.open(fname_sfftdiff_f444w_noise) as hdul:
    im = hdul[0].data
    im[mask_f444w] = 0.
    hdr = hdul[0].header
    fits.writeto(maindir_sfft + 'DIFF_F444W.noise.fits', im, hdr, overwrite=True)
with fits.open(fname_sfftdiff_f444w_mask) as hdul:
    im = hdul[0].data
    im[mask_f444w] = 1
    hdr = hdul[0].header
    fits.writeto(maindir_sfft + 'DIFF_F444W.mask.fits', im, hdr, overwrite=True)