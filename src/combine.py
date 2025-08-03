import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

import subprocess
import glob
import shutil
import os

import sys
sys.path.append('psf/')
import psf_euclid as psfe



def make_swarp_config(maindir, fname_swarp_default, prefix, fname_out, swarp_type='data', ncpu=1):
    
    with open(fname_swarp_default, 'r') as f:
        content = f.readlines()
        
        
    content[4]   = "IMAGEOUT_NAME          {}.fits          # Output filename\n".format(maindir+prefix)
    content[5]   = "WEIGHTOUT_NAME         {}.weight.fits   # Output weight-map filename\n".format(maindir+prefix)
    
    if swarp_type in ['mask', 'data', 'psf']:
        content[12]  = "WEIGHT_TYPE            MAP_RMS         # BACKGROUND,MAP_RMS,MAP_VARIANCE\n"    
    else:
        content[12]  = "WEIGHT_TYPE            NONE            # BACKGROUND,MAP_RMS,MAP_VARIANCE\n"
    content[14]  = "RESCALE_WEIGHTS        Y               # Rescale input weights/variances (Y/N)?\n"
    content[15]  = "WEIGHT_SUFFIX          _err.fits        # Suffix to use for weight-maps\n"

    content[22]  = "COMBINE                Y               # Combine resampled images (Y/N)?\n"
    if swarp_type in ['mask']:
        content[23]  = "COMBINE_TYPE           OR              # MEDIAN,AVERAGE,MIN,MAX,WEIGHTED,CLIPPED\n"        
    elif swarp_type in ['data', 'psf']:
        content[23]  = "COMBINE_TYPE           MEDIAN          # MEDIAN,AVERAGE,MIN,MAX,WEIGHTED,CLIPPED\n"
    elif swarp_type in ['var']:
        content[23]  = "COMBINE_TYPE           SUM             # MEDIAN,AVERAGE,MIN,MAX,WEIGHTED,CLIPPED\n"

    content[76]  = "SUBTRACT_BACK          N               # Subtraction sky background (Y/N)?\n"
    
    content[113] = "NTHREADS               {}              # Number of simultaneous threads for\n".format(ncpu)
        
    f = open(fname_out, 'w+')
    f.writelines(content)
    f.close()
    
    return


def run_swarp(maindir, fnames_sci, fname_default, prefix, swarp_type='data', ncpu=1, verbose=False):
    
    #Make config file
    fname_swarp = maindir + '{}.swarp'.format(prefix)
    make_swarp_config(maindir, fname_default, prefix, fname_swarp, swarp_type, ncpu)
    
    #Run SWarp
    command = ['swarp']
    command += list(fnames_sci)
    command += ['-c', fname_swarp]
    
    if verbose:
        stdout = subprocess.PIPE
    else:
        stdout = None
    
    print('Running SWarp for {}'.format(prefix))
    out = subprocess.run(command, check=True, stdout=stdout, stderr=stdout).stdout
    print(out)
    print('SWarp done for {}'.format(prefix))

    return


def combine_frames(maindir, fname_default, sci_name, ncpu=1, verbose=False):
    
    fnames_sci = glob.glob(maindir + '{}*_data.fits'.format(sci_name))
    
    for f in fnames_sci:
        fname_err_i = f.replace('_data.fits', '_err.fits')
        fname_mask_i = f.replace('_data.fits', '_mask.fits')
        fname_gridpsf_i = f.replace('_data.fits', '_gridpsf.fits')
        
        if not os.path.exists(fname_err_i):
            raise FileNotFoundError('Error file {} not found'.format(fname_err_i))
        
        if not os.path.exists(fname_mask_i):
            raise FileNotFoundError('Mask file {} not found'.format(fname_mask_i))
        
        if not os.path.exists(fname_gridpsf_i):
            raise FileNotFoundError('GridPSF file {} not found'.format(fname_gridpsf_i))


    fnames_input = [f.replace('_data', '') for f in fnames_sci]
    
    ####################################################################################################

    #Run SWarp for data images
    for f, fi in zip(fnames_sci, fnames_input):
        shutil.copy(f, fi)
    
    prefix = '{}_data.combine'.format(sci_name)
    run_swarp(maindir, fnames_input, fname_default, prefix, swarp_type='data', ncpu=ncpu, verbose=verbose)
    
    for f in fnames_input:
        os.remove(f)
        
    ####################################################################################################
    
    #Run SWarp for masks
    for f, fi in zip(fnames_sci, fnames_input):
        fname_mask_i = f.replace('_data.fits', '_mask.fits')
        shutil.copy(fname_mask_i, fi)        
    
    
    prefix = '{}_mask.combine'.format(sci_name)
    run_swarp(maindir, fnames_input, fname_default, prefix, swarp_type='mask', ncpu=ncpu, verbose=verbose)
    
    for f in fnames_input:
        os.remove(f)
        
    ####################################################################################################
    
    #Run SWarp for error files
    for f, fi in zip(fnames_sci, fnames_input):
        fname_err_i = f.replace('_data.fits', '_err.fits')
        fname_var_i = f.replace('_data.fits', '_var.fits')
        
        #Turn error into variance
        hdul = fits.open(fname_err_i)
        hdul[0].data = hdul[0].data**2
        hdul.writeto(fname_var_i, overwrite=True)
        hdul.close()
        shutil.copy(fname_var_i, fi)
        

    prefix = '{}_var.combine'.format(sci_name)
    run_swarp(maindir, fnames_input, fname_default, prefix, swarp_type='var', ncpu=ncpu, verbose=verbose)
    
    for f in fnames_input:
        os.remove(f)
        
    ####################################################################################################
        
    #Delete weight files
    os.remove(maindir + '{}_data.combine.weight.fits'.format(sci_name))
    os.remove(maindir + '{}_mask.combine.weight.fits'.format(sci_name))
    os.remove(maindir + '{}_var.combine.weight.fits'.format(sci_name))

    ####################################################################################################
    
    #Convert var to error
    hdul = fits.open(maindir + '{}_var.combine.fits'.format(sci_name))  
    hdul[0].data = np.sqrt(hdul[0].data)
    hdul[0].data[ np.isnan(hdul[0].data) ] = 0.
    hdul.writeto(maindir + '{}_err.combine.fits'.format(sci_name), overwrite=True)
    hdul.close()
    os.remove(maindir + '{}_var.combine.fits'.format(sci_name))  
    
    
    #Remove var files
    for f in fnames_sci:
        fname_var_i = f.replace('_data.fits', '_var.fits')
        os.remove(fname_var_i)
        
    ####################################################################################################
    
    #Add MAG_ZP and FILTER keywords to header if they exist
    for suffix in ['data', 'mask', 'err']:    
        
        fnames_in = [f.replace('_data.fits', '_{}.fits'.format(suffix)) for f in fnames_sci]
        filters = []
        mag_zeros = []
        mag_zps = []
        for f in fnames_sci:
            hdr = fits.open(f)[0].header
            if 'FILTER' in hdr:
                filters.append(hdr['FILTER'])
            
            if 'MAGZERO' in hdr:
                mag_zeros.append(hdr['MAGZERO'])
                
            if 'MAG_ZP' in hdr:
                mag_zps.append(hdr['MAG_ZP'])
        
    

        fname = maindir + '{}_{}.combine.fits'.format(sci_name, suffix)
        hdul = fits.open(fname)

        dat = hdul[0].data        
        hdr = hdul[0].header

        if len(mag_zeros) > 0:
            assert np.all(np.array(mag_zeros) == mag_zeros[0])
            hdr['MAGZERO'] = mag_zeros[0]

        if len(mag_zps) > 0:
            assert np.all(np.array(mag_zps) == mag_zps[0])
            hdr['MAG_ZP'] = mag_zps[0]

        if len(filters) > 0:
            assert np.all(np.array(filters) == filters[0])
            hdr['FILTER'] = filters[0]
            
        fits.writeto(fname, dat, hdr, overwrite=True)
        hdul.close()

    return




def combine_psf_euclid(maindir, sci_name, ref_shape=(201,201), ncpu=1, verbose=False):
    
    fnames_gridpsf = glob.glob(maindir + '{}*_gridpsf.fits'.format(sci_name))
    
    #Get PSF images
    for f in fnames_gridpsf:
        prefix = '_'.join(  os.path.basename(f).split('_')[:-1]  )
        fout = maindir + '{}.psfin'.format(prefix)
        
        if not os.path.exists(fout):
            psfe.get_psf_all([f], fout)
    
        if os.path.exists(maindir + '{}_psf.fits'.format(prefix)) and os.path.exists(maindir + '{}_psf_err.fits'.format(prefix)):
            continue
        
        #Load PSF image
        im = fits.open(fout)[1].data        
        im_err = fits.open(fout)[2].data
        header = fits.open(fout)[1].header
        header_err = fits.open(fout)[2].header
        
        for h in [header, header_err]:            
            fout = maindir + '{}_psf.fits'.format(prefix)
            fits.writeto(fout, im, header, overwrite=True)
            
            fout = maindir + '{}_psf_err.fits'.format(prefix)
            fits.writeto(fout, im_err, header_err, overwrite=True)

    #########################################################################################  

    #Get PSF images
    for f in fnames_gridpsf:
        prefix = '_'.join(  os.path.basename(f).split('_')[:-1]  )
        fout = maindir + '{}.psfin'.format(prefix)
        
        if not os.path.exists(fout):
            psfe.get_psf_all([f], fout)
    
        if os.path.exists(maindir + '{}_psf.fits'.format(prefix)) and os.path.exists(maindir + '{}_psf_err.fits'.format(prefix)):
            continue
        
        #Load PSF image
        im = fits.open(fout)[1].data        
        im_err = fits.open(fout)[2].data
        header = fits.open(fout)[1].header
        header_err = fits.open(fout)[2].header
        
        N0 = im.shape[0]
        N1 = im.shape[1]
        for h in [header, header_err]:
            h['NAXIS1'] = N0
            h['NAXIS2'] = N1
            
            if N0 % 2 == 0:
                h['CRPIX1'] = N0 / 2 + 0.5
            else:
                h['CRPIX1'] = N0 // 2
                
            if N1 % 2 == 0:
                h['CRPIX2'] = N1 / 2 + 0.5
            else:
                h['CRPIX2'] = N1 // 2            

            h['CRVAL1'] = hdr_ref['CRVAL1']
            h['CRVAL2'] = hdr_ref['CRVAL2']
            h['CTYPE1'] = hdr_ref['CTYPE1']
            h['CTYPE2'] = hdr_ref['CTYPE2']
            h['CDELT1'] = .1/3600.
            h['CDELT2'] = .1/3600.
            h['PC1_1'] = -1.
            h['PC1_2'] = 0.
            h['PC2_1'] = 0.
            h['PC2_2'] = 1.
            
            fout = maindir + '{}_psf.fits'.format(prefix)
            fits.writeto(fout, im, header, overwrite=True)
            
            fout = maindir + '{}_psf_err.fits'.format(prefix)
            fits.writeto(fout, im_err, header_err, overwrite=True)

    #######################################################################################  
    
    
    if ref_shape[0] % 2 == 0:
        xc = ref_shape[0] / 2 + 0.5
    else:
        xc = ref_shape[0] // 2
        
    if ref_shape[1] % 2 == 0:
        yc = ref_shape[1] / 2 + 0.5
    else:
        yc = ref_shape[1] // 2
    
    #Combine PSF images
    fnames_psf = glob.glob(maindir + '{}*_psf.fits'.format(sci_name))
    
    psf_all = []
    for f in fnames_psf:
        psf_og = fits.open(f)[0].data
        
        #Pad PSF to ref_shape
        padx1 = int((ref_shape[0] - psf_og.shape[0]) / 2)
        padx2 = ref_shape[0] - psf_og.shape[0] - padx1
        pady1 = int((ref_shape[1] - psf_og.shape[1]) / 2)
        pady2 = ref_shape[1] - psf_og.shape[1] - pady1
        psf = np.pad(psf_og, ((padx1, padx2), (pady1, pady2)), mode='constant', constant_values=0.)
        
        #Get center of PSF
        max_ind = np.unravel_index(np.nanargmax(psf), psf.shape)

        #Shift PSF to center of frame
        if max_ind[0] != xc:
            psf = np.roll(psf, int(xc-max_ind[0]), axis=0)
            print('PSF shifted in x by {}'.format(int(xc-max_ind[0])))
        if max_ind[1] != yc:
            psf = np.roll(psf, int(yc-max_ind[1]), axis=1)
            print('PSF shifted in y by {}'.format(int(yc-max_ind[1])))
        
        max_ind_new = np.unravel_index(np.nanargmax(psf), psf.shape)
        assert max_ind_new[0] == xc
        assert max_ind_new[1] == yc

        psf_all.append(psf)
        
    psf_out = np.nanmedian(np.array(psf_all), axis=0)
    psf_out /= np.nansum(psf_out)
    psf_out[np.isnan(psf_out)] = 0.
    
    #Save
    fname_out = maindir + '{}_psf.fits'.format(sci_name)
    header = fits.open(fnames_psf[0])[0].header
    fits.writeto(fname_out, psf_out, header, overwrite=True)    

    return