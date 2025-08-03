import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u

import subprocess
import glob
import shutil
import os

mdir = '/home/stone28/projects/WebbDiffImg/pipeline/src/'

import sys
sys.path.append(mdir + 'psf/')
import psf_euclid as psfe



def get_ref_wcsfile(fname_ref, fname_wcs):
    hdr = fits.open(fname_ref)[0].header
    
    if (hdr['CTYPE1'] == 'RA---TAN') and ('PV1_0' in hdr):
        _hdr = hdr.copy()
        _hdr['CTYPE1'] = 'RA---TPV'
        _hdr['CTYPE2'] = 'DEC--TPV'
    else: 
        _hdr = hdr

    w_ref = WCS(_hdr)

    wcshdr_ref = w_ref.to_header(relax=True)   # SIP distorsion requires relax=True

    wcshdr_ref['BITPIX'] = hdr['BITPIX']
    wcshdr_ref['NAXIS'] = hdr['NAXIS']

    wcshdr_ref['NAXIS1'] = hdr['NAXIS1']
    wcshdr_ref['NAXIS2'] = hdr['NAXIS2']

    wcshdr_ref.tofile(fname_wcs, overwrite=True)
    return




def make_swarp_config(maindir, fname_swarp_default, prefix, fname_out, swarp_type='data', ncpu=1):

    with open(fname_swarp_default, 'r') as f:
        content = f.readlines()



    content[4]   = "IMAGEOUT_NAME          {}.fits          # Output filename\n".format(maindir+prefix)
    content[5]   = "WEIGHTOUT_NAME         {}.weight.fits   # Output weight-map filename\n".format(maindir+prefix)

    if swarp_type in ['data', 'mask', 'psf']:
        content[12]  = "WEIGHT_TYPE            MAP_RMS         # BACKGROUND,MAP_RMS,MAP_VARIANCE\n"    
    else:
        content[12]  = "WEIGHT_TYPE            NONE            # BACKGROUND,MAP_RMS,MAP_VARIANCE\n"

    if swarp_type in ['psf_jwst']:
        content[14]  = "RESCALE_WEIGHTS        Y               # Rescale input weights/variances (Y/N)?\n"
    else:
        content[14]  = "RESCALE_WEIGHTS        N               # Rescale input weights/variances (Y/N)?\n"

    content[15]  = "WEIGHT_SUFFIX          _err.fits        # Suffix to use for weight-maps\n"

    content[22]  = "COMBINE                Y               # Combine resampled images (Y/N)?\n"

    if swarp_type in ['']:
        content[23]  = "COMBINE_TYPE           OR             # MEDIAN,AVERAGE,MIN,MAX,WEIGHTED,CLIPPED\n"
    elif swarp_type in ['data', 'psf']:
        content[23]  = "COMBINE_TYPE           MEDIAN          # MEDIAN,AVERAGE,MIN,MAX,WEIGHTED,CLIPPED\n"
    elif swarp_type in ['var', 'mask']:
        content[23]  = "COMBINE_TYPE           SUM             # MEDIAN,AVERAGE,MIN,MAX,WEIGHTED,CLIPPED\n"

    content[76]  = "SUBTRACT_BACK          N               # Subtraction sky background (Y/N)?\n"

    content[113] = "NTHREADS               {}              # Number of simultaneous threads for\n".format(ncpu)

    f = open(fname_out, 'w+')
    f.writelines(content)
    f.close()

    return



def run_swarp(maindir, fname_ref, fnames_sci, fname_default, prefix, swarp_type='data', ncpu=1, verbose=False):
    
    #Make WCS file
    fname_wcs = maindir + '{}.head'.format(prefix)
    get_ref_wcsfile(fname_ref, fname_wcs)
    
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





def align_frames(maindir, fname_default, ref_name, sci_name, masktype='swarp', ncpu=1, verbose=False):
    
    fname_ref = maindir + '{}_data.fits'.format(ref_name)
    fnames_sci = glob.glob(maindir + '{}*_data.fits'.format(sci_name))
    
    for f in fnames_sci:
        fname_err_i = f.replace('_data.fits', '_err.fits')
        fname_mask_i = f.replace('_data.fits', '_mask.fits')
        
        if not os.path.exists(fname_err_i):
            raise FileNotFoundError('Error file {} not found'.format(fname_err_i))
        
        if not os.path.exists(fname_mask_i):
            raise FileNotFoundError('Mask file {} not found'.format(fname_mask_i))


    fnames_input = [f.replace('_data', '') for f in fnames_sci]
    
    ####################################################################################################

    #Run SWarp for data images
    for f, fi in zip(fnames_sci, fnames_input):
        shutil.copy(f, fi)
    
    prefix = '{}_data.align'.format(sci_name)
    run_swarp(maindir, fname_ref, fnames_input, fname_default, prefix, swarp_type='data', ncpu=ncpu, verbose=verbose)
    
    for f in fnames_input:
        os.remove(f)
        
    ####################################################################################################
    
    if masktype == 'swarp':    
        #Run SWarp for masks
        for f, fi in zip(fnames_sci, fnames_input):
            fname_mask_i = f.replace('_data.fits', '_mask.fits')
            shutil.copy(fname_mask_i, fi)        
        
        
        prefix = '{}_mask.align'.format(sci_name)
        run_swarp(maindir, fname_ref, fnames_input, fname_default, prefix, swarp_type='mask', ncpu=ncpu, verbose=verbose)
        
        for f in fnames_input:
            os.remove(f)
            
    elif masktype == 'manual':
        f_ref = maindir + '{}_data.fits'.format(ref_name)
        with fits.open(f_ref) as hdul:
            im_r = hdul[0].data
            hdr_r = hdul[0].header
            
        m1 = (im_r == 0.) | np.isnan(im_r)
        
        
        
        f_sci = maindir + '{}_data.align.fits'.format(sci_name)
        with fits.open(f_sci) as hdul:
            im_s = hdul[0].data
            hdr_s = hdul[0].header
            
        m2 = (im_s == 0.) | np.isnan(im_s)



        fout = maindir + '{}_mask.align.fits'.format(sci_name)
        fits.writeto(fout, m2.astype(np.int16), hdr_r, overwrite=True)
        
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
        

    prefix = '{}_var.align'.format(sci_name)
    run_swarp(maindir, fname_ref, fnames_input, fname_default, prefix, swarp_type='var', ncpu=ncpu, verbose=verbose)
    
    for f in fnames_input:
        os.remove(f)
        
    ####################################################################################################
        
    #Delete weight files
    os.remove(maindir + '{}_data.align.weight.fits'.format(sci_name))
    os.remove(maindir + '{}_var.align.weight.fits'.format(sci_name))
    if masktype == 'swarp':
        os.remove(maindir + '{}_mask.align.weight.fits'.format(sci_name))

    ####################################################################################################
    
    #Convert var to error
    hdul = fits.open(maindir + '{}_var.align.fits'.format(sci_name))  
    hdul[0].data = np.sqrt(hdul[0].data)
    hdul[0].data[ np.isnan(hdul[0].data) ] = 0.
    hdul.writeto(maindir + '{}_err.align.fits'.format(sci_name), overwrite=True)
    hdul.close()
    os.remove(maindir + '{}_var.align.fits'.format(sci_name)) 
    
    #Remove var files
    for f in fnames_sci:
        fname_var_i = f.replace('_data.fits', '_var.fits')
        os.remove(fname_var_i)
        
        
    #Convert masksum to mask
    # hdul = fits.open(maindir + '{}_mask.align.fits'.format(sci_name))
    # mask = hdul[0].data
    # mask[mask > 0] = 1
    # mask[np.isnan(mask)] = 0.
    # hdul[0].data = mask
    # hdul.writeto(maindir + '{}_mask.align.fits'.format(sci_name), overwrite=True)
    # hdul.close()
        
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
        
    
    
        fname = maindir + '{}_{}.align.fits'.format(sci_name, suffix)
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







def align_psf_euclid(maindir, fname_default, ref_name, sci_name, ref_shape=(201,201), ncpu=1, verbose=False):
    
    fname_ref = maindir + '{}_data.fits'.format(ref_name)
    fnames_gridpsf = glob.glob(maindir + '{}*_gridpsf.fits'.format(sci_name))
    fnames_input = [f.replace('_gridpsf', '') for f in fnames_gridpsf]
    
    
    hdr_ref = fits.open(fname_ref)[0].header

    #######################################################################################
    #Make the output PSF grid with the reference image WCS
    
    header = fits.open(fname_ref)[0].header
    header['NAXIS1'] = ref_shape[0]
    header['NAXIS2'] = ref_shape[1]
    
    if ref_shape[0] % 2 == 0:
        header['CRPIX1'] = ref_shape[0] / 2 + 0.5
    else:
        header['CRPIX1'] = ref_shape[0] // 2
        
    if ref_shape[1] % 2 == 0:
        header['CRPIX2'] = ref_shape[1] / 2 + 0.5
    else:
        header['CRPIX2'] = ref_shape[1] // 2
        
    header['CRVAL1'] = hdr_ref['CRVAL1']
    header['CRVAL2'] = hdr_ref['CRVAL2']
    header['CTYPE1'] = hdr_ref['CTYPE1']
    header['CTYPE2'] = hdr_ref['CTYPE2']
    header['CDELT1'] = hdr_ref['CDELT1']
    header['CDELT2'] = hdr_ref['CDELT2']

    if 'PC1_1' in hdr_ref:
        header['PC1_1'] = hdr_ref['PC1_1']
    else:
        header['PC1_1'] = 0.
    
    if 'PC1_2' in hdr_ref:
        header['PC1_2'] = hdr_ref['PC1_2']
    else:
        header['PC1_2'] = 0.

    if 'PC2_1' in hdr_ref:
        header['PC2_1'] = hdr_ref['PC2_1']
    else:
        header['PC2_1'] = 0.
    
    if 'PC2_2' in hdr_ref:
        header['PC2_2'] = hdr_ref['PC2_2']
    else:
        header['PC2_2'] = 1.
    
    
    im = np.zeros(ref_shape)
    fname_ref_empty = maindir + 'ref_empty.fits'
    fits.writeto(maindir + 'ref_empty.fits', im, header, overwrite=True)
    
    #######################################################################################

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

    #Run SWarp for PSF images
    for f, fi in zip(fnames_gridpsf, fnames_input):
        prefix = '_'.join(  os.path.basename(f).split('_')[:-1]  )
        prefix_out = prefix + '.align'
    
        fname_in = maindir + '{}_psf.fits'.format(prefix)        
        run_swarp(maindir, fname_ref_empty, [fname_in], fname_default, prefix_out, swarp_type='psf', ncpu=ncpu, verbose=verbose)
        
        #Remove weight file
        os.remove(maindir + '{}.align.weight.fits'.format(prefix))

        #Rename output file
        shutil.move(maindir + '{}.align.fits'.format(prefix), maindir + '{}_psf.align.fits'.format(prefix))


    #########################################################################################
    
    if ref_shape[0] % 2 == 0:
        xc = ref_shape[0] / 2 + 0.5
    else:
        xc = ref_shape[0] // 2
        
    if ref_shape[1] % 2 == 0:
        yc = ref_shape[1] / 2 + 0.5
    else:
        yc = ref_shape[1] // 2
    
    
    
    
    #Combine PSF images
    fnames_psf = glob.glob(maindir + '{}*_psf.align.fits'.format(sci_name))
    
    psf_all = []
    for f in fnames_psf:
        psf = fits.open(f)[0].data
        max_ind = np.unravel_index(np.nanargmax(psf), psf.shape)

        #Shift PSF to center
        if max_ind[0] != xc:
            psf = np.roll(psf, int(xc-max_ind[0]), axis=0)
        if max_ind[1] != yc:
            psf = np.roll(psf, int(yc-max_ind[1]), axis=1)
        
        max_ind_new = np.unravel_index(np.nanargmax(psf), psf.shape)
        assert max_ind_new[0] == xc
        assert max_ind_new[1] == yc

        psf_all.append(psf)
        
    psf_out = np.nanmedian(np.array(psf_all), axis=0)
    psf_out /= np.nansum(psf_out)
    psf_out[np.isnan(psf_out)] = 0.
    
    #Save
    fname_out = maindir + '{}_psf.align.fits'.format(sci_name)
    header = fits.open(fnames_psf[0])[0].header
    fits.writeto(fname_out, psf_out, header, overwrite=True)    

    return







def align_psf_jwst(maindir, outdir, fname_default, ref_name, sci_name, ref_shape=(201,201), ncpu=1, verbose=False):
    
    fname_ref = maindir + '{}_data.fits'.format(ref_name)
    fname_psfin = maindir + '{}_psfin.fits'.format(sci_name)
    
    
    hdr_ref = fits.open(fname_ref)[0].header

    #######################################################################################
    #Make the output PSF grid with the reference image WCS
    
    header = fits.open(fname_ref)[0].header
    
    n = 1
    while n > 0:
        
        n = 0
        for i, k in enumerate(header.keys()):
            if k in ['SIMPLE', 'BITPIX', 'PSF_FWHM', 'PS', 'NAXIS']:
                continue
            else:
                n += 1
                del header[i]
    
    header['NAXIS1'] = ref_shape[0]
    header['NAXIS2'] = ref_shape[1]
    
    if ref_shape[0] % 2 == 0:
        header['CRPIX1'] = float(   ref_shape[0] / 2 + 0.5   )
    else:
        header['CRPIX1'] = float(   ref_shape[0] // 2   )
        
    if ref_shape[1] % 2 == 0:
        header['CRPIX2'] = float(   ref_shape[1] / 2 + 0.5   )
    else:
        header['CRPIX2'] = float(   ref_shape[1] // 2   )
        
    header['CRVAL1'] = hdr_ref['CRVAL1']
    header['CRVAL2'] = hdr_ref['CRVAL2']
    header['CTYPE1'] = hdr_ref['CTYPE1']
    header['CTYPE2'] = hdr_ref['CTYPE2']
    header['CDELT1'] = hdr_ref['CD2_2']
    header['CDELT2'] = hdr_ref['CD2_2']
    header['CD1_1'] = np.sign(hdr_ref['CD1_1'])
    header['CD1_2'] = np.sign(hdr_ref['CD1_2'])
    header['CD2_1'] = np.sign(hdr_ref['CD2_1'])
    header['CD2_2'] = np.sign(hdr_ref['CD2_2'])
    
    im = np.zeros(ref_shape)
    fname_ref_empty = outdir + 'ref_empty.fits'
    fits.writeto(fname_ref_empty, im, header=header, overwrite=True)
    
    #######################################################################################

    #Get PSF image (JWST)
    
    #Load PSF image
    im = fits.open(fname_psfin)[0].data        
    header = fits.open(fname_psfin)[0].header
    
    n = 1
    while n > 0:
        
        n = 0
        for i, k in enumerate(header.keys()):
            if k in ['SIMPLE', 'BITPIX', 'PSF_FWHM', 'PS', 'NAXIS']:
                continue
            else:
                n += 1
                del header[i]
    
    
    N0, N1 = ref_shape
    
    #JWST PS is 3x smaller than Euclid, so make input PSF image 3x larger
    if N0 % 2 == 0:
        N0 *= 3
    else:
        N0 = (N0-1)*3 + 1
        
    if N1 % 2 == 0:
        N1 *= 3
    else:
        N1 = (N1-1)*3 + 1
        

    
    if (N0 - im.shape[0]) % 2 == 0:
        val = int(  (N0 - im.shape[0]) / 2  )
        padx = (val, val)
    else:
        val = int(  (N0 - im.shape[0]) / 2  )
        padx = (val, val+1)
        
    if (N1 - im.shape[1]) % 2 == 0:
        val = int(  (N1 - im.shape[1]) / 2  )
        pady = (val, val)
    else:
        val = int(  (N1 - im.shape[1]) / 2  )
        pady = (val, val+1)
        
        

    im = np.pad(im, (padx, pady), mode='constant', constant_values=0.) 


    header['NAXIS1'] = N0
    header['NAXIS2'] = N1
    
    if N0 % 2 == 0:
        header['CRPIX1'] = float(  N0 / 2 + 0.5   )
    else:
        header['CRPIX1'] = float(  N0 // 2   )
        
    if N1 % 2 == 0:
        header['CRPIX2'] = float(  N1 / 2 + 0.5   )
    else:
        header['CRPIX2'] = float(  N1 // 2   )       

    header['CRVAL1'] = hdr_ref['CRVAL1']
    header['CRVAL2'] = hdr_ref['CRVAL2']
    header['CTYPE1'] = hdr_ref['CTYPE1']
    header['CTYPE2'] = hdr_ref['CTYPE2']
    header['CDELT1'] = header['PS']/3600.
    header['CDELT2'] = header['PS']/3600.
    header['CD1_1'] = np.sign(hdr_ref['CD1_1'])
    header['CD1_2'] = np.sign(hdr_ref['CD1_2'])
    header['CD2_1'] = np.sign(hdr_ref['CD2_1'])
    header['CD2_2'] = np.sign(hdr_ref['CD2_2'])
    
    fout = outdir + '{}_psf.fits'.format(sci_name)
    fits.writeto(fout, im, header=header, overwrite=True)
            
    #######################################################################################

    #Run SWarp for PSF image
    prefix_out = sci_name + '.align'

    fname_in = outdir + '{}_psf.fits'.format(sci_name)        
    run_swarp(outdir, fname_ref_empty, [fname_in], fname_default, prefix_out, swarp_type='psf_jwst', ncpu=ncpu, verbose=verbose)
    
    #Remove weight file
    os.remove(outdir + '{}.align.weight.fits'.format(sci_name))

    #Rename output file
    shutil.move(outdir + '{}.align.fits'.format(sci_name), outdir + '{}_psf.align.fits'.format(sci_name))

    #########################################################################################
    
    if ref_shape[0] % 2 == 0:
        xc = ref_shape[0] / 2 + 0.5
    else:
        xc = ref_shape[0] // 2
        
    if ref_shape[1] % 2 == 0:
        yc = ref_shape[1] / 2 + 0.5
    else:
        yc = ref_shape[1] // 2
    
    
    #Center and normalize PSF image
    fname_psf = outdir + '{}_psf.align.fits'.format(sci_name)
    
    psf = fits.open(fname_psf)[0].data
    max_ind = np.unravel_index(np.nanargmax(psf), psf.shape)

    print('xshift ', int(xc-max_ind[0]))
    print('yshift ', int(yc-max_ind[1]))

    #Shift PSF to center
    if max_ind[0] != xc:
        psf = np.roll(psf, int(xc-max_ind[0]), axis=0)
    if max_ind[1] != yc:
        psf = np.roll(psf, int(yc-max_ind[1]), axis=1)
    
    max_ind_new = np.unravel_index(np.nanargmax(psf), psf.shape)
    assert max_ind_new[0] == xc
    assert max_ind_new[1] == yc
        
    psf_out = psf.copy()
    psf_out /= np.nansum(psf_out)
    psf_out[np.isnan(psf_out)] = 0.
    
    #Save
    fname_out = outdir + '{}_psf.align.fits'.format(sci_name)
    header = fits.open(fname_psf)[0].header
    fits.writeto(fname_out, psf_out, header, overwrite=True)    

    return







def shift_center(maindir, name_large, name_small, 
                 prefix_large, prefix_small, 
                 prefix_out, outdir):
    
    fname_l = maindir + '{}_data.{}fits'.format(name_large, prefix_large)
    fname_s = maindir + '{}_data.{}fits'.format(name_small, prefix_small)
    
    #Load small image
    im_s = fits.open(fname_s)[0].data
    hdr_l = fits.open(fname_l)[0].header
    wcs_l = WCS(hdr_l)
    old_shape = im_s.shape
    
    
    #Find bounds
    nonzero_ind = np.argwhere(im_s != 0)
    y1 = nonzero_ind[:,0].min()-1
    y2 = nonzero_ind[:,0].max()+1
    x1 = nonzero_ind[:,1].min()-1
    x2 = nonzero_ind[:,1].max()+1
    
    if y1 < 0:
        y1 = 0
    if y2 > old_shape[0]:
        y2 = old_shape[0]
    if x1 < 0:
        x1 = 0
    if x2 > old_shape[1]:
        x2 = old_shape[1]
    
    
    #Find new center (px)
    new_shape = (y2-y1, x2-x1)
    
    if new_shape[0] % 2 == 0:
        yc_new = new_shape[0] / 2 + 0.5
    else:
        yc_new = new_shape[0] // 2
        
    if new_shape[1] % 2 == 0:
        xc_new = new_shape[1] / 2 + 0.5
    else:
        xc_new = new_shape[1] // 2
    
    
    
    #Find new RA/DEC center
    rac_new, decc_new = wcs_l.array_index_to_world_values(y1 + yc_new, x1 + xc_new)
    dx = 2*np.ceil( (x2-x1)/2 +1 )
    dy = 2*np.ceil( (y2-y1)/2 +1 )
    
    pos = SkyCoord(float(rac_new), float(decc_new), unit=(u.deg, u.deg), frame='icrs')
    shape = np.array([dy, dx]) * u.pixel
    
    for suffix in ['data', 'err', 'mask']:
        
        #Load small image
        fname = maindir + '{}_{}.{}fits'.format(name_small, suffix, prefix_small)
        im_s = fits.open(fname)[0].data
        cutout_s = Cutout2D(im_s, pos, shape, wcs=wcs_l)
    
        #Save shifted image
        fname_out = outdir + '{}_{}.{}.fits'.format(name_small, suffix, prefix_out)
        fits.writeto(fname_out, cutout_s.data, header=cutout_s.wcs.to_header(), overwrite=True)        
        
        
        
        #Load large image
        fname = maindir + '{}_{}.{}fits'.format(name_large, suffix, prefix_large)
        im_l = fits.open(fname)[0].data
        cutout_l = Cutout2D(im_l, pos, shape, wcs=wcs_l)
        
        #Save shifted image
        fname_out = outdir + '{}_{}.{}.fits'.format(name_large, suffix, prefix_out)
        fits.writeto(fname_out, cutout_l.data, header=cutout_l.wcs.to_header(), overwrite=True)
        

    return
    
    
    
###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################

def align_psf(fname_dat, fname_psf, outdir, prefix, fname_default, ref_shape=(201,201), ncpu=1, verbose=False):
    
    
    hdr_ref = fits.open(fname_dat)[0].header
    
    im_psf = fits.open(fname_psf)[0].data        
    hdr_psf = fits.open(fname_psf)[0].header
    
    if ref_shape is None:
        ref_shape = im_psf.shape

    #######################################################################################
    #Make the output PSF grid with the reference image WCS
    
    header = fits.open(fname_dat)[0].header
    
    n = 1
    while n > 0:

        n = 0
        for i, k in enumerate(header.keys()):
            if k in ['SIMPLE', 'BITPIX', 'PSF_FWHM', 'PS', 'NAXIS']:
                continue
            else:
                n += 1
                del header[i]
    
    header['NAXIS1'] = ref_shape[0]
    header['NAXIS2'] = ref_shape[1]
    
    if ref_shape[0] % 2 == 0:
        header['CRPIX1'] = float(   ref_shape[0] / 2 + 0.5   )
    else:
        header['CRPIX1'] = float(   ref_shape[0] // 2   )
        
    if ref_shape[1] % 2 == 0:
        header['CRPIX2'] = float(   ref_shape[1] / 2 + 0.5   )
    else:
        header['CRPIX2'] = float(   ref_shape[1] // 2   )
        
    header['CRVAL1'] = hdr_ref['CRVAL1']
    header['CRVAL2'] = hdr_ref['CRVAL2']
    header['CTYPE1'] = hdr_ref['CTYPE1']
    header['CTYPE2'] = hdr_ref['CTYPE2']
    header['CDELT1'] = hdr_ref['CDELT1']
    header['CDELT2'] = hdr_ref['CDELT2']
    
    if 'PC1_1' in hdr_ref:
        header['PC1_1'] = hdr_ref['PC1_1']
    else:
        header['PC1_1'] = 0.
    
    if 'PC1_2' in hdr_ref:
        header['PC1_2'] = hdr_ref['PC1_2']
    else:
        header['PC1_2'] = 0.

    if 'PC2_1' in hdr_ref:
        header['PC2_1'] = hdr_ref['PC2_1']
    else:
        header['PC2_1'] = 0.
    
    if 'PC2_2' in hdr_ref:
        header['PC2_2'] = hdr_ref['PC2_2']
    else:
        header['PC2_2'] = 1.
    
    im = np.zeros(ref_shape)
    fname_ref_empty = outdir + 'ref_empty.fits'
    fits.writeto(fname_ref_empty, im, header=header, overwrite=True)
    
    #######################################################################################

    #Get PSF image (JWST)
    
    #Load PSF image
    im = fits.open(fname_psf)[0].data        
    header = fits.open(fname_psf)[0].header
    
    n = 1
    while n > 0:
        
        n = 0
        for i, k in enumerate(header.keys()):
            if k in ['SIMPLE', 'BITPIX', 'PSF_FWHM', 'PS', 'NAXIS']:
                continue
            else:
                n += 1
                del header[i]
    
    
    # N0, N1 = ref_shape
    
    # if im.shape == ref_shape:
    #     pass
    # elif N0 > im.shape[0] or N1 > im.shape[1]:
    #     if (N0 - im.shape[0]) % 2 == 0:
    #         val = int(  (N0 - im.shape[0]) / 2  )
    #         padx = (val, val)
    #     else:
    #         val = int(  (N0 - im.shape[0]) / 2  )
    #         padx = (val, val+1)
            
    #     if (N1 - im.shape[1]) % 2 == 0:
    #         val = int(  (N1 - im.shape[1]) / 2  )
    #         pady = (val, val)
    #     else:
    #         val = int(  (N1 - im.shape[1]) / 2  )
    #         pady = (val, val+1)
    
    #     im = np.pad(im, (padx, pady), mode='constant', constant_values=0.) 
            
    # else:
    #     #Crop image
    #     val = int(  (im.shape[0] - N0) / 2  )
    #     padx = (val, val)
    #     val = int(  (im.shape[1] - N1) / 2  )
    #     pady = (val, val+1)
        
    #     im = im[padx[0]:-padx[1], pady[0]:-pady[1]].copy()
    
    N0, N1 = im.shape


    header['NAXIS1'] = N0
    header['NAXIS2'] = N1
    
    if N0 % 2 == 0:
        header['CRPIX1'] = float(  N0 / 2 + 0.5   )
    else:
        header['CRPIX1'] = float(  N0 // 2   )
        
    if N1 % 2 == 0:
        header['CRPIX2'] = float(  N1 / 2 + 0.5   )
    else:
        header['CRPIX2'] = float(  N1 // 2   )       

    header['CRVAL1'] = hdr_ref['CRVAL1']
    header['CRVAL2'] = hdr_ref['CRVAL2']
    header['CTYPE1'] = hdr_ref['CTYPE1']
    header['CTYPE2'] = hdr_ref['CTYPE2']
    header['CD1_1'] = hdr_psf['CD1_1']
    header['CD1_2'] = hdr_psf['CD1_2']
    header['CD2_1'] = hdr_psf['CD2_1']
    header['CD2_2'] = hdr_psf['CD2_2']
    
    fout = outdir + '{}_psf.fits'.format(prefix)
    fits.writeto(fout, im, header=header, overwrite=True)
            
    #######################################################################################

    #Run SWarp for PSF image
    prefix_out = prefix + '.align'

    fname_in = outdir + '{}_psf.fits'.format(prefix)        
    run_swarp(outdir, fname_ref_empty, [fname_in], fname_default, prefix_out, swarp_type='psf_jwst', ncpu=ncpu, verbose=verbose)
    
    #Remove weight file
    os.remove(outdir + '{}.align.weight.fits'.format(prefix))

    #Rename output file
    shutil.move(outdir + '{}.align.fits'.format(prefix), outdir + '{}_psf.align.fits'.format(prefix))

    #########################################################################################
    
    if ref_shape[0] % 2 == 0:
        xc = ref_shape[0] / 2 + 0.5
    else:
        xc = ref_shape[0] // 2
        
    if ref_shape[1] % 2 == 0:
        yc = ref_shape[1] / 2 + 0.5
    else:
        yc = ref_shape[1] // 2
    
    
    #Center and normalize PSF image
    fname_psf = outdir + '{}_psf.align.fits'.format(prefix)
    
    psf = fits.open(fname_psf)[0].data
    max_ind = np.unravel_index(np.nanargmax(psf), psf.shape)

    print('xshift ', int(xc-max_ind[0]))
    print('yshift ', int(yc-max_ind[1]))

    #Shift PSF to center
    if max_ind[0] != xc:
        psf = np.roll(psf, int(xc-max_ind[0]), axis=0)
    if max_ind[1] != yc:
        psf = np.roll(psf, int(yc-max_ind[1]), axis=1)
    
    max_ind_new = np.unravel_index(np.nanargmax(psf), psf.shape)
    assert max_ind_new[0] == xc
    assert max_ind_new[1] == yc
        
    psf_out = psf.copy()
    psf_out /= np.nansum(psf_out)
    psf_out[np.isnan(psf_out)] = 0.
    
    #Save
    fname_out = outdir + '{}_psf.align.fits'.format(prefix)
    header = fits.open(fname_psf)[0].header
    fits.writeto(fname_out, psf_out, header, overwrite=True)    

    return