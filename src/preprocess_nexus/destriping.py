import os
import glob
import shutil
import logging
import subprocess
from mpire import WorkerPool
from functools import partial

import numpy as np
from datetime import datetime
from astropy.io import fits
import astropy.stats as astrostats
from astropy.convolution import Ring2DKernel, Gaussian2DKernel, convolve_fft
from scipy.optimize import curve_fit

from photutils.segmentation import make_2dgaussian_kernel, detect_threshold, detect_sources
from photutils.utils import circular_footprint

# jwst-related imports
from jwst.datamodels import ImageModel, FlatModel, dqflags
from jwst.flatfield.flat_field import do_correction
from stdatamodels import util
import crds
# Pipeline 
log = logging.getLogger('stpipe')
log.setLevel(logging.CRITICAL)
log.disabled = True
# After first call to a jwst module, all logging will appear to come from 
# stpipe, and use the stpipe configuration/format.

log = logging.getLogger('CRDS')
log.disabled = True


def get_imname(f):
    list_of_str = os.path.basename(f).split('.')[0].split('_')
    return '_'.join(list_of_str[:4])


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)  

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


####################################################################################################
#Getting the source mask before destriping

def get_source_mask(im, dq_mask, nsig=2, kernel_fwhm=3, dilation_radius=8, bkg_rms=None):

    dat = im.copy()
    dat[np.isnan(dat)] = 0.
    
    if kernel_fwhm == 3:
        kernel = make_2dgaussian_kernel(3.0, size=5)
    else:
        kernel = make_2dgaussian_kernel(kernel_fwhm, size=int(2*kernel_fwhm-1))
        
    dat_conv = convolve_fft(dat, kernel, allow_huge=True)
    if bkg_rms is None:
        threshold = detect_threshold(dat, nsigma=nsig, mask=dq_mask)
    else:
        threshold = nsig*bkg_rms
    
    dat_seg = detect_sources(dat_conv, threshold, npixels=10, mask=dq_mask)
    footprint = circular_footprint(radius=dilation_radius)  # dilate the mask using a circular footprint with a radius=5 pixel
    
    try:
        source_mask = dat_seg.make_source_mask(footprint=footprint)
    except AttributeError:
        source_mask = np.zeros_like(dat, dtype=np.int8).astype(bool)

    return source_mask


####################################################################################################
#Destriping
#All from: https://github.com/ceers/ceers-nircam/blob/main/remstriping.py

### from jwst/refpix/reference_pixels.py:
# NIR Reference section dictionaries are zero indexed and specify the values
# to be used in the following slice: (rowstart: rowstop, colstart:colstop)
# The 'stop' values are one more than the actual final row or column, in
# accordance with how Python slices work
NIR_reference_sections = {'A': {'top': (2044, 2048, 0, 512),
                                'bottom': (0, 4, 0, 512),
                                'side': (0, 2048, 0, 4),
                                'data': (0, 2048, 0, 512)},
                          'B': {'top': (2044, 2048, 512, 1024),
                                'bottom': (0, 4, 512, 1024),
                                'data': (0, 2048, 512, 1024)},
                          'C': {'top': (2044, 2048, 1024, 1536),
                                'bottom': (0, 4, 1024, 1536),
                                'data': (0, 2048, 1024, 1536)},
                          'D': {'top': (2044, 2048, 1536, 2048),
                                'bottom': (0, 4, 1536, 2048),
                                'side': (0, 2048, 2044, 2048),
                                'data': (0, 2048, 1536, 2048)}
                         }

### taking the reference rows/columns into account
NIR_amps = {'A': {'data': (4, 2044, 4, 512)},
            'B': {'data': (4, 2044, 512, 1024)},
            'C': {'data': (4, 2044, 1024, 1536)},
            'D': {'data': (4, 2044, 1536, 2044)}
            }


MASKTHRESH = 0.8

def gaussian(x, a, mu, sig):
    return a * np.exp(-(x-mu)**2/(2*sig**2))


def fit_sky(data):
    """Fit distribution of sky fluxes with a Gaussian"""
    bins = np.arange(-1, 1.5, 0.001)
    h,b = np.histogram(data, bins=bins)
    bc = 0.5 * (b[1:] + b[:-1])
    binsize = b[1] - b[0]

    p0 = [10, bc[np.argmax(h)], 0.01]
    popt,pcov = curve_fit(gaussian, bc, h, p0=p0)

    return popt[1]


def collapse_image(im, mask, dimension='y', sig=2.):
    """collapse an image along one dimension to check for striping.

    By default, collapse columns to show horizontal striping (collapsing
    along columns). Switch to vertical striping (collapsing along rows)
    with dimension='x' 

    Striping is measured as a sigma-clipped median of all unmasked pixels 
    in the row or column.

    Args:
        im (float array): image data array
        mask (bool array): image mask array, True where pixels should be 
            masked from the fit (where DQ>0, source flux has been masked, etc.)
        dimension (Optional [str]): specifies which dimension along which 
            to collapse the image. If 'y', collapses along columns to 
            measure horizontal striping. If 'x', collapses along rows to 
            measure vertical striping. Default is 'y'
        sig (Optional [float]): sigma to use in sigma clipping
    """
    # axis=1 results in array along y
    # axis=0 results in array along x
    if dimension == 'y':
        res = astrostats.sigma_clipped_stats(im, mask=mask, sigma=sig, 
                                             cenfunc='median',
                                             stdfunc='std', axis=1)
    elif dimension == 'x':
        res = astrostats.sigma_clipped_stats(im, mask=mask, sigma=sig, 
                                             cenfunc='median',
                                             stdfunc='std', axis=0)

    return res[1]



def measure_fullimage_striping(fitdata, mask):
    """Measures striping in countrate images using the full rows.

    Measures the horizontal & vertical striping present across the 
    full image. The full image median will be used for amp-rows that
    are entirely or mostly masked out.

    Args:
        fitdata (float array): image data array for fitting
        mask (bool array): image mask array, True where pixels should be 
            masked from the fit (where DQ>0, source flux has been masked, etc.)

    Returns:
        (horizontal_striping, vertical_striping): 
    """

    # fit horizontal striping, collapsing along columns
    horizontal_striping = collapse_image(fitdata, mask, dimension='y')
    # remove horizontal striping, requires taking transpose of image
    temp_image = fitdata.T - horizontal_striping
    # transpose back
    temp_image2 = temp_image.T

    # fit vertical striping, collapsing along rows
    vertical_striping = collapse_image(temp_image2, mask, dimension='x')

    return horizontal_striping, vertical_striping












def measure_striping(fname_in, fname_out, fname_out_mask, fname_out_bkg,
                     fname_out_v, fname_out_h, logger, 
                     thresh=None, apply_flat=True, mask_sources=True, 
                     save_patterns=False):
    
    """Removes striping in rate.fits files before flat fielding.

    Measures and subtracts the horizontal & vertical striping present in 
    countrate images. The striping is most likely due to 1/f noise, and 
    the RefPixStep with odd_even_columns=True and use_side_ref_pixels=True
    does not fully remove the pattern, no matter what value is chosen for 
    side_smoothing_length. There is also residual vertical striping in NIRCam 
    images simulated with Mirage.

    The measurement/subtraction is done along one axis at a time, since 
    the measurement along x will depend on what has been subtracted from y.
    

    Args:
        fname_in (str): input rate image filename, including full relative path
        fname_out (str): output destriped image filename
        fname_out_v (str): output vertical striping pattern filename
        fname_out_h (str): output horizontal striping pattern filename
        thresh (Optional [float]): fraction of masked amp-row pixels above 
            which full row fit is used
        apply_flat (Optional [bool]): if True, identifies and applies the 
            corresponding flat field before measuring striping pattern. 
            Applying the flat first allows for a cleaner measure of the 
            striping, especially for the long wavelength detectors. 
            Default is True.
        mask_sources (Optional [bool]): If True, masks out sources in image
            before measuring the striping pattern so that source flux is 
            not included in the calculation of the sigma-clipped median.
            Sources are identified using the Mirage seed images.
            Default is True.
        mask_fname (Optional [str]): filename of source mask to use. If
            mask_sources is True, must provide a source mask filename. If 
            set to None, mask_sources is set to False.
            Default is None.
        save_patterns (Optional [bool]): if True, saves the horizontal and
            vertical striping patterns to files called *horiz.fits and 
            *vert.fits, respectively
    """
    logger.info('Starting destriping')
    
    try:
        crds_context = os.environ['CRDS_CONTEXT']
    except KeyError:
        crds_context = crds.get_default_context()

        
    # if mask_sources is True, must provide a source mask filename
    # if not, set mask_sources to False
    if (mask_fname is None) and mask_sources:
        logger.error('Must provide a source mask filename if mask_sources=True. Assuming mask_sources=False')
        mask_sources = False
        
    if mask_sources and (not os.path.exists(mask_fname)):
        logger.error('Source mask %s not found. Assuming mask_sources=False'%mask_fname)
        mask_sources = False
    
    # if thresh is not defined by user, use global default
    if thresh is None:
        thresh = 0.4

    # set up output filename, this will also be used for saving 
    # other outputs like the source mask and striping patterns
    #####outputbase = os.path.join(OUTPUTDIR, os.path.basename(fname_in))

    model = ImageModel(fname_in)
    logger.info('Measuring image striping')
    logger.info('Working on %s'%os.path.basename(fname_in))

    # apply the flat to get a cleaner meausurement of the striping
    if apply_flat:
        logger.info('Applying flat for cleaner measurement of striping patterns')
        # pull flat from CRDS using the current context
        crds_dict = {'INSTRUME':'NIRCAM', 
                     'DETECTOR':model.meta.instrument.detector, 
                     'FILTER':model.meta.instrument.filter, 
                     'PUPIL':model.meta.instrument.pupil, 
                     'DATE-OBS':model.meta.observation.date,
                     'TIME-OBS':model.meta.observation.time}
        flats = crds.getreferences(crds_dict, reftypes=['flat'], 
                                   context=crds_context)
        # if the CRDS loopup fails, should return a CrdsLookupError, but 
        # just in case:
        try:
            flatfile = flats['flat']
        except KeyError:
            logger.error('Flat was not found in CRDS with the parameters: {}'.format(crds_dict))
            exit()

        logger.info('Using flat: %s'%(os.path.basename(flatfile)))
        with FlatModel(flatfile) as flat:
            # use the JWST Calibration Pipeline flat fielding Step 
            model,applied_flat = do_correction(model, flat)

    # construct mask for median calculation
    mask = np.zeros(model.data.shape, dtype=bool)
    mask[model.dq == 0] = True
    
    
    # mask out sources
    if mask_sources:        
        if 'long' in fname_in:
            radius = 5
        else:
            radius = 10
        
        
        #First run
        logger.info('First run of source mask')
        source_mask = get_source_mask(model.data, mask, nsig=2, kernel_fwhm=3, dilation_radius=radius)
        
        logger.info('Measuring the pedestal in the image')
        pedestal_data = model.data[ ~(mask|source_mask) ]
        pedestal_data = pedestal_data.flatten()
        median_image = astrostats.sigma_clipped_stats(pedestal_data, sigma=3, maxiters=10)[1]
        logger.info('Image median (unmasked, not DO_NOT_USE, and de-sourced): {:.3f}'.format(med_bkg))
        
        try:
            pedestal = fit_sky(bkg_dat)
        except RuntimeError as e:
            logger.error("Can't fit sky, using median value instead")
            pedestal = median_image
        else:
            logger.info('Fit pedestal: {:.3f}'.format(pedestal))
        
        # subtract off pedestal so it's not included in fit  
        model.data -= pedestal
        
        
    
        
        #Second run
        logger.info('Second run of source mask')
        source_mask = get_source_mask(model.data, mask, nsig=2, kernel_fwhm=3, dilation_radius=radius)
        
        estimator = SExtractorBackground()
        bkg = Background2D(model.data, (256, 256), mask=source_mask|mask, bkg_estimator=estimator, exclude_percentile=30)      
        
        
        
        
        #Third run
        logger.info('Third run of source mask')
        source_mask = get_source_mask(model.data-bkg.background, mask, nsig=2, kernel_fwhm=3, dilation_radius=radius)
        
        
        mask[source_mask] = True 
        logger.info('Masked out sources')
        
        bkg = Background2D(model.data, (256, 256), mask=mask, bkg_estimator=estimator, exclude_percentile=30)
        model.data -= bkg.background
        
        #Save mask
        fits.writeto(fname_out_source, source_mask, overwrite=True)
        logger.info('Saved source mask to %s'%fname_out_source)


        
        #Save background
        header = fits.open(fname_in)[0].header
        hdu1 = fits.PrimaryHDU(header=header)
        hdu2 = fits.ImageHDU(bkg2.background)
        hdu3 = fits.ImageHDU(bkg2.background_rms)
        
        hdu2.name = 'BKG'
        hdu3.name = 'RMS'

        hdu2.header['SIZE'] = 256
        hdu2.header['BKG_EST'] = 'SExtractorBackground'
        
        
        hdul = fits.HDUList([hdu1, hdu2, hdu3])
        hdul.writeto(fname_out_bkg, overwrite=True)
        
    else:
        pedestal = 0.
         
        


    # measure full pattern across image
    full_horizontal, vertical_striping = measure_fullimage_striping(model.data, 
                                                                    mask)

    horizontal_striping = np.zeros(model.data.shape)
    vertical_striping = np.zeros(model.data.shape)

    # keep track of number of number of times the number of masked pixels 
    # in an amp-row exceeds thersh and a full-row median is used instead
    ampcounts = []
    for amp in ['A','B','C','D']:
        ampcount = 0
        rowstart, rowstop, colstart, colstop = NIR_amps[amp]['data']
        ampdata = model.data[:, colstart:colstop]
        ampmask = mask[:, colstart:colstop]
        # fit horizontal striping in amp, collapsing along columns
        hstriping_amp = collapse_image(ampdata, ampmask, dimension='y')
        # check that at least 1/4 of pixels in each row are unmasked
        nmask = np.sum(ampmask, axis=1)
        for i,row in enumerate(ampmask):
            if nmask[i] > (ampmask.shape[1]*thresh):
                # use median from full row
                horizontal_striping[i,colstart:colstop] = full_horizontal[i]
                ampcount += 1
            else:
                # use the amp fit 
                horizontal_striping[i,colstart:colstop] = hstriping_amp[i]
        ampcounts.append('%s-%i'%(amp,ampcount))   

    ampinfo = ', '.join(ampcounts)
    logger.info('%s, full row medians used: %s /%i'%(os.path.basename(fname_in), 
                                                  ampinfo, rowstop-rowstart))

    # remove horizontal striping    
    temp_sub = model.data - horizontal_striping

    # fit vertical striping, collapsing along rows
    vstriping = collapse_image(temp_sub, mask, dimension='x')
    vertical_striping[:,:] = vstriping

    # save horizontal and vertical patterns 
    if save_patterns:
        fits.writeto(fname_out_h, 
                     horizontal_striping, overwrite=True)
        logging.info('Saved horizontal striping pattern to %s'%fname_out_h)
        
        fits.writeto(fname_out_v, 
                     vertical_striping, overwrite=True)
        logging.info('Saved vertical striping pattern to %s'%fname_out_v)

    model.close()
    
    # # copy image 
    # log.info('Copying input to %s'%origfilename)
    # shutil.copy2(fname_in, origfilename)

    # remove striping from science image
    with ImageModel(fname_in) as immodel:
        sci = immodel.data
        sci -= (pedestal + bkg.background)  #Subtract 2d background

        # to replace zeros
 #       wzero = np.where(sci == 0)
        temp_sci = sci - horizontal_striping

        # transpose back
        outsci = temp_sci - vertical_striping
#        outsci[wzero] = 0

        # replace NaNs with zeros and update DQ array
        # the image has NaNs where an entire row/column has been masked out
        # so no median could be calculated.
        # All of the NaNs on LW detectors and most of them on SW detectors
        # are the reference pixels around the image edges. But there is one
        # additional row on some SW detectors 
        wnan = np.isnan(outsci)
        bpflag = dqflags.pixel['DO_NOT_USE']
        outsci[wnan] = np.nan
        immodel.dq[wnan] = np.bitwise_or(immodel.dq[wnan], bpflag)

        # change pixels which are originally 0 back to nan, those are modified by striping (MY
        mask_err0 = (immodel.err==0) | (immodel.data == 0)
        outsci[mask_err0] = np.nan
        immodel.var_poisson[mask_err0] = np.inf
        immodel.var_rnoise[mask_err0] = np.inf

        # write output
        immodel.data = outsci
        # add history entry
        time = datetime.now()
        stepdescription = 'Removed horizontal,vertical striping; remstriping.py %s'%time.strftime('%Y-%m-%d %H:%M:%S')
        # writing to file doesn't save the time stamp or software dictionary 
        # with the History object, but left here for completeness
        software_dict = {'name':'remstriping.py',
                         'author':'Micaela Bagley',
                         'version':'1.0',
                         'homepage':'ceers.github.io'}
        substr = util.create_history_entry(stepdescription, 
                                              software=software_dict)
        immodel.history.append(substr)
        logger.info('Saving cleaned image to %s'%fname_out)
        immodel.save(fname_out)

        
    return


####################################################################################################
# Get source mask and destripe all images


def get_mask_destripe_indiv(maindir, fname_in, imname, logger, overwrite=False, save_mask=True):    
    wispdir = maindir + 'preprocess/wisp/'
    stripedir = maindir + 'preprocess/stripe/'
    tmpdir = maindir + 'preprocess/tmp/'
    
    logger.info('Starting DESTRIPE for {}'.format(imname))
    logger.info('Input fname: {}'.format(fname_in))
    
    
    #Move to tmpdir
    shutil.copy(fname_in, tmpdir + imname + '_prewisp.fits.gz')
    fname_in_tmp = tmpdir + imname + '_prewisp.fits.gz'
    logger.info('Moved temp input file to {}'.format(fname_in_tmp))
    
    #Unzip file
    _ = subprocess.run(['gunzip', '-k', fname_in_tmp], check=True, capture_output=True).stdout
    fname_in_unzip = fname_in_tmp[:-3]
    logger.info('Decompressed temp input file to {}'.format(fname_in_unzip))
    
    #Get NoiseChisel mask
    mask_fname = get_sourcemask_indiv(maindir, fname_in_unzip, imname, logger, overwrite=overwrite)
    logger.info('Source mask complete for {}'.format(imname))  
    
    fname_out_v = stripedir + imname + '_mask_vert.fits.gz'
    fname_out_h = stripedir + imname + '_mask_horiz.fits.gz'
    fname_out = tmpdir + imname + '_destripe.fits'
    
    
    if overwrite:
        logger.info('Overwriting existing DESTRIPE output files')
        
        for f in [fname_out_v, fname_out_h, fname_out]:
            if os.path.exists(f):
                logger.info('Removing {}'.format(f))
                os.remove(f)

    else:
        if os.path.exists(fname_out):
            logger.info('Found existing destriped image, skipping...')
            return    
    
    
    _ = measure_striping(fname_in_unzip, fname_out, fname_out_v, fname_out_h, logger,
                         mask_fname, apply_flat=True, mask_sources=True, save_patterns=save_mask)
    
    
    #Delete copied
    os.remove(fname_in_tmp)
    logger.info('Deleted temp input file {}'.format(fname_in_tmp))
    
    #Delete unzipped file
    os.remove(fname_in_unzip)
    logger.info('Deleted decompressed input file {}'.format(fname_in_unzip))
    
    #Zip output file
    output = subprocess.run(['gzip', '--best', fname_out], check=True, capture_output=True).stdout
    for line in output.decode('utf-8').split('\n'):
        logger.info(line)
    
    fname_out += '.gz'
    logger.info('Compressed output file {}'.format(fname_out))
    
    #Move output to correct directory
    shutil.move(fname_out, stripedir + imname + '_destripe.fits.gz')
    logging.info('Moved destriped file to'.format(stripedir + imname + '_destripe.fits.gz'))
    
    
    logger.info('DESTRIPE complete for {}'.format(imname))
    
    return


def get_mask_destripe_all(indir, maindir, ncpu=1, overwrite=False, save_mask=True):
    
    wispdir = maindir + 'preprocess/wisp/'
    stripedir = maindir + 'preprocess/stripe/'
    tmpdir = maindir + 'preprocess/tmp/'
    logdir = maindir + 'preprocess/logs/'
    
    os.makedirs(stripedir, exist_ok=True)
    
    fnames_in_all = glob.glob(indir + '*.fits.gz')
    imnames_all = [get_imname(fname) for fname in fnames_in_all]

    wisp_mask = np.array([os.path.exists(wispdir + i + '_wisp.fits.gz') for i in imnames_all], dtype=bool)
    fnames_in = []
    fnames_log = []
    for i in range(len(imnames_all)):
        fnames_log.append(logdir + imnames_all[i] + '.log')
        
        if wisp_mask[i]:
            fnames_in.append(wispdir + imnames_all[i] + '_wisp.fits.gz')
        else:
            fnames_in.append(indir + imnames_all[i] + '_rate.fits.gz')
            
        
    
    
    print('Destriping {} images with {} cores'.format(len(imnames_all), ncpu))
    
    loggers = []
    for i in range(len(imnames_all)):
        loggers.append(setup_logger('NVP:{}'.format(imnames[i]), fnames_log[i], logging.INFO))
    
    
    func = partial(get_mask_destripe_indiv, maindir, overwrite=overwrite, save_mask=save_mask)
    
    pool = WorkerPool(n_jobs=ncpu)
    pool.map(func, zip(fnames_in, imnames_all, loggers), iterable_len=len(fnames_in),
             progress_bar=True, progress_bar_style='rich')
    pool.join()
    
    print('Destriping complete for all images')
        
    return

