import os
import sys
from functools import partial

import warnings
from astropy.io import fits
from mpire import WorkerPool
from numba import njit
from tqdm import tqdm

import numpy as np
from astropy.convolution import convolve_fft
from scipy.stats import skew
import pyfftw

# from BSplineSFFT_numpy import BSpline_MatchingKernel, BSpline_GridConvolve
from sky_level_estimator import SkyLevel_Estimator
from multi_proc import Multi_Proc

# from sfft.BSplineSFFT import BSpline_MatchingKernel, BSpline_DeCorrelation, BSpline_GridConvolve
# from sfft.utils.SkyLevelEstimator import SkyLevel_Estimator
# from sfft.utils.meta.MultiProc import Multi_Proc

############################################################################################################
################################################   SFFT   ##################################################
############################################################################################################


def mask_images(maindir, ref_name, sci_name, logger, skysub=False, conv_ref=False, conv_sci=False):
    """Mask cross-convolved (or not) REF and SCI images using the SFFT mask. Masked images are stored in maindir/output/.

    Arguments:
    -----------
    maindir : str
        Main directory for the NEXUS Variability Pipeline.
    ref_name : str
        Name of the reference image.
    sci_name : str
        Name of the science image.
    logger : logging.Logger
        Logger for logging messages.
    skysub : bool, optional
        Whether the images are sky-subtracted. Default is False.
    conv_ref : bool, optional
        Whether the reference image is cross-convolved. Default is False.
    conv_sci : bool, optional
        Whether the science image is cross-convolved. Default is False.
    
    """
    
    
    indir = maindir + 'input/'
    outdir = maindir + 'output/'
    maskdir = maindir + 'mask/'
    
    
    if conv_ref:
        fname_ref = outdir + '{}.crossconvd.fits'.format(ref_name)
    else:
        if skysub:
            fname_ref = indir + '{}.skysub.fits'.format(ref_name)
        else:
            fname_ref = indir + '{}.fits'.format(ref_name)
            
    if conv_sci:
        fname_sci = outdir + '{}.crossconvd.fits'.format(sci_name)
    else:
        if skysub:
            fname_sci = indir + '{}.skysub.fits'.format(sci_name)
        else:
            fname_sci = indir + '{}.fits'.format(sci_name)

    
    ##############################################################
    #Get image mask
    
    logger.info('Loading image mask')
    fname_mask = maskdir + '{}.mask4sfft.fits'.format(ref_name)
    subtmask = fits.getdata(fname_mask, ext=0).T.astype(bool)
    
    ##############################################################
    # Generate masked REF
    
    if conv_ref:
        fname_mref = outdir + '{}.crossconvd.masked.fits'.format(ref_name)
    else:
        fname_mref = outdir + '{}.masked.fits'.format(ref_name)

    if os.path.exists(fname_mref):
        logger.info('Masked REF already exists, skipping masking')
    else:

        with fits.open(fname_ref) as hdul:
            mssg = 'Masking {}'.format(ref_name)
            logger.info('MelOn CheckPoint: {}'.format(mssg))
            
            im_temp = hdul[0].data.T.copy()
            im_temp[~subtmask] = 0.
            
            hdul[0].data[:,:] = im_temp.T
            hdul.writeto(fname_mref, overwrite=True)
            
        del im_temp
        
    ##############################################################
    # Generate masked SCI
    
    if conv_sci:
        fname_msci = outdir + '{}.crossconvd.masked.fits'.format(sci_name)
    else:
        fname_msci = outdir + '{}.masked.fits'.format(sci_name)
    
    if os.path.exists(fname_msci):
        logger.info('Masked SCI already exists, skipping masking')
    else:    
        with fits.open(fname_sci) as hdul:
            mssg = 'Masking {}'.format(sci_name)
            logger.info('MelOn CheckPoint: {}'.format(mssg))
            
            im_temp = hdul[0].data.T.copy()
            im_temp[~subtmask] = 0.
            
            hdul[0].data[:,:] = im_temp.T
            hdul.writeto(fname_msci, overwrite=True)
            
        del im_temp
        

    del subtmask
        
    return


def run_sfft_bspline(maindir, ref_name, sci_name, logger, channel='SW', force_conv='REF', gkerhw=11,
                     kernel_type='B-Spline', kernel_deg=2, nknot_x=3, nknot_y=3, separate_scaling=True,
                     scaling_type='Polynomial', scaling_deg=2, bkg_type='Polynomial', bkg_deg=0, 
                     regularize_kernel=True, ignore_laplacian_kercent=True, xy_regularize_fname='', lambda_regularize=3e-5,                     
                     max_threads_per_block=8, minimize_memory_usage=False, ncpu=8, use_gpu=False, ngpu=1,
                     skysub=False, conv_ref=False, conv_sci=False):
    
    """Run the SFFT difference imaging algorithm on the REF and SCI images. The results are stored in maindir/output/.

    Arguments:
    -----------
    maindir : str
        Main directory for the NEXUS Variability Pipeline.
    ref_name : str
        Name of the reference image.
    sci_name : str
        Name of the science image.
    logger : logging.Logger
        Logger for logging messages.
    channel : str, optional
        Channel of the images (e.g., 'SW', 'LW', 'euclid'). Default is 'SW'.
    force_conv : str, optional
        The same as the FORCE_CONV parameter in the SFFT algorithm. Can be either 'REF' or 'SCI'. If 'REF', the 
        REF image is matched to the SCI image. If 'SCI', the SCI image is matched to the REF image. Default is 'REF'.
    gkerhw : int, optional
        The half-width of the Gaussian kernel used in the SFFT algorithm. Default is 11.
    kernel_type : str, optional
        The type of spatial variation for the matching kernel across the image. Can be 'B-Spline' or 'Polynomial'. Default is 'B-Spline'.
    kernel_deg : int, optional
        The degree of the B-Spline or Polynomial used for the matching kernel variation. Default is 2.
    nknot_x : int, optional
        The number of knots along the X-axis for the B-Spline kernel. Default is 3.
    nknot_y : int, optional
        The number of knots along the Y-axis for the B-Spline kernel. Default is 3.
    separate_scaling : bool, optional
        Whether or not the photometric scaling variations are modeled separately from the matching kernel. Default is True. If the ype of variation
        is the same for both the matching kernel and photometric scaling, and kernel_deg <= scaling deg, then this will be set to False.
    scaling_type : str, optional
        The type of spatial variation for the photometric scaling across the image. Can be 'B-Spline' or 'Polynomial'. Default is 'Polynomial'.
    scaling_deg : int, optional
        The degree of the B-Spline or Polynomial used for the photometric scaling variation. Default is 2.
    bkg_type : str, optional
        The type of spatial variation for the differential background across the image. Can be 'B-Spline' or 'Polynomial'. Default is 'Polynomial'.
    bkg_deg : int, optional
        The degree of the B-Spline or Polynomial used for the differential background variation. Default is 0 (constant background).
    regularize_kernel : bool, optional
        Whether to apply Tikhonov regularization to the matching kernel. Default is True.
    ignore_laplacian_kercent : bool, optional
        Whether to ignore the Laplacian kernel percent when regularizing the matching kernel. Default is True.
    xy_regularize_fname : str, optional
        Path to a file containing random image coordinates for regularization. If empty, random coordinates will be generated. Default is ''.
    lambda_regularize : float, optional
        The strength of the Tikhonov regularization applied to the matching kernel. Default is 3e-5.
    max_threads_per_block : int, optional
        Maximum number of threads per block for GPU processing. Default is 8.
    minimize_memory_usage : bool, optional
        Whether to minimize GPU memory usage during processing. Default is False.
    ncpu : int, optional
        Number of CPU threads to use for processing. Default is 8.
    use_gpu : bool, optional
        Whether to use GPU for processing. If False, CPU will be used. Default is False.
    ngpu : int, optional
        Number of GPUs to use for processing. Default is 1.
    skysub : bool, optional
        Whether the images are sky-subtracted. Default is False.
    conv_ref : bool, optional
        Whether the reference image is cross-convolved. Default is False.
    conv_sci : bool, optional
        Whether the science image is cross-convolved. Default is False.

    """
    
    
    if use_gpu:
        from BSplineSFFT_cupy import BSpline_Packet
        backend = 'Cupy'
    else:
        from BSplineSFFT_numpy import BSpline_Packet
        from mkl_fft import _numpy_fft as nfft
        backend = 'Numpy'
    
    indir = maindir + 'input/'
    maskdir = maindir + 'mask/'
    outdir = maindir + 'output/'        

    fname_diff = outdir + '{}.sfftdiff.fits'.format(sci_name)
    fname_soln = outdir + '{}.sfftsoln.fits'.format(sci_name)
    if os.path.exists(fname_diff) and os.path.exists(fname_soln):
        logger.info('SFFT output files found, skipping SFFT')
        return


    if conv_ref:
        fname_ref = outdir + '{}.crossconvd.fits'.format(ref_name)
        fname_mref = outdir + '{}.crossconvd.masked.fits'.format(ref_name)
    else:
        if skysub:
            fname_ref = indir + '{}.skysub.fits'.format(ref_name)
        else:
            fname_ref = indir + '{}.fits'.format(ref_name)
            
        fname_mref = outdir + '{}.masked.fits'.format(ref_name)
        
        
    if conv_sci:
        fname_sci = outdir + '{}.crossconvd.fits'.format(sci_name)
        fname_msci = outdir + '{}.crossconvd.masked.fits'.format(sci_name)
    else:
        if skysub:
            fname_sci = indir + '{}.skysub.fits'.format(sci_name)
        else:
            fname_sci = indir + '{}.fits'.format(sci_name)
            
        fname_msci = outdir + '{}.masked.fits'.format(sci_name)
    
    
    fname_mask = maskdir + '{}.mask4sfft.fits'.format(ref_name)
    subtmask = fits.getdata(fname_mask, ext=0).T.astype(bool)
    
    n0, n1 = subtmask.shape
    del subtmask
    
    ForceConv = force_conv
    
    if channel == 'SW':
        GKerHW = 11
    elif channel == 'LW':
        GKerHW = 5
    elif channel == 'euclid':
        GKerHW = 3
    
    GKerHW = gkerhw
        
    ##############################################################
    # Set spatial variation for kernel
    
    #   For long-wave channel: Can tune the internal knots to make a grid of 150x150 px
    KerSpType = kernel_type                                                            # B-Spline form
    KerSpDegree = kernel_deg                                                           # B-Spline degree
    
    if gkerhw == 11:
        dspline_x = dspline_y = 300
    elif gkerhw == 5:
        dspline_x = dspline_y = 150
    elif channel == 'euclid':
        dspline_x = dspline_y = 100


    if KerSpType == 'B-Spline':
        dspline_x = n0 // int(nknot_x)
        dspline_y = n1 // int(nknot_y)
        
        KerIntKnotX = list(np.arange(0, n0, dspline_x) )[1:]                                     # Internal knots along X, using a grid of 300x300 px
        KerIntKnotY = list(np.arange(0, n1, dspline_y) )[1:]                                     # Internal knots along Y, using a grid of 300x300 px


        # KerIntKnotX = KerIntKnotX[:-1]  #TESTING
        # KerIntKnotY = KerIntKnotY[:-1]


        logger.info('Internal knot positions for kernel (X): {}'.format(KerIntKnotX))
        logger.info('Internal knot positions for kernel (Y): {}'.format(KerIntKnotY))
        
    else:
        KerIntKnotX = []
        KerIntKnotY = []

    ##############################################################
    # Set spatial variation for photometric scaling

    SEPARATE_SCALING = separate_scaling                 # Disentagled from kernel spatial variation
    ScaSpType = scaling_type                            # Polynomial form
    ScaSpDegree = scaling_deg                           # Polynomial degree
    
    if ScaSpType == 'B-Spline':
        ScaIntKnotX = [.5 + i*n0/(ScaSpDegree+1) for i in range(1,ScaSpDegree+1)]
        ScaIntKnotY = [.5 + i*n1/(ScaSpDegree+1) for i in range(1,ScaSpDegree+1)]
    else:
        ScaIntKnotX = []
        ScaIntKnotY = []

    ##############################################################
    # Set spatial variation for differential background
    
    #   Assume background subtraction has been well performed - use constant
    BkgSpType = bkg_type
    BkgSpDegree = bkg_deg
    
    if BkgSpType == 'B-Spline':
        BkgIntKnotX = [.5 + i*n0/(BkgSpDegree+1) for i in range(1,BkgSpDegree+1)]
        BkgIntKnotY = [.5 + i*n1/(BkgSpDegree+1) for i in range(1,BkgSpDegree+1)]
    else:
        BkgIntKnotX = []
        BkgIntKnotY = []
        
    ##############################################################
    # Set parameters for Tikhonov regularization
    
    REGULARIZE_KERNEL = regularize_kernel                   # Activate Tikhonov regularization?
    IGNORE_LAPLACIAN_KERCENT = ignore_laplacian_kercent     # Do not suppress delta-like kernel?
    np.random.seed(10086)
    
    if xy_regularize_fname != '':
        XY_REGULARIZE = np.loadtxt(xy_regularize_fname).T
    else:
        XY_REGULARIZE = np.array([
            np.random.uniform(10., n0-10., 512),
            np.random.uniform(10., n1-10., 512)
        ]).T                                                    #Random image coords for regularization
    
    #On regularization strength:
    #   Empirically, the recommended tuning strength for JWST/NIRCam is 1e-7 to 1e-3

    LAMBDA_REGULARIZE = lambda_regularize                       # Regularization strength factor
    
    ##############################################################
    # Run SFFT
    
    im_diff = BSpline_Packet.BSP(FITS_REF=fname_ref, FITS_SCI=fname_sci, FITS_mREF=fname_mref, FITS_mSCI=fname_msci,
                                 FITS_DIFF=fname_diff, FITS_Solution=fname_soln, ForceConv=ForceConv, GKerHW=GKerHW,
                                 KerSpType=KerSpType, KerSpDegree=KerSpDegree, KerIntKnotX=KerIntKnotX, KerIntKnotY=KerIntKnotY,
                                 SEPARATE_SCALING=SEPARATE_SCALING, ScaSpType=ScaSpType, ScaSpDegree=ScaSpDegree,
                                 ScaIntKnotX=ScaIntKnotX, ScaIntKnotY=ScaIntKnotY, BkgSpType=BkgSpType, BkgSpDegree=BkgSpDegree,
                                 BkgIntKnotX=BkgIntKnotX, BkgIntKnotY=BkgIntKnotY, 
                                 REGULARIZE_KERNEL=REGULARIZE_KERNEL, IGNORE_LAPLACIAN_KERCENT=IGNORE_LAPLACIAN_KERCENT,
                                 XY_REGULARIZE=XY_REGULARIZE, WEIGHT_REGULARIZE=None, LAMBDA_REGULARIZE=LAMBDA_REGULARIZE,
                                 BACKEND_4SUBTRACT=backend, CUDA_DEVICE_4SUBTRACT='0', MAX_THREADS_PER_BLOCK=max_threads_per_block,
                                 NUM_CPU_THREADS_4SUBTRACT=ncpu, MINIMIZE_GPU_MEMORY_USAGE=minimize_memory_usage, 
                                 logger=logger, ngpu=ngpu, VERBOSE_LEVEL=2)[1]
    
    del im_diff
    return


############################################################################################################
###########################################   DECORRELATION   ##############################################
############################################################################################################

@njit(fastmath=True)
def get_dcker_fstack(i, XY_TiC, DCKerStack):
    return np.append([ XY_TiC[i,0], XY_TiC[i,1] ], DCKerStack[i].flatten())

#Function to calculate propagated noise map that went through convolutions (using MCMC approach)
def MultiConvolveNoise_CPU(PixA_Noise, ConvKerSeq, KerNormalizeSeq,
                       MCNSAMP=1024, RANDOM_SEED=10086, NPROC=32):
    
    def func_multiconv(idx):
        np.random.seed(RANDOM_SEED+idx)
        
        PixA_SampledNoise = np.random.normal(0, 1, PixA_Noise.shape) * PixA_Noise
        PixA_CSampledNoise = PixA_SampledNoise.copy()
        
        for ConvKer, KerNormalize in zip(ConvKerSeq, KerNormalizeSeq):
            if (np.nansum(ConvKer) < .01) and KerNormalize:
                normalize_kernel = False
                kernel = ConvKer / np.nansum(ConvKer)
            else:
                normalize_kernel = KerNormalize
                kernel = ConvKer.copy()
            

            PixA_CSampledNoise = convolve_fft(PixA_CSampledNoise, kernel, boundary='fill',
                                            nan_treatment='fill', fill_value=0., 
                                            normalize_kernel=normalize_kernel)
            
        return PixA_CSampledNoise


    Ntask = MCNSAMP
    # taskid_lst = np.arange(Ntask)
    # MPDICT = Multi_Proc.MP(taskid_lst=taskid_lst, func=func_multiconv, nproc=NPROC, mode='mp')
    
    with WorkerPool(n_jobs=NPROC) as pool:
        output = pool.map(func_multiconv, range(Ntask), progress_bar=True, progress_bar_style='rich')
    
    # SPixA_CSampledNoise = np.array([ MPDICT[i] for i in taskid_lst ])
    SPixA_CSampledNoise = output.reshape((Ntask, *PixA_Noise.shape))
    return SPixA_CSampledNoise


#Function to calculate propagated noise map that went through convolutions (using MCMC approach)
def MultiConvolveNoise_GPU(PixA_Noise, ConvKerSeq, KerNormalizeSeq,
                           MCNSAMP=1024, CUDA_DEVICE='0', 
                           RANDOM_SEED=10086, CLEAN_GPU_MEMORY=False, NPROC=32):
    
    import cupy as cp
    from cupyx.scipy.signal import fftconvolve, convolve2d
    device = cp.cuda.Device(int(CUDA_DEVICE))
    device.use()
    
    if CLEAN_GPU_MEMORY:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        

    def get_resamp(idx):
        np.random.seed(RANDOM_SEED+idx)
        PixA_SampledNoise = np.random.normal(0, 1, PixA_Noise.shape) * PixA_Noise
        return PixA_SampledNoise


    Ntask = MCNSAMP
    with WorkerPool(n_jobs=NPROC) as pool:
        output = pool.map(get_resamp, range(Ntask), progress_bar=True, progress_bar_style='rich')
    
    PixA_SampledNoise = output.reshape((Ntask, *PixA_Noise.shape))


    SPixA_CSampledNoise_GPU = cp.zeros((Ntask, *PixA_Noise.shape), dtype=np.float32)
    
    for idx in tqdm( range(Ntask) ):
        PixA_CSampledNoise_GPU = cp.asarray(PixA_SampledNoise[idx].astype(np.float32))
        
        for ConvKer, KerNormalize in zip(ConvKerSeq, KerNormalizeSeq):
            if KerNormalize:
                kernel = ConvKer.copy() / np.nansum(ConvKer)
                
            kernel_GPU = cp.asarray(kernel.astype(np.float32))
            PixA_CSampledNoise_GPU = fftconvolve(PixA_CSampledNoise_GPU, kernel_GPU, mode='same')
        
        SPixA_CSampledNoise_GPU[idx] = PixA_CSampledNoise_GPU

    
    SPixA_CSampledNoise = cp.asnumpy(SPixA_CSampledNoise_GPU)
    return SPixA_CSampledNoise
    

def decorrelate_noise_get_snr(maindir, ref_name, sci_name, conv_ref, conv_sci, skysub, logger, tilesize_ratio=1, nsamp=1024, ncpu=8, use_gpu=False):
    
    """Decorrelate the raw difference image and calculate the differential SNR map. The results are stored in maindir/output/.

    Arguments:
    -----------
    maindir : str
        Main directory for the NEXUS Variability Pipeline.
    ref_name : str
        Name of the reference image.
    sci_name : str
        Name of the science image.
    conv_ref : bool
        Whether the reference image is cross-convolved.
    conv_sci : bool
        Whether the science image is cross-convolved.
    skysub : bool
        Whether the images are sky-subtracted.
    logger : logging.Logger
        Logger for logging messages.
    tilesize_ratio : int, optional
        Ratio for the size of the tiles used in decorrelation to GKerHW. Default is 1.
    nsamp : int, optional
        Number of samples for the Monte Carlo noise propagation in calculating the SNR map. Default is 1024.
    ncpu : int, optional
        Number of CPU threads to use for processing. Default is 8.
    use_gpu : bool, optional
        Whether to use GPU for processing. If False, CPU will be used. Default is False

    """
    
    
    
    # import pyfftw    
    # pyfftw.interfaces.cache.enable()
    # pyfftw.interfaces.cache.set_keepalive_time(10)
    
    if use_gpu:
        # from BSplineSFFT_cupy import BSpline_DeCorrelation
        import cupy as cp
        from PureCupyDeCorrelationCalculator import PureCupy_DeCorrelation_Calculator
        
        from BSplineSFFT_cupy import BSpline_MatchingKernel, BSpline_GridConvolve
    else:
        from BSplineSFFT_numpy import BSpline_DeCorrelation
        from BSplineSFFT_numpy import BSpline_MatchingKernel, BSpline_GridConvolve
    
    
    indir = maindir + 'input/'   
    noisedir = maindir + 'noise/' 
    psfdir = maindir + 'psf/'    
    outdir = maindir + 'output/'
    
    
    if skysub:
        fname_lsci = indir + '{}.skysub.fits'.format(sci_name)
        fname_lref = indir + '{}.skysub.fits'.format(ref_name)
    else:
        fname_lsci = indir + '{}.fits'.format(sci_name)
        fname_lref = indir + '{}.fits'.format(ref_name)
    
    
    fname_noise_lref = noisedir + '{}.noise.fits'.format(ref_name)
    fname_noise_lsci = noisedir + '{}.noise.fits'.format(sci_name)
    fname_psf_lref = psfdir + '{}.psf.fits'.format(ref_name)
    fname_psf_lsci = psfdir + '{}.psf.fits'.format(sci_name)
    fname_soln = outdir + '{}.sfftsoln.fits'.format(sci_name)
    fname_diff = outdir + '{}.sfftdiff.fits'.format(sci_name)
    
    fname_decorr_diff = outdir + '{}.sfftdiff.decorr.fits'.format(sci_name)
    fname_decorr_diff_snr = outdir + '{}.sfftdiff.decorr.snr.fits'.format(sci_name)
    fname_match_ker_mean = outdir + '{}.sfftmatchker.mean.fits'.format(sci_name)
    fname_decorr_ker_mean = outdir + '{}.sfftdecorrker.mean.fits'.format(sci_name)
    fname_match_ker_grid = outdir + '{}.sfftmatchker.grid.fits.gz'.format(sci_name)

    
    if os.path.exists(fname_decorr_diff) and os.path.exists(fname_decorr_diff_snr):
        logger.info('Decorrelated DIFF and SNR map already exist, skipping decorrelation')
        return

    logger.info('Getting matching/decorrelation kernels across image')
    
    ##############################################################
    # Get grid for decorrelation
    
    with fits.open(fname_lsci) as hdul:
        im_lsci = hdul[0].data.T.copy()
    with fits.open(fname_lref) as hdul:
        im_lref = hdul[0].data.T.copy()
    
    TILESIZE_RATIO = tilesize_ratio                               #Each tile is (2A+1)x(2A+1) pixels, where A=TILESIZE_RATIO*GKerHW
    _hdr = fits.getheader(fname_lsci, ext=0)
    N0, N1 = int(_hdr['NAXIS1']), int(_hdr['NAXIS2'])
    
    _hdr = fits.getheader(fname_soln, ext=0)
    GKerHW = int(_hdr['KERHW'])

    lab = 0
    XY_TiC = []
    TiHW = round(TILESIZE_RATIO * GKerHW)
    TiN = 2*TiHW + 1

    AllocatedL = np.zeros((N0, N1), dtype=int)
    for x in np.arange(0, N0, TiN):    
        xe = np.min([x + TiN, N0])

        for y in range(0, N1, TiN):
            ye = np.min([y + TiN, N1])
            
            AllocatedL[x:xe, y:ye] = lab
            
            xq = .5 + x + (xe-x)/2.             #tile center (x)
            yq = .5 + y + (ye-y)/2.             #tile center (y)
            XY_TiC.append([xq, yq])
            
            lab += 1
            
    XY_TiC = np.array(XY_TiC)
    NTILE = XY_TiC.shape[0]

    logger.info('MeLOn CheckPoint: [%d] DeCorrelation Tiles of size [%d x %d]' %(NTILE, TiN, TiN))
    
    ##############################################################
    # Run decorrelation
    
    #Read the matching kernels over the grid
    MKerStack = BSpline_MatchingKernel(XY_q=XY_TiC).FromFITS(FITS_Solution=fname_soln, 
                                                            logger=logger
                                                            )


    # MKerFStack = np.array([ np.append([ XY_TiC[i,0], XY_TiC[i,1] ], MKerStack[i].flatten()) for i in range(NTILE) ])
                                #Each row is x, y + flatten matching kernel

    #Read Webb PSF models
    im_psf_lref = fits.getdata(fname_psf_lref, ext=0).T
    im_psf_lsci = fits.getdata(fname_psf_lsci, ext=0).T

    #Measure background noise for uncolvolved sci and ref images
    bkgsig_lref = SkyLevel_Estimator.SLE(PixA_obj=fits.getdata(fname_lref, ext=0).T )[1]
    bkgsig_lsci = SkyLevel_Estimator.SLE(PixA_obj=fits.getdata(fname_lsci, ext=0).T )[1]


    #For the decorrelation in the paper, they assume that the images are cross-convolved beforehand
    #They are deconvolving a "corss-convolved" difference image
    #In the case when we don't want to cross-convolve ***I THINK*** we can just set the PSFs=delta(x) in the calculation
    #This is because we aren't convolving the REF/SCI images in the beginning by each other's respective PSFs
    #    so we don't need to decorrelate the noise introduced by doing so
    if conv_ref:
        im_ilst = im_psf_lsci.copy()
    else:
        im_ilst = np.zeros_like(im_psf_lsci)
        im_ilst[im_psf_lsci.shape[0]//2, im_psf_lsci.shape[1]//2] = 1.0
        
    if conv_sci:
        im_jlst = im_psf_lref.copy()
    else:
        im_jlst = np.zeros_like(im_psf_lref)
        im_jlst[im_psf_lref.shape[0]//2, im_psf_lref.shape[1]//2] = 1.0
        
        
    #Get tiles with no data
    nodat_ind = []
    for i in range(NTILE):
        mask = (AllocatedL == i)
        if (np.nansum(im_lref[mask]) == 0.) or (np.nansum(im_lsci[mask]) == 0.):
            nodat_ind.append(i)
    logger.info('Found [%d] tiles with no data' %len(nodat_ind))
    
    if use_gpu:
        im_jlst_gpu = cp.array(im_jlst.astype(np.float32))
        im_ilst_gpu = cp.array(im_ilst.astype(np.float32))
        

    #Define func for calculating noise decorrelation kernels
    def func_decorr(idx):
        nside = im_psf_lref.shape[0] *2 + 1
        
        if idx in nodat_ind:
            return np.zeros((nside, nside), dtype=float)
        
        MKer = MKerStack[idx]
        
        
        if use_gpu:
            MKer_gpu = cp.array(MKer.astype(np.float32))
            DCKer = PureCupy_DeCorrelation_Calculator.PCDC(NX_IMG=N0, NY_IMG=N1,
                                                           KERNEL_GPU_JQueue=[im_jlst_gpu], BKGSIG_JQueue=[bkgsig_lsci],
                                                           KERNEL_GPU_ILst=[im_ilst_gpu], BKGSIG_IQueue=[bkgsig_lref],
                                                           MATCH_KERNEL_GPU=MKer_gpu, 
                                                           REAL_OUTPUT=False, REAL_OUTPUT_SIZE=(nside,nside),
                                                           NORMALIZE_OUTPUT=True, VERBOSE_LEVEL=0)
            DCKer = cp.asnumpy(DCKer)

        else:
            DCKer = BSpline_DeCorrelation.BDC(MK_JLst=[im_jlst], SkySig_JLst=[bkgsig_lsci],
                                            MK_ILst=[im_ilst], SkySig_ILst=[bkgsig_lref],
                                            MK_Fin=MKer, KERatio=2., DENO_CLIP_RATIO=100000.,  
                                            VERBOSE_LEVEL=0)
        return DCKer

    nside = im_psf_lref.shape[0] *2 + 1

    #Run decorr in parallel
    with WorkerPool(n_jobs=ncpu) as pool:
        dckerstack = pool.map(func_decorr, range(NTILE), progress_bar=True, progress_bar_style='rich')
        
    DCKerStack = dckerstack.reshape((NTILE, nside, nside))
    del dckerstack         


    # taskid_lst = np.arange(NTILE)
    # mp_dict = Multi_Proc.MP(taskid_lst=taskid_lst, func=func_decorr, nproc=ncpu, mode='mp')
    # DCKerStack = np.array([ mp_dict[i] for i in taskid_lst ])
        
    # DCKerFStack = np.array([ np.append([ XY_TiC[i,0], XY_TiC[i,1] ], DCKerStack[i].flatten()) for i in range(NTILE) ])
                                #Each row is x, y + flatten matching kernel

    MKerFStack_small = MKerStack.reshape(NTILE, -1)
    DCKerFStack_small = DCKerStack.reshape(NTILE, -1)
                                
    ##############################################################
    # Save PSF grids
    
    logger.info('Saving PSF grids')
    
    _L_match = 2*GKerHW + 1    
    hdus_match = [ fits.PrimaryHDU() ]
        
    for i in range(NTILE):
        im_i = MKerStack[i].reshape((_L_match, _L_match)).T
        hdu_i = fits.ImageHDU(data=im_i, name='Tile_{}'.format(i))
        hdus_match.append(hdu_i)        

    
    hdu0 = fits.PrimaryHDU(data=AllocatedL.T)
    hdul = fits.HDUList([hdu0])
    fname_gridlabel = outdir + '{}.psfgridlabel.fits'.format(sci_name)
    hdul.writeto(fname_gridlabel, overwrite=True)
    logger.info('Wrote grid label to file {}'.format(fname_gridlabel))
    
    hdul = fits.HDUList(hdus_match)
    hdul.writeto(fname_match_ker_grid, overwrite=True)
    logger.info('Wrote matching kernel grid to file {}'.format(fname_match_ker_grid))

    del im_ilst, im_jlst
    
    ##############################################################
    # Save decorrelation
    
    #Get difference image
    im_diff = fits.getdata(fname_diff, ext=0).T

    #Zero-out boundary before decorrelation
    boundary_mask = np.ones((N0, N1)).astype(bool)
    boundary_mask[GKerHW:-GKerHW, GKerHW:-GKerHW] = False
    
    #Perform decorrelation for DIFF on the grid, while masking boundary
    im_obj = im_diff.copy()
    im_obj[boundary_mask] = 0.
    
    logger.info('Performing decorrelation')
    _gsvc = BSpline_GridConvolve(PixA_obj=im_diff, AllocatedL=AllocatedL, KerStack=DCKerStack,
                                nan_fill_value=0., use_fft=True, normalize_kernel=True)
    
    if use_gpu:
        logger.info('\t With GPU')
        im_decorr_diff = _gsvc.GSVC_GPU(CLEAN_GPU_MEMORY=True, nproc=ncpu)
    else:
        logger.info('\t With CPU')
        im_decorr_diff = _gsvc.GSVC_CPU(nproc=ncpu)
    
    im_decorr_diff[boundary_mask] = 0.
    
    with fits.open(fname_diff) as hdul:
        hdul[0].data[:,:] = im_decorr_diff.T
        hdul.writeto(fname_decorr_diff, overwrite=True)
        
    logger.info('MeLOn CheckPoint: DeCorrelated DIFF Saved!')
    logger.info('Saved to file {}'.format(fname_decorr_diff))
    
    del im_diff, im_decorr_diff, _gsvc, boundary_mask
    del DCKerStack, MKerStack
    

    #######################################################
    # Get SNR map
    
    logger.info('Getting differential SNR map')
        
    #Read noise maps
    im_lref_noise = fits.getdata(fname_noise_lref, ext=0).T
    im_lsci_noise = fits.getdata(fname_noise_lsci, ext=0).T
    
    
    #Get mean decorrelation kernel (approx)
    _L = round(np.sqrt(DCKerFStack_small.shape[1]))
    im_decorr_kernel_mean = np.mean(DCKerFStack_small, axis=0).reshape((_L, _L))
    
    #Get mean matching kernel (approx)
    _hdr = fits.getheader(fname_soln, ext=0)
    GKerHW = int(_hdr['KERHW'])
    
    _L = 2*GKerHW + 1
    im_match_kernel_mean = np.mean(MKerFStack_small, axis=0).reshape((_L, _L))
    
    del DCKerFStack_small, MKerFStack_small
    
    
    #Same as for decorr: ***I THINK*** we can just set the PSFs=delta(x) in the calculation
    if conv_sci:
        im_nprop_s = im_psf_lref.copy()
    else:
        im_nprop_s = np.zeros_like(im_psf_lref)
        im_nprop_s[im_psf_lref.shape[0]//2, im_psf_lref.shape[1]//2] = 1.0
        
    if conv_ref:
        im_nprop_r = im_psf_lsci.copy()
    else:
        im_nprop_r = np.zeros_like(im_psf_lsci)
        im_nprop_r[im_psf_lsci.shape[0]//2, im_psf_lsci.shape[1]//2] = 1.0
    
    
    #Noise propagation for SCI
    im_noise = im_lsci_noise.copy()
    ConvKerSeq, KerNormalizeSeq = [im_nprop_s, im_decorr_kernel_mean], [True, True]
    
    if use_gpu:
        im_noise_sci_samp = MultiConvolveNoise_GPU(im_noise, ConvKerSeq, KerNormalizeSeq, 
                                            MCNSAMP=nsamp, CUDA_DEVICE='0', RANDOM_SEED=10086, 
                                            CLEAN_GPU_MEMORY=True, NPROC=ncpu)
    else:
        im_noise_sci_samp = MultiConvolveNoise_CPU(im_noise, ConvKerSeq, KerNormalizeSeq, 
                                            MCNSAMP=nsamp, RANDOM_SEED=10086, NPROC=ncpu)
    
    #Noise propagation for REF
    im_noise = im_lref_noise.copy()
    ConvKerSeq, KerNormalizeSeq = [im_nprop_r, im_match_kernel_mean, im_decorr_kernel_mean], [True, False, True]
    
    if use_gpu:
        im_noise_ref_samp = MultiConvolveNoise_GPU(im_noise, ConvKerSeq, KerNormalizeSeq,
                                           MCNSAMP=nsamp, CUDA_DEVICE='0', RANDOM_SEED=2*10086, 
                                           CLEAN_GPU_MEMORY=True, NPROC=ncpu)
    else:
        im_noise_ref_samp = MultiConvolveNoise_CPU(im_noise, ConvKerSeq, KerNormalizeSeq,
                                            MCNSAMP=nsamp, RANDOM_SEED=2*10086, NPROC=ncpu)
                                                            #Avoid using overlapping seed
                                                        
    #Noise proppagation for the decorrelated DIFF
    im_noise_diff_samp = im_noise_sci_samp - im_noise_ref_samp
    im_noise_diff = np.std(im_noise_diff_samp, axis=0)
    
    #Divide decorrelated difference by propagated noise
    im_decorr_diff = fits.getdata(fname_decorr_diff, ext=0).T
    im_decorr_diff_snr = im_decorr_diff / im_noise_diff
    
    #Save
    with fits.open(fname_decorr_diff) as hdul:
        hdul[0].data[:,:] = im_decorr_diff_snr.T
        hdul.writeto(fname_decorr_diff_snr, overwrite=True)
        
    logger.info('MeLOn CheckPoint: SNR Map of DeCorrelated DIFF Saved!')
    logger.info('Saved to file %s' %fname_decorr_diff_snr)
    
    
    
    # Save mean kernels
    hdu0 = fits.PrimaryHDU(data=im_match_kernel_mean.T)
    hdul = fits.HDUList([hdu0])
    hdul.writeto(fname_match_ker_mean, overwrite=True)
    logger.info('Wrote mean matching kernel to file {}'.format(fname_match_ker_mean))
    
    
    
    hdu0 = fits.PrimaryHDU(data=im_decorr_kernel_mean.T)
    hdul = fits.HDUList([hdu0])
    hdul.writeto(fname_decorr_ker_mean, overwrite=True)
    logger.info('Wrote mean decorrelation kernel to file {}'.format(fname_decorr_ker_mean))
    
    return



############################################################################################################
#############################################   STATISTICS   ###############################################
############################################################################################################

def get_background(fname):
    from sfft.utils.pyAstroMatic.PYSEx import PY_SEx
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        SEx_param = ['X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLAGS'
                     'FLUX_RADIUS', 'FWHM_IMAGE', 'A_IMAGE', 'B_IMAGE', 'KRON_RADIUS', 'THETA_IMAGE', 'SNR_WIN']
        
        bkg_mask = PY_SEx.PS(FITS_obj=fname, SExParam=SEx_param, GAIN_KEY='GAIN', SATUR_KEY='SATURATE',
                             BACK_TYPE='AUTO', BACK_VALUE=0., BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=1.5,
                             DETECT_MINAREA=5, DETECT_MAXAREA=0, DEBLEND_MINCONT=.001, BACKPHOTO_TYPE='LOCAL',
                             CHECKIMAGE_TYPE='SEGMENTATION', AddRD=True, ONLY_FLAGS=None, XBoundary=0., YBoundary=0.,
                             MDIR=None, VERBOSE_LEVEL=1)[1][0] == 0
        
        # bkg_mask = PY_SEx.PS(FITS_obj=fname, SExParam=SEx_param, GAIN_KEY='XXX', SATUR_KEY='NOEXIT',
        #                     BACK_TYPE='AUTO', BACK_VALUE=0., BACK_SIZE=256, BACK_FILTERSIZE=3, DETECT_THRESH=1.5,
        #                     DETECT_MINAREA=15, DETECT_MAXAREA=0, DEBLEND_MINCONT=.005, BACKPHOTO_TYPE='LOCAL',
        #                     CHECKIMAGE_TYPE='SEGMENTATION', AddRD=True, ONLY_FLAGS=None, XBoundary=0., YBoundary=0.,
        #                     MDIR=None, VERBOSE_LEVEL=1)[1][0] == 0
        
        return bkg_mask
    
    
def get_signal(fname, nsig=10):
    im_obj = fits.getdata(fname, ext=0).T
    bkgsig = SkyLevel_Estimator.SLE(PixA_obj=im_obj)[1]
    
    signal_mask = im_obj > nsig * bkgsig
    return signal_mask


#Get statistics of decorrelated snr in different regions
def get_snr_stats(dcd_snr):    
    if len(dcd_snr) == 0:
        return np.nan, np.nan, np.nan
    
    q1 = np.percentile(dcd_snr, 25)
    q3 = np.percentile(dcd_snr, 75)
    iqr = q3 - q1
    
    min_i = q1 - 1.5*iqr
    max_i = q3 + 1.5*iqr
    outlier_mask = np.logical_or(dcd_snr < min_i, dcd_snr > max_i)
    
    dcd_snr_mean = np.mean(dcd_snr)
    dcd_snr_std = iqr / 1.349
    dcd_snr_skew = skew(dcd_snr[~outlier_mask])
    
    return dcd_snr_mean, dcd_snr_std, dcd_snr_skew



def get_statistics(maindir, ref_name, sci_name, skysub, logger, npx_boundary=30):
    """Get statistics of the differential SNR map. Outputs are printed to files in maindir/output/.

    Arguments:
    -----------
    maindir : str
        Main directory for the NEXUS Variability Pipeline.
    ref_name : str
        Name of the reference image.
    sci_name : str
        Name of the science image.
    skysub : bool
        Whether the images are sky-subtracted.
    logger : logging.Logger
        Logger for logging messages.
    npx_boundary : int, optional
        Number of pixels to exclude from the boundary of the image when calculating statistics. Default is 30.
    
    """
    
    
    from sfft.utils.pyAstroMatic.PYSEx import PY_SEx
    
    indir = maindir + 'input/'
    outdir = maindir + 'output/'
    
    logger.info('Getting statistics on difference image')

    if skysub:
        fname_lref = indir + '{}.skysub.fits'.format(ref_name)
        fname_lsci = indir + '{}.skysub.fits'.format(sci_name)
    else:
        fname_lref = indir + '{}.fits'.format(ref_name)
        fname_lsci = indir + '{}.fits'.format(sci_name)
        
    fname_mref = indir + '{}.maskin.fits'.format(ref_name)
    fname_msci = indir + '{}.maskin.fits'.format(sci_name)

    fname_decorr_diff_snr = outdir + '{}.sfftdiff.decorr.snr.fits'.format(sci_name)
    
    im_lsci = fits.getdata(fname_lsci, ext=0).T
    im_decorr_diff_snr = fits.getdata(fname_decorr_diff_snr, ext=0).T
    
    im_mref = fits.getdata(fname_mref, ext=0).T.astype(bool)
    im_msci = fits.getdata(fname_msci, ext=0).T.astype(bool)
    im_mask = im_mref | im_msci
    
    
    #Get mask of boundary
    boundary_mask = np.ones(im_lsci.shape).astype(bool)
    boundary_mask[npx_boundary:-npx_boundary, npx_boundary:-npx_boundary] = False
    
    #Get mask of background
    bkg_mask_lsci = get_background(fname_lsci)    
    bkg_mask_lref = get_background(fname_lref)    
    bkg_mask = np.logical_and.reduce((bkg_mask_lsci, bkg_mask_lref, ~boundary_mask, ~im_mask))
    

    #Get mask of source signal (10 sigma above background)
    signal_mask_lsci = get_signal(fname_lsci, 10.)
    signal_mask_lref = get_signal(fname_lref, 10.)
    signal_mask10 = np.logical_and.reduce((signal_mask_lsci, signal_mask_lref, ~boundary_mask, ~im_mask))
    
    #Get mask of source signal (100 sigma above background)
    signal_mask_lsci = get_signal(fname_lsci, 100.)
    signal_mask_lref = get_signal(fname_lref, 100.)
    signal_mask100 = np.logical_and.reduce((signal_mask_lsci, signal_mask_lref, ~boundary_mask, ~im_mask))
    
    #Get mask of source_signal (150 sigma above background)
    signal_mask_lsci = get_signal(fname_lsci, 150.)
    signal_mask_lref = get_signal(fname_lref, 150.)
    signal_mask150 = np.logical_and.reduce((signal_mask_lsci, signal_mask_lref, ~boundary_mask, ~im_mask))
    
    
    #Get statistics for all masks
    decorr_snr_bkg = im_decorr_diff_snr.copy()
    decorr_snr_signal10 = im_decorr_diff_snr.copy()
    decorr_snr_signal100 = im_decorr_diff_snr.copy()
    decorr_snr_signal150 = im_decorr_diff_snr.copy()
    decorr_snr_bkg[~bkg_mask] = np.nan
    decorr_snr_signal10[~signal_mask10] = np.nan
    decorr_snr_signal100[~signal_mask100] = np.nan
    decorr_snr_signal150[~signal_mask150] = np.nan
    

    decorr_snr_bkg_flat = im_decorr_diff_snr[bkg_mask]
    decorr_snr_signal10_flat = im_decorr_diff_snr[signal_mask10]
    decorr_snr_signal100_flat = im_decorr_diff_snr[signal_mask100]
    decorr_snr_signal150_flat = im_decorr_diff_snr[signal_mask150]
    
    
    #All dat
    mean_bkg, std_bkg, skew_bkg = get_snr_stats(decorr_snr_bkg_flat)
    mean_signal10, std_signal10, skew_signal10 = get_snr_stats(decorr_snr_signal10_flat)
    mean_signal100, std_signal100, skew_signal100 = get_snr_stats(decorr_snr_signal100_flat)
    mean_signal150, std_signal150, skew_signal150 = get_snr_stats(decorr_snr_signal150_flat)

    logger.info('------------------------------------------------------')
    logger.info('| Region           | Mean      | Std       | Skew    |')
    logger.info('------------------------------------------------------')
    logger.info('| Background       | %.4f      | %.4f      | %.4f    |' %(mean_bkg, std_bkg, skew_bkg))
    logger.info('| Signal > 10sig   | %.4f      | %.4f      | %.4f    |' %(mean_signal10, std_signal10, skew_signal10))
    logger.info('| Signal > 100sig  | %.4f      | %.4f      | %.4f    |' %(mean_signal100, std_signal100, skew_signal100))
    logger.info('| Signal > 150sig  | %.4f      | %.4f      | %.4f    |' %(mean_signal150, std_signal150, skew_signal150))
    logger.info('------------------------------------------------------')
    
    
    
    #Save SNR images
    fname = outdir + '{}.decorrbkg.fits'.format(sci_name)
    fits.writeto(fname, decorr_snr_bkg.T, overwrite=True)
    
    fname = outdir + '{}.decorrsig10.fits'.format(sci_name)
    fits.writeto(fname, decorr_snr_signal10.T, overwrite=True)
    
    fname = outdir + '{}.decorrsig100.fits'.format(sci_name)
    fits.writeto(fname, decorr_snr_signal100.T, overwrite=True)
    
    fname = outdir + '{}.decorrsig150.fits'.format(sci_name)
    fits.writeto(fname, decorr_snr_signal150.T, overwrite=True)
    
    
    #Save statistics
    fname_stats = outdir + '{}.decorrstats.dat'.format(sci_name)
    with open(fname_stats, 'w+') as f:
        f.write('# MaskType,Mean,STD,Skew\n')
        f.write('Background,%.4f,%.4f,%.4f\n' %(mean_bkg, std_bkg, skew_bkg))
        f.write('Signal10,%.4f,%.4f,%.4f\n' %(mean_signal10, std_signal10, skew_signal10))
        f.write('Signal100,%.4f,%.4f,%.4f\n' %(mean_signal100, std_signal100, skew_signal100))
        f.write('Signal150,%.4f,%.4f,%.4f\n' %(mean_signal150, std_signal150, skew_signal150))
        
    logger.info('Wrote statistics to file %s \n' %fname_stats)
    
    
    
    #Dat with |SNR| < 10
    mean_bkg, std_bkg, skew_bkg = get_snr_stats(decorr_snr_bkg_flat[np.abs(decorr_snr_bkg_flat) < 10])
    mean_signal10, std_signal10, skew_signal10 = get_snr_stats(decorr_snr_signal10_flat[np.abs(decorr_snr_signal10_flat) < 10])
    mean_signal100, std_signal100, skew_signal100 = get_snr_stats(decorr_snr_signal100_flat[np.abs(decorr_snr_signal100_flat) < 10])
    mean_signal150, std_signal150, skew_signal150 = get_snr_stats(decorr_snr_signal150_flat[np.abs(decorr_snr_signal150_flat) < 10])
    
    logger.info('Only pixels with |SNR| < 10')
    logger.info('------------------------------------------------------')
    logger.info('| Region           | Mean      | Std       | Skew    |')
    logger.info('------------------------------------------------------')
    logger.info('| Background       | %.4f      | %.4f      | %.4f    |' %(mean_bkg, std_bkg, skew_bkg))
    logger.info('| Signal > 10sig   | %.4f      | %.4f      | %.4f    |' %(mean_signal10, std_signal10, skew_signal10))
    logger.info('| Signal > 100sig  | %.4f      | %.4f      | %.4f    |' %(mean_signal100, std_signal100, skew_signal100))
    logger.info('| Signal > 150sig  | %.4f      | %.4f      | %.4f    |' %(mean_signal150, std_signal150, skew_signal150))
    logger.info('------------------------------------------------------')
    
    
    #Save statistics
    fname_stats = outdir + '{}.decorrstats.lt10.dat'.format(sci_name)
    with open(fname_stats, 'w+') as f:
        f.write('# MaskType,Mean,STD,Skew\n')
        f.write('Background,%.4f,%.4f,%.4f\n' %(mean_bkg, std_bkg, skew_bkg))
        f.write('Signal10,%.4f,%.4f,%.4f\n' %(mean_signal10, std_signal10, skew_signal10))
        f.write('Signal100,%.4f,%.4f,%.4f\n' %(mean_signal100, std_signal100, skew_signal100))
        f.write('Signal150,%.4f,%.4f,%.4f\n' %(mean_signal150, std_signal150, skew_signal150))
        
    logger.info('Wrote statistics to file %s' %fname_stats)
        
    return