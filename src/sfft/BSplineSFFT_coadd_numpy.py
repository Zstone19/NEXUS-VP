import time
import numpy as np
import os.path as pa
from scipy import signal
from astropy.io import fits

from BSplineSFFT_numpy import SingleSFFTConfigure_Numpy, SingleSFFTConfigure, ElementalSFFTSubtract_Numpy






class ElementalSFFTCoadd:
    @staticmethod
    def ESN(PixA_I, PixA_J, SFFTConfig, wI=1., wJ=1., SFFTSolution=None, Subtract=False, Coadd=False, \
        BACKEND_4SUBTRACT='Numpy', NUM_CPU_THREADS_4SUBTRACT=8, VERBOSE_LEVEL=2):

        if BACKEND_4SUBTRACT == 'Numpy':
            Solution, PixA_COADD = ElementalSFFTSubtract_Numpy.ESSN(PixA_I=PixA_I, PixA_J=PixA_J, \
                SFFTConfig=SFFTConfig, wI=wI, wJ=wJ, SFFTSolution=SFFTSolution, Subtract=Subtract, Coadd=Coadd, \
                VERBOSE_LEVEL=VERBOSE_LEVEL)

        if BACKEND_4SUBTRACT == 'Cupy':
            print('MeLOn ERROR: Use other class for cupy integration!')
        
        return Solution, PixA_COADD
    
    
    
class GeneralSFFTCoadd:
    @staticmethod
    def GSN(PixA_I, PixA_J, PixA_mI, PixA_mJ, wI, wJ, SFFTConfig, ContamMask_I=None, \
        BACKEND_4SUBTRACT='Numpy', NUM_CPU_THREADS_4SUBTRACT=8, VERBOSE_LEVEL=2):

        """
        # Perform image subtraction on I & J with SFFT parameters solved from mI & mJ
        #
        # Arguments:
        # -PixA_I: Image I that will be convolved [NaN-Free]                                    
        # -PixA_J: Image J that won't be convolved [NaN-Free]                                   
        # -PixA_mI: Masked version of Image I. 'same' means it is identical with I [NaN-Free]  
        # -PixA_mJ: Masked version of Image J. 'same' means it is identical with J [NaN-Free]
        # -wI: Weight of Image I (e.g., 1.0)
        # -wJ: Weight of Image J (e.g., 1.0)
        # -SFFTConfig: Configurations of SFFT
        #
        # -ContamMask_I: Contamination Mask of Image I (e.g., Saturation and Bad pixels)
        # -BACKEND_4SUBTRACT: The backend with which you perform SFFT subtraction              | ['Cupy', 'Numpy']
        # -NUM_CPU_THREADS_4SUBTRACT: The number of CPU threads for Numpy-SFFT subtraction     | e.g., 8
        # -VERBOSE_LEVEL: The level of verbosity, can be 0/1/2: QUIET/NORMAL/FULL              | [0, 1, 2]
        #
        """
        
        SFFT_BANNER = r"""
                                __    __    __    __
                               /  \  /  \  /  \  /  \
                              /    \/    \/    \/    \
            █████████████████/  /██/  /██/  /██/  /█████████████████████████
                            /  / \   / \   / \   / \  \____
                           /  /   \_/   \_/   \_/   \    o \__,
                          / _/                       \_____/  `
                          |/
        
                      █████████  ███████████ ███████████ ███████████        
                     ███░░░░░███░░███░░░░░░█░░███░░░░░░█░█░░░███░░░█            
                    ░███    ░░░  ░███   █ ░  ░███   █ ░ ░   ░███  ░ 
                    ░░█████████  ░███████    ░███████       ░███    
                     ░░░░░░░░███ ░███░░░█    ░███░░░█       ░███    
                     ███    ░███ ░███  ░     ░███  ░        ░███    
                    ░░█████████  █████       █████          █████   
                     ░░░░░░░░░  ░░░░░       ░░░░░          ░░░░░         
        
                    Saccadic Fast Fourier Transform (SFFT) algorithm
                    sfft (v1.*) supported by @LeiHu
        
                    GitHub: https://github.com/thomasvrussell/sfft
                    Related Paper: https://arxiv.org/abs/2109.09334
                    
            ████████████████████████████████████████████████████████████████
            
            """
        
        if VERBOSE_LEVEL in [2]:
            print(SFFT_BANNER)
        
        # * Check Size of input images
        tmplst = [PixA_I.shape, PixA_J.shape, PixA_mI.shape, PixA_mI.shape]
        if len(set(tmplst)) > 1:
            raise Exception('MeLOn ERROR: Input images should have same size!')
        
        # * Subtraction Solution derived from input masked image-pair
        Solution = ElementalSFFTCoadd.ESN(PixA_I=PixA_mI, PixA_J=PixA_mJ, wI=wI, wJ=wJ, SFFTConfig=SFFTConfig, \
            SFFTSolution=None, Subtract=False, Coadd=False, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, VERBOSE_LEVEL=VERBOSE_LEVEL)[0]
            
        # * Subtraction of the input image-pair (use above solution)
        PixA_COADD = ElementalSFFTCoadd.ESN(PixA_I=PixA_I, PixA_J=PixA_J, wI=wI, wJ=wJ, SFFTConfig=SFFTConfig, \
            SFFTSolution=Solution, Subtract=False, Coadd=True, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, VERBOSE_LEVEL=VERBOSE_LEVEL)[1]
        
        # * Identify propagated contamination region through convolving I
        ContamMask_CI = None
        if ContamMask_I is not None:
            tSolution = Solution.copy()
            Fpq = SFFTConfig[0]['Fpq']
            tSolution[-Fpq:] = 0.0

            _tmpI = ContamMask_I.astype(np.float64)
            _tmpJ = np.zeros(PixA_J.shape).astype(np.float64)
            _tmpD = ElementalSFFTSubtract.ESN(PixA_I=_tmpI, PixA_J=_tmpJ, wI=wI, wJ=wJ, SFFTConfig=SFFTConfig, \
                SFFTSolution=tSolution, Coadd=True, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
                NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, VERBOSE_LEVEL=VERBOSE_LEVEL)[1]
            
            FTHRESH = -0.001  # emperical value
            ContamMask_CI = _tmpD < FTHRESH
        
        return Solution, PixA_COADD, ContamMask_CI
    
    
    
    
class BSpline_Packet:
    @staticmethod
    def BSP(FITS_REF, FITS_SCI, FITS_mREF, FITS_mSCI, wREF=1., wSCI=1., FITS_COADD=None, FITS_Solution=None, \
        ForceConv='REF', GKerHW=8, KerSpType='Polynomial', KerSpDegree=2, KerIntKnotX=[], KerIntKnotY=[], \
        SEPARATE_SCALING=True, ScaSpType='Polynomial', ScaSpDegree=0, ScaIntKnotX=[], ScaIntKnotY=[], \
        BkgSpType='Polynomial', BkgSpDegree=2, BkgIntKnotX=[], BkgIntKnotY=[], \
        REGULARIZE_KERNEL=False, IGNORE_LAPLACIAN_KERCENT=True, XY_REGULARIZE=None, 
        WEIGHT_REGULARIZE=None, LAMBDA_REGULARIZE=1e-6, BACKEND_4SUBTRACT='Cupy', \
        CUDA_DEVICE_4SUBTRACT='0', MAX_THREADS_PER_BLOCK=8, MINIMIZE_GPU_MEMORY_USAGE=False, \
        NUM_CPU_THREADS_4SUBTRACT=8, VERBOSE_LEVEL=2):
        
        """
        * Parameters of Customized SFFT
        # @ Customized: use a customized masked image-pair and skip the built-in preprocessing image masking.
        #
        # ----------------------------- Computing Enviornment --------------------------------- #
        
        -BACKEND_4SUBTRACT ['Cupy']         # 'Cupy' or 'Numpy'.
                                            # Cupy backend require GPU(s) capable of performing double-precision calculations,
                                            # while Numpy backend is a pure CPU-based backend for sfft subtraction.
                                            # The Cupy backend is generally much faster than Numpy backend.
                                            # NOTE: 'Pycuda' backend is no longer supported since sfft v1.4.0.

        -CUDA_DEVICE_4SUBTRACT ['0']        # it specifies certain GPU device (index) to conduct the subtraction task.
                                            # the GPU devices are usually numbered 0 to N-1 (you may use command nvidia-smi to check).
                                            # NOTE: the argument only works for Cupy backend.

        -MAX_THREADS_PER_BLOCK [8]          # Maximum Threads per Block for CUDA configuration.
                                            # Emperically, the default 8 is generally a good choice.
                                            # NOTE: if one gets error "Entry function 'kmain' uses too much shared data",
                                            #       which means the device cannot provide enough GPU shared memory, 
                                            #       one can try a lower MAX_THREADS_PER_BLOCK, e.g., 4 or 2, to fix.
                                            # NOTE: the argument only works for Cupy backend.

        -MINIMIZE_GPU_MEMORY_USAGE [False]  # Minimize the GPU Memory Usage? can be True or False
                                            # NOTE: doing so (True) can efficiently reduce the total GPU memory usage, 
                                            #       while it would also largely affect the speed of sfft. Please activate this 
                                            #       memory-friendly mode only when memory is insufficient!
                                            # NOTE: the argument only works for Cupy backend.

        -NUM_CPU_THREADS_4SUBTRACT [8]      # it specifies the number of CPU threads used for sfft subtraction within Numpy backend.
                                            # Numpy backend sfft is implemented with pyFFTW and numba, that allow for 
                                            # parallel computing on CPUs.
                                            # NOTE: the argument only works for Numpy backend.
        
        # ----------------------------- SFFT Subtraction --------------------------------- #

        -ForceConv ['REF']                  # it determines which image will be convolved, can be 'REF' or 'SCI'.
                                            # here ForceConv CANNOT be 'AUTO'!

        -GKerHW [8]                         # the given kernel half-width (pix). 

        # spatial variation of matching kernel

        -KerSpType ['Polynomial']           # 'Polynomial' or 'B-Spline'
                                            # Spatial Varaition Type of Matching Kernel

        -KerSpDegree [2]                    # Polynomial / B-Spline Degree of Kernel Spatial Varaition

        -KerIntKnotX [[]]                   # Internal Knots of Kernel B-Spline Spatial Varaition along X

        -KerIntKnotY [[]]                   # Internal Knots of Kernel B-Spline Spatial Varaition along Y

        -ScaSpType ['Polynomial']           # 'Polynomial' or 'B-Spline'
                                            # Spatial Varaition Type of Matching Kernel

        # spatial variation of convolution scaling

        -SEPARATE_SCALING [True]            # True or False
                                            # True: Convolution Scaling (kernel sum) is a separate varaible
                                            # False: Convolution Scaling is entangled with matching kernel

        -ScaSpType ['Polynomial']           # 'Polynomial' or 'B-Spline'
                                            # Spatial Varaition Type of Convolution Scaling

        -ScaSpDegree [0]                    # Polynomial / B-Spline Degree of Scaling Spatial Varaition

        -ScaIntKnotX [[]]                   # Internal Knots of Scaling B-Spline Spatial Varaition along X

        -ScaIntKnotY [[]]                   # Internal Knots of Scaling B-Spline Spatial Varaition along Y

        P.S. the default configuration means a constant convolution scaling!

        # spatial variation of differential background

        -BkgSpType ['Polynomial']           # 'Polynomial' or 'B-Spline'
                                            # Spatial Varaition Type of Differential Background

        -BkgSpDegree [2]                    # Polynomial / B-Spline Degree of Background Spatial Varaition

        -BkgIntKnotX [[]]                   # Internal Knots of Background B-Spline Spatial Varaition along X

        -BkgIntKnotY [[]]                   # Internal Knots of Background B-Spline Spatial Varaition along Y


        # kernel regularization

        -REGULARIZE_KERNEL [False]          # Regularize matching kernel by applying penalty on
                                            # kernel's second derivates using Laplacian matrix

        -IGNORE_LAPLACIAN_KERCENT [True]    # zero out the rows of Laplacian matrix
                                            # corresponding the kernel center pixels by zeros. 
                                            # If True, the regularization will not impose any penalty 
                                            # on a delta-function-like matching kernel

        -XY_REGULARIZE [None]               # The coordinates at which the matching kernel regularized.
                                            # Numpy array of (x, y) with shape (N_points, 2), 
                                            # where x in (0.5, NX+0.5) and y in (0.5, NY+0.5)

        -WEIGHT_REGULARIZE [None]           # The weights of the coordinates sampled for regularization.
                                            # Use 1d numpy array with shape (XY_REGULARIZE.shape[0])
                                            # Note: -WEIGHT_REGULARIZE = None means uniform weights of 1.0

        -LAMBDA_REGULARIZE [1e-6]           # Tunning paramater lambda for regularization
                                            # it controls the strength of penalty on kernel overfitting

        # ----------------------------- Input & Output --------------------------------- #

        -FITS_REF []                        # File path of input reference image.

        -FITS_SCI []                        # File path of input science image.

        -FITS_mREF []                       # File path of input masked reference image (NaN-free).

        -FITS_mSCI []                       # File path of input masked science image (NaN-free).

        -FITS_DIFF [None]                   # File path of output difference image.

        -FITS_Solution [None]               # File path of the solution of the linear system.
                                            # it is an array of (..., a_ijab, ... b_pq, ...).

        # ----------------------------- Miscellaneous --------------------------------- #
        
        -VERBOSE_LEVEL [2]                  # The level of verbosity, can be [0, 1, 2]
                                            # 0/1/2: QUIET/NORMAL/FULL mode

        # Important Notice:
        #
        # a): if reference is convolved in SFFT (-ForceConv='REF'), then DIFF = SCI - Convolved_REF
        #     [the difference image is expected to have PSF & flux zero-point consistent with science image]
        #
        # b): if science is convolved in SFFT (-ForceConv='SCI'), then DIFF = Convolved_SCI - REF
        #     [the difference image is expected to have PSF & flux zero-point consistent with reference image]
        #
        # Remarks: this convention is to guarantee that transients emerge on science image 
        #          always show a positive signal on difference images.
        #
        """

        # * Read input images
        PixA_REF = fits.getdata(FITS_REF, ext=0).T
        PixA_SCI = fits.getdata(FITS_SCI, ext=0).T
        PixA_mREF = fits.getdata(FITS_mREF, ext=0).T
        PixA_mSCI = fits.getdata(FITS_mSCI, ext=0).T

        if not PixA_REF.flags['C_CONTIGUOUS']:
            PixA_REF = np.ascontiguousarray(PixA_REF, np.float64)
        else: PixA_REF = PixA_REF.astype(np.float64)

        if not PixA_SCI.flags['C_CONTIGUOUS']:
            PixA_SCI = np.ascontiguousarray(PixA_SCI, np.float64)
        else: PixA_SCI = PixA_SCI.astype(np.float64)

        if not PixA_mREF.flags['C_CONTIGUOUS']:
            PixA_mREF = np.ascontiguousarray(PixA_mREF, np.float64)
        else: PixA_mREF = PixA_mREF.astype(np.float64)

        if not PixA_mSCI.flags['C_CONTIGUOUS']:
            PixA_mSCI = np.ascontiguousarray(PixA_mSCI, np.float64)
        else: PixA_mSCI = PixA_mSCI.astype(np.float64)

        NaNmask_U = None
        NaNmask_REF = np.isnan(PixA_REF)
        NaNmask_SCI = np.isnan(PixA_SCI)
        if NaNmask_REF.any() or NaNmask_SCI.any():
            NaNmask_U = np.logical_or(NaNmask_REF, NaNmask_SCI)

        assert np.sum(np.isnan(PixA_mREF)) == 0
        assert np.sum(np.isnan(PixA_mSCI)) == 0
        
        assert ForceConv in ['REF', 'SCI']
        ConvdSide = ForceConv
        KerHW = GKerHW

        # * Choose GPU device for Cupy backend
        if BACKEND_4SUBTRACT == 'Cupy':
            import cupy as cp
            device = cp.cuda.Device(int(CUDA_DEVICE_4SUBTRACT))
            device.use()
        
        # * Compile Functions in SFFT Subtraction
        if VERBOSE_LEVEL in [0, 1, 2]:
            print('MeLOn CheckPoint: TRIGGER Function Compilations of SFFT-COADDITION!')

        Tcomp_start = time.time()
        SFFTConfig = SingleSFFTConfigure.SSN(NX=PixA_REF.shape[0], NY=PixA_REF.shape[1], KerHW=KerHW, \
            KerSpType=KerSpType, KerSpDegree=KerSpDegree, KerIntKnotX=KerIntKnotX, KerIntKnotY=KerIntKnotY, \
            SEPARATE_SCALING=SEPARATE_SCALING, ScaSpType=ScaSpType, ScaSpDegree=ScaSpDegree, \
            ScaIntKnotX=ScaIntKnotX, ScaIntKnotY=ScaIntKnotY, BkgSpType=BkgSpType, \
            BkgSpDegree=BkgSpDegree, BkgIntKnotX=BkgIntKnotX, BkgIntKnotY=BkgIntKnotY, \
            REGULARIZE_KERNEL=REGULARIZE_KERNEL, IGNORE_LAPLACIAN_KERCENT=IGNORE_LAPLACIAN_KERCENT, \
            XY_REGULARIZE=XY_REGULARIZE, WEIGHT_REGULARIZE=WEIGHT_REGULARIZE, \
            LAMBDA_REGULARIZE=LAMBDA_REGULARIZE, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
            MAX_THREADS_PER_BLOCK=MAX_THREADS_PER_BLOCK, MINIMIZE_GPU_MEMORY_USAGE=MINIMIZE_GPU_MEMORY_USAGE, \
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, VERBOSE_LEVEL=VERBOSE_LEVEL)

        if VERBOSE_LEVEL in [1, 2]:
            _message = 'FUNCTION COMPILATIONS OF SFFT-COADDITION TAKES [%.3f s]' %(time.time() - Tcomp_start)
            print('\nMeLOn Report: %s \n' %_message)

        # * Perform SFFT Coaddition
        if ConvdSide == 'REF':
            PixA_mI, PixA_mJ = PixA_mREF, PixA_mSCI
            wI, wJ = wREF, wSCI
            if NaNmask_U is not None:
                PixA_I, PixA_J = PixA_REF.copy(), PixA_SCI.copy()
                PixA_I[NaNmask_U] = PixA_mI[NaNmask_U]
                PixA_J[NaNmask_U] = PixA_mJ[NaNmask_U]
            else: PixA_I, PixA_J = PixA_REF, PixA_SCI

        if ConvdSide == 'SCI':
            PixA_mI, PixA_mJ = PixA_mSCI, PixA_mREF
            wI, wJ = wSCI, wREF
            if NaNmask_U is not None:
                PixA_I, PixA_J = PixA_SCI.copy(), PixA_REF.copy()
                PixA_I[NaNmask_U] = PixA_mI[NaNmask_U]
                PixA_J[NaNmask_U] = PixA_mJ[NaNmask_U]
            else: PixA_I, PixA_J = PixA_SCI, PixA_REF
        
        if VERBOSE_LEVEL in [0, 1, 2]:
            print('MeLOn CheckPoint: TRIGGER SFFT-COADDITION!')

        Tsub_start = time.time()
        _tmp = GeneralSFFTCoadd.GSN(PixA_I=PixA_I, PixA_J=PixA_J, PixA_mI=PixA_mI, PixA_mJ=PixA_mJ, \
            wI=wI, wJ=wJ, SFFTConfig=SFFTConfig, ContamMask_I=None, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, VERBOSE_LEVEL=VERBOSE_LEVEL)

        Solution, PixA_COADD = _tmp[:2]
        if VERBOSE_LEVEL in [1, 2]:
            _message = 'SFFT-COADDITION TAKES [%.3f s]' %(time.time() - Tsub_start)
            print('\nMeLOn Report: %s \n' %_message)
        
        # * Modifications on the difference image
        #   a) when REF is convolved, DIFF = SCI + Conv(REF)
        #      PSF(DIFF) is coincident with PSF(SCI)
        #   b) when SCI is convolved, DIFF = Conv(SCI) + REF
        #      PSF(DIFF) is coincident with PSF(REF)

        if NaNmask_U is not None:
            # ** Mask Union-NaN region
            PixA_COADD[NaNmask_U] = np.nan
        
        
        # * Save difference image
        if FITS_COADD is not None:
            with fits.open(FITS_SCI) as hdl:
                hdl[0].data[:, :] = PixA_COADD.T
                hdl[0].header['NAME_REF'] = (pa.basename(FITS_REF), 'SFFT')
                hdl[0].header['NAME_SCI'] = (pa.basename(FITS_SCI), 'SFFT')
                hdl[0].header['BEND4SUB'] = (BACKEND_4SUBTRACT, 'SFFT')
                hdl[0].header['CONVD'] = (ConvdSide, 'SFFT')
                hdl[0].header['KERHW'] = (KerHW, 'SFFT')

                hdl[0].header['KSPTYPE'] = (str(KerSpType), 'SFFT')
                hdl[0].header['KSPDEG'] = (KerSpDegree, 'SFFT')
                hdl[0].header['NKIKX'] = (len(KerIntKnotX), 'SFFT')
                for i, knot in enumerate(KerIntKnotX):
                    hdl[0].header['KIKX%d' %i] = (knot, 'SFFT')
                hdl[0].header['NKIKY'] = (len(KerIntKnotY), 'SFFT')
                for i, knot in enumerate(KerIntKnotY):
                    hdl[0].header['KIKY%d' %i] = (knot, 'SFFT')

                hdl[0].header['SEPSCA'] = (str(SEPARATE_SCALING), 'SFFT')
                if SEPARATE_SCALING:
                    hdl[0].header['SSPTYPE'] = (str(ScaSpType), 'SFFT')
                    hdl[0].header['SSPDEG'] = (ScaSpDegree, 'SFFT')
                    hdl[0].header['NSIKX'] = (len(ScaIntKnotX), 'SFFT')
                    for i, knot in enumerate(ScaIntKnotX):
                        hdl[0].header['SIKX%d' %i] = (knot, 'SFFT')
                    hdl[0].header['NSIKY'] = (len(ScaIntKnotY), 'SFFT')
                    for i, knot in enumerate(ScaIntKnotY):
                        hdl[0].header['SIKY%d' %i] = (knot, 'SFFT')

                hdl[0].header['BSPTYPE'] = (str(BkgSpType), 'SFFT')
                hdl[0].header['BSPDEG'] = (BkgSpDegree, 'SFFT')
                hdl[0].header['NBIKX'] = (len(BkgIntKnotX), 'SFFT')
                for i, knot in enumerate(BkgIntKnotX):
                    hdl[0].header['BIKX%d' %i] = (knot, 'SFFT')
                hdl[0].header['NBIKY'] = (len(BkgIntKnotY), 'SFFT')
                for i, knot in enumerate(BkgIntKnotY):
                    hdl[0].header['BIKY%d' %i] = (knot, 'SFFT')
                
                hdl[0].header['BSPTYPE'] = (str(BkgSpType), 'SFFT')
                hdl[0].header['BSPDEG'] = (BkgSpDegree, 'SFFT')
                hdl[0].header['NBIKX'] = (len(BkgIntKnotX), 'SFFT')
                for i, knot in enumerate(BkgIntKnotX):
                    hdl[0].header['BIKX%d' %i] = (knot, 'SFFT')
                hdl[0].header['NBIKY'] = (len(BkgIntKnotY), 'SFFT')
                for i, knot in enumerate(BkgIntKnotY):
                    hdl[0].header['BIKY%d' %i] = (knot, 'SFFT')
                
                hdl[0].header['REGKER'] = (str(REGULARIZE_KERNEL), 'SFFT')
                hdl[0].header['ILKC'] = (str(IGNORE_LAPLACIAN_KERCENT), 'SFFT')
                if XY_REGULARIZE is None: hdl[0].header['NREG'] = (-1, 'SFFT')
                else: hdl[0].header['NREG'] = (XY_REGULARIZE.shape[0], 'SFFT')
                if WEIGHT_REGULARIZE is None: hdl[0].header['REGW'] = ('UNIFORM', 'SFFT')
                else: hdl[0].header['REGW'] = ('SPECIFIED', 'SFFT')
                hdl[0].header['REGLAMB'] = (LAMBDA_REGULARIZE, 'SFFT')
                hdl.writeto(FITS_COADD, overwrite=True)
        
        # * Save solution array
        if FITS_Solution is not None:
            
            phdu = fits.PrimaryHDU()
            phdu.header['NAME_REF'] = (pa.basename(FITS_REF), 'SFFT')
            phdu.header['NAME_SCI'] = (pa.basename(FITS_SCI), 'SFFT')
            phdu.header['BEND4SUB'] = (BACKEND_4SUBTRACT, 'SFFT')
            phdu.header['CONVD'] = (ConvdSide, 'SFFT')
            phdu.header['KERHW'] = (KerHW, 'SFFT')
        
            phdu.header['KSPTYPE'] = (str(KerSpType), 'SFFT')
            phdu.header['KSPDEG'] = (KerSpDegree, 'SFFT')
            phdu.header['NKIKX'] = (len(KerIntKnotX), 'SFFT')
            for i, knot in enumerate(KerIntKnotX):
                phdu.header['KIKX%d' %i] = (knot, 'SFFT')
            phdu.header['NKIKY'] = (len(KerIntKnotY), 'SFFT')
            for i, knot in enumerate(KerIntKnotY):
                phdu.header['KIKY%d' %i] = (knot, 'SFFT')

            phdu.header['SEPSCA'] = (str(SEPARATE_SCALING), 'SFFT')
            if SEPARATE_SCALING:
                phdu.header['SSPTYPE'] = (str(ScaSpType), 'SFFT')
                phdu.header['SSPDEG'] = (ScaSpDegree, 'SFFT')
                phdu.header['NSIKX'] = (len(ScaIntKnotX), 'SFFT')
                for i, knot in enumerate(ScaIntKnotX):
                    phdu.header['SIKX%d' %i] = (knot, 'SFFT')
                phdu.header['NSIKY'] = (len(ScaIntKnotY), 'SFFT')
                for i, knot in enumerate(ScaIntKnotY):
                    phdu.header['SIKY%d' %i] = (knot, 'SFFT')

            phdu.header['BSPTYPE'] = (str(BkgSpType), 'SFFT')
            phdu.header['BSPDEG'] = (BkgSpDegree, 'SFFT')
            phdu.header['NBIKX'] = (len(BkgIntKnotX), 'SFFT')
            for i, knot in enumerate(BkgIntKnotX):
                phdu.header['BIKX%d' %i] = (knot, 'SFFT')
            phdu.header['NBIKY'] = (len(BkgIntKnotY), 'SFFT')
            for i, knot in enumerate(BkgIntKnotY):
                phdu.header['BIKY%d' %i] = (knot, 'SFFT')
            
            phdu.header['REGKER'] = (str(REGULARIZE_KERNEL), 'SFFT')
            phdu.header['ILKC'] = (str(IGNORE_LAPLACIAN_KERCENT), 'SFFT')
            if XY_REGULARIZE is None: phdu.header['NREG'] = (-1, 'SFFT')
            else: phdu.header['NREG'] = (XY_REGULARIZE.shape[0], 'SFFT')
            if WEIGHT_REGULARIZE is None: phdu.header['REGW'] = ('UNIFORM', 'SFFT')
            else: phdu.header['REGW'] = ('SPECIFIED', 'SFFT')
            phdu.header['REGLAMB'] = (LAMBDA_REGULARIZE, 'SFFT')
            
            phdu.header['N0'] = (SFFTConfig[0]['N0'], 'SFFT')
            phdu.header['N1'] = (SFFTConfig[0]['N1'], 'SFFT')
            phdu.header['W0'] = (SFFTConfig[0]['w0'], 'SFFT')
            phdu.header['W1'] = (SFFTConfig[0]['w1'], 'SFFT')
            phdu.header['DK'] = (SFFTConfig[0]['DK'], 'SFFT')
            phdu.header['DB'] = (SFFTConfig[0]['DB'], 'SFFT')
            if SEPARATE_SCALING: 
                phdu.header['DS'] = (SFFTConfig[0]['DS'], 'SFFT')
            
            phdu.header['L0'] = (SFFTConfig[0]['L0'], 'SFFT')
            phdu.header['L1'] = (SFFTConfig[0]['L1'], 'SFFT')
            phdu.header['FAB'] = (SFFTConfig[0]['Fab'], 'SFFT')
            phdu.header['FI'] = (SFFTConfig[0]['Fi'], 'SFFT')
            phdu.header['FJ'] = (SFFTConfig[0]['Fj'], 'SFFT')
            phdu.header['FIJ'] = (SFFTConfig[0]['Fij'], 'SFFT')
            phdu.header['FP'] = (SFFTConfig[0]['Fp'], 'SFFT')
            phdu.header['FQ'] = (SFFTConfig[0]['Fq'], 'SFFT')
            phdu.header['FPQ'] = (SFFTConfig[0]['Fpq'], 'SFFT')
        
            if SEPARATE_SCALING and ScaSpDegree > 0:
                phdu.header['SCAFI'] = (SFFTConfig[0]['ScaFi'], 'SFFT')
                phdu.header['SCAFJ'] = (SFFTConfig[0]['ScaFj'], 'SFFT')
                phdu.header['SCAFIJ'] = (SFFTConfig[0]['ScaFij'], 'SFFT')
            phdu.header['FIJAB'] = (SFFTConfig[0]['Fijab'], 'SFFT')
            
            phdu.header['NEQ'] = (SFFTConfig[0]['NEQ'], 'SFFT')
            phdu.header['NEQT'] = (SFFTConfig[0]['NEQt'], 'SFFT')
            PixA_Solution = Solution.reshape((-1, 1))
            phdu.data = PixA_Solution.T
            fits.HDUList([phdu]).writeto(FITS_Solution, overwrite=True)
        
        return Solution, PixA_COADD
