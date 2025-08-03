import time
import math
import numpy as np
import os.path as pa
from scipy import signal
from astropy.io import fits
from scipy.interpolate import BSpline

from astropy.convolution import convolve, convolve_fft
from mkl_fft import _numpy_fft as nfft
import pyfftw

from multi_proc import Multi_Proc
from mpire import WorkerPool
from tqdm import tqdm
# version: Mar 18, 2025

class SingleSFFTConfigure_Numpy:
    @staticmethod
    def SSCN(NX, NY, KerHW=8, KerSpType='Polynomial', KerSpDegree=2, KerIntKnotX=[], KerIntKnotY=[], \
        SEPARATE_SCALING=True, ScaSpType='Polynomial', ScaSpDegree=0, ScaIntKnotX=[], ScaIntKnotY=[], \
        BkgSpType='Polynomial', BkgSpDegree=2, BkgIntKnotX=[], BkgIntKnotY=[], \
        REGULARIZE_KERNEL=False, IGNORE_LAPLACIAN_KERCENT=True, XY_REGULARIZE=None, WEIGHT_REGULARIZE=None, \
        LAMBDA_REGULARIZE=1e-6, 
        NUM_CPU_THREADS_4SUBTRACT=8, MAX_THREADS_PER_BLOCK=8, MINIMIZE_GPU_MEMORY_USAGE=False, logger=None, VERBOSE_LEVEL=2):

        import numba as nb
        nb.set_num_threads = NUM_CPU_THREADS_4SUBTRACT

        N0, N1 = int(NX), int(NY)
        w0, w1 = int(KerHW), int(KerHW)

        SCALE = np.float64(1/(N0*N1))     # Scale of Image-Size
        SCALE_L = np.float64(1/SCALE)     # Reciprocal Scale of Image-Size

        # kernel spatial variation
        DK = int(KerSpDegree)
        assert DK >= 0
        assert KerSpType in ['Polynomial', 'B-Spline']
        if KerSpType == 'B-Spline' and DK == 0:
            assert len(KerIntKnotX) == 0  # otherwise, discontinuity
            assert len(KerIntKnotY) == 0  # otherwise, discontinuity
        
        # scaling spatial variation
        if SEPARATE_SCALING:
            DS = int(ScaSpDegree)
            assert DS >= 0
            assert ScaSpType in ['Polynomial', 'B-Spline']
            if ScaSpType == 'B-Spline' and DS == 0:
                assert len(ScaIntKnotX) == 0  # otherwise, discontinuity
                assert len(ScaIntKnotY) == 0  # otherwise, discontinuity

        # Remarks on SCALING_MODE
        # SEPARATE_SCALING & ScaSpDegree >>>      SCALING_MODE
        #        N         &     any     >>>       'ENTANGLED'
        #        Y         &      0      >>>   'SEPARATE-CONSTANT'
        #        Y         &     > 0     >>>   'SEPARATE-VARYING'

        SCALING_MODE = None
        if not SEPARATE_SCALING:
            SCALING_MODE = 'ENTANGLED'
        elif ScaSpDegree == 0:
            SCALING_MODE = 'SEPARATE-CONSTANT'
        else: SCALING_MODE = 'SEPARATE-VARYING'
        assert SCALING_MODE is not None

        if SCALING_MODE == 'SEPARATE-CONSTANT':
            assert KerSpDegree != 0   # otherwise, reduced to ENTANGLED

        if SCALING_MODE == 'SEPARATE-VARYING':
            # force to activate MINIMIZE_GPU_MEMORY_USAGE
            #assert MINIMIZE_GPU_MEMORY_USAGE

            if KerSpType == 'Polynomial' and ScaSpType == 'Polynomial':
                assert ScaSpDegree != KerSpDegree   # otherwise, reduced to ENTANGLED
            
            if KerSpType == 'B-Spline' and ScaSpType == 'B-Spline':
                if np.all(KerIntKnotX == ScaIntKnotX) and np.all(KerIntKnotY == ScaIntKnotY):
                    assert ScaSpDegree != KerSpDegree   # otherwise, reduced to ENTANGLED
        
        # background spatial variation
        DB = int(BkgSpDegree)
        assert DB >= 0
        assert BkgSpType in ['Polynomial', 'B-Spline']
        if BkgSpType == 'B-Spline' and BkgSpDegree == 0:
            assert len(BkgIntKnotX) == 0  # otherwise, discontinuity
            assert len(BkgIntKnotY) == 0  # otherwise, discontinuity

        # NOTE input image should not has dramatically small size
        assert N0 > MAX_THREADS_PER_BLOCK and N1 > MAX_THREADS_PER_BLOCK

        if REGULARIZE_KERNEL:
            assert XY_REGULARIZE is not None
            assert len(XY_REGULARIZE.shape) == 2 and XY_REGULARIZE.shape[1] == 2

        if VERBOSE_LEVEL in [1, 2]:
            logger.info('--//--//--//--//-- TRIGGER SFFT COMPILATION [Numpy] --//--//--//--//--')

            if KerSpType == 'Polynomial':
                logger.info('---//--- Polynomial Kernel | KerSpDegree %d | KerHW %d ---//---' %(KerSpDegree, KerHW))
                if not SEPARATE_SCALING:
                    logger.info('---//--- [ENTANGLED] Polynomial Scaling | KerSpDegree %d ---//---' %KerSpDegree)

            if KerSpType == 'B-Spline':
                logger.info('---//--- B-Spline Kernel | Internal Knots %d,%d | KerSpDegree %d | KerHW %d ---//---' \
                      %(len(KerIntKnotX), len(KerIntKnotY), KerSpDegree, KerHW))
                if not SEPARATE_SCALING: 
                    logger.info('---//--- [ENTANGLED] B-Spline Scaling | Internal Knots %d,%d | KerSpDegree %d ---//---' \
                          %(len(KerIntKnotX), len(KerIntKnotY), KerSpDegree))
            
            if SEPARATE_SCALING:
                if ScaSpType == 'Polynomial':
                    logger.info('---//--- [SEPARATE] Polynomial Scaling | ScaSpDegree %d ---//---' %ScaSpDegree)
                
                if ScaSpType == 'B-Spline':
                    logger.info('---//--- [SEPARATE] B-Spline Scaling | Internal Knots %d,%d | ScaSpDegree %d ---//---' \
                          %(len(ScaIntKnotX), len(ScaIntKnotY), ScaSpDegree))
            
            if BkgSpType == 'Polynomial':
                logger.info('---//--- Polynomial Background | BkgSpDegree %d ---//---' %BkgSpDegree)
            
            if BkgSpType == 'B-Spline':
                logger.info('---//--- B-Spline Background | Internal Knots %d,%d | BkgSpDegree %d ---//---' \
                    %(len(BkgIntKnotX), len(BkgIntKnotY), BkgSpDegree))

        SFFTParam_dict = {}
        SFFTParam_dict['KerHW'] = KerHW
        SFFTParam_dict['KerSpType'] = KerSpType
        SFFTParam_dict['KerSpDegree'] = KerSpDegree
        SFFTParam_dict['KerIntKnotX'] = KerIntKnotX
        SFFTParam_dict['KerIntKnotY'] = KerIntKnotY

        SFFTParam_dict['SEPARATE_SCALING'] = SEPARATE_SCALING
        if SEPARATE_SCALING:
            SFFTParam_dict['ScaSpType'] = ScaSpType
            SFFTParam_dict['ScaSpDegree'] = ScaSpDegree
            SFFTParam_dict['ScaIntKnotX'] = ScaIntKnotX
            SFFTParam_dict['ScaIntKnotY'] = ScaIntKnotY

        SFFTParam_dict['BkgSpType'] = BkgSpType
        SFFTParam_dict['BkgSpDegree'] = BkgSpDegree
        SFFTParam_dict['BkgIntKnotX'] = BkgIntKnotX
        SFFTParam_dict['BkgIntKnotY'] = BkgIntKnotY

        SFFTParam_dict['REGULARIZE_KERNEL'] = REGULARIZE_KERNEL
        SFFTParam_dict['IGNORE_LAPLACIAN_KERCENT'] = IGNORE_LAPLACIAN_KERCENT
        SFFTParam_dict['XY_REGULARIZE'] = XY_REGULARIZE
        SFFTParam_dict['WEIGHT_REGULARIZE'] = WEIGHT_REGULARIZE
        SFFTParam_dict['LAMBDA_REGULARIZE'] = LAMBDA_REGULARIZE

        SFFTParam_dict['NUM_CPU_THREADS_4SUBTRACT'] = NUM_CPU_THREADS_4SUBTRACT
        SFFTParam_dict['MAX_THREADS_PER_BLOCK'] = MAX_THREADS_PER_BLOCK
        SFFTParam_dict['MINIMIZE_GPU_MEMORY_USAGE'] = MINIMIZE_GPU_MEMORY_USAGE
        
        # * Make a dictionary for SFFT parameters
        L0 = 2*w0+1                                       # matching-kernel XSize
        L1 = 2*w1+1                                       # matching-kernel YSize
        Fab = L0*L1                                       # dof for index ab

        if KerSpType == 'Polynomial':
            Fi, Fj = -1, -1                               # not independent, placeholder
            Fij = ((DK+1)*(DK+2))//2                      # dof for matching-kernel polynomial index ij 
        
        if KerSpType == 'B-Spline':
            Fi = len(KerIntKnotX) + KerSpDegree + 1       # dof for matching-kernel B-spline index i (control points/coefficients)
            Fj = len(KerIntKnotY) + KerSpDegree + 1       # dof for matching-kernel B-spline index j (control points/coefficients)
            Fij = Fi*Fj                                   # dof for matching-kernel B-spline index ij
        
        if BkgSpType == 'Polynomial':
            Fp, Fq = -1, -1                               # not independent, placeholder
            Fpq = ((DB+1)*(DB+2))//2                      # dof for diff-background polynomial index pq 
        
        if BkgSpType == 'B-Spline':
            Fp = len(BkgIntKnotX) + BkgSpDegree + 1       # dof for diff-background B-spline index p (control points/coefficients)
            Fq = len(BkgIntKnotY) + BkgSpDegree + 1       # dof for diff-background B-spline index q (control points/coefficients)  
            Fpq = Fp*Fq                                   # dof for diff-background B-spline index pq

        if SCALING_MODE == 'SEPARATE-VARYING':
            if ScaSpType == 'Polynomial':
                ScaFi, ScaFj = -1, -1                     # not independent, placeholder
                ScaFij = ((DS+1)*(DS+2))//2               # effective dof for scaling polynomial index ij
            
            if ScaSpType == 'B-Spline':
                ScaFi = len(ScaIntKnotX) + ScaSpDegree + 1    # dof for scaling B-spline index i (control points/coefficients)
                ScaFj = len(ScaIntKnotY) + ScaSpDegree + 1    # dof for scaling B-spline index j (control points/coefficients)
                ScaFij = ScaFi*ScaFj                          # effective dof for scaling B-spline index ij
            
            # Remarks on the scaling effective dof
            # I. current version not support scaling effective dof no higher than kernel variation.
            #    for simplicity, we use trivail zero basis as placeholder so that 
            #    the apparent dof of scaling and kernel are consistent.
            # II. ScaFij = Fij is allowed, e.g.m, B-Spline, same degree and 
            #     same number of internal knots but at different positions.

            assert ScaFij <= Fij
        
        Fijab = Fij*Fab                                   # Linear-System Major side-length
        FOMG, FGAM, FTHE = Fij**2, Fij*Fpq, Fij           # OMG / GAM / THE has shape (dof, N0, N1)
        FPSI, FPHI, FDEL = Fpq*Fij, Fpq**2, Fpq           # PSI / PHI / DEL has shape (dof, N0, N1)
        NEQ = Fij*Fab+Fpq                                 # Linear-System side-length
        
        NEQt = NEQ
        if SCALING_MODE == 'SEPARATE-CONSTANT':
            NEQt = NEQ-Fij+1                    # tweaked Linear-System side-length for constant scaling

        if SCALING_MODE == 'SEPARATE-VARYING':
            NEQt = NEQ-(Fij-ScaFij)             # tweaked Linear-System side-length for polynomial-varying scaling

        SFFTParam_dict['N0'] = N0               # a.k.a, NX
        SFFTParam_dict['N1'] = N1               # a.k.a, NY
        SFFTParam_dict['w0'] = w0               # a.k.a, KerHW
        SFFTParam_dict['w1'] = w1               # a.k.a, KerHW
        SFFTParam_dict['DK'] = DK               # a.k.a, KerSpDegree
        SFFTParam_dict['DB'] = DB               # a.k.a, BkgSpDegree
        if SEPARATE_SCALING: 
            SFFTParam_dict['DS'] = DS           # a.k.a, ScaSpDegree

        SFFTParam_dict['SCALE'] = SCALE
        SFFTParam_dict['SCALE_L'] = SCALE_L

        SFFTParam_dict['L0'] = L0
        SFFTParam_dict['L1'] = L1
        SFFTParam_dict['Fab'] = Fab
        SFFTParam_dict['Fi'] = Fi
        SFFTParam_dict['Fj'] = Fj
        SFFTParam_dict['Fij'] = Fij
        SFFTParam_dict['Fp'] = Fp
        SFFTParam_dict['Fq'] = Fq
        SFFTParam_dict['Fpq'] = Fpq

        if SCALING_MODE == 'SEPARATE-VARYING':
            SFFTParam_dict['ScaFi'] = ScaFi
            SFFTParam_dict['ScaFj'] = ScaFj
            SFFTParam_dict['ScaFij'] = ScaFij        
        SFFTParam_dict['Fijab'] = Fijab
        
        SFFTParam_dict['FOMG'] = FOMG
        SFFTParam_dict['FGAM'] = FGAM
        SFFTParam_dict['FTHE'] = FTHE
        SFFTParam_dict['FPSI'] = FPSI
        SFFTParam_dict['FPHI'] = FPHI
        SFFTParam_dict['FDEL'] = FDEL
        
        SFFTParam_dict['NEQ'] = NEQ
        SFFTParam_dict['NEQt'] = NEQt

        # * Load SFFT CUDA modules
        #   NOTE: Generally, a kernel function is defined without knowledge about Grid-Block-Thread Management.
        #         However, we need to know the size of threads per block if SharedMemory is called.

        SFFTModule_dict = {}
        # ******************************************* FFT ******************************************* #
        
        _strdec = 'c16[:](c16[:])'
        @nb.njit(_strdec, fastmath=True)
        def fft1d(x):
            return np.fft.fft2(x)
        
        SFFTModule_dict['fft1d'] = fft1d
        
        
        _strdec = 'c16[:,:](c16[:,:])'
        @nb.njit(_strdec, fastmath=True)
        def fft2d(x):
            return np.fft.fft2(x)
        
        SFFTModule_dict['fft2d'] = fft2d
        
        
        _strdec = 'c16[:,:,:](c16[:,:,:])'
        @nb.njit(_strdec, fastmath=True)
        def fft3d(x):
            return np.fft.fft2(x)
        
        SFFTModule_dict['fft3d'] = fft3d
        
        
        _strdec = 'c16[:](c16[:])'
        @nb.njit(_strdec, fastmath=True)
        def ifft1d(x):
            return np.fft.ifft2(x)
        
        SFFTModule_dict['ifft1d'] = ifft1d
        
        
        _strdec = 'c16[:,:](c16[:,:])'
        @nb.njit(_strdec, fastmath=True)
        def ifft2d(x):
            return np.fft.ifft2(x)
        
        SFFTModule_dict['ifft2d'] = ifft2d
        
        
        _strdec = 'c16[:,:,:](c16[:,:,:])'
        @nb.njit(_strdec, fastmath=True)
        def ifft3d(x):
            return np.fft.ifft2(x)
        
        SFFTModule_dict['ifft3d'] = ifft3d
        
        # ************************************ Spatial Variation ************************************ #

        # <*****> produce spatial coordinate X/Y/oX/oY-map <*****> #
        _strdec = 'Tuple((i4[:,:], i4[:,:], f8[:,:], f8[:,:]))(i4[:,:], i4[:,:], f8[:,:], f8[:,:])'
        @nb.njit(_strdec, parallel=True)
        def SpatialVariation(PixA_X, PixA_Y, PixA_CX, PixA_CY):
            
            for ROW in nb.prange(N0):
                for COL in nb.prange(N1):
                    PixA_X[ROW][COL] = ROW
                    PixA_Y[ROW][COL] = COL
                    PixA_CX[ROW][COL] = (float(ROW) + 1.0) / N0
                    PixA_CY[ROW][COL] = (float(COL) + 1.0) / N1

            return PixA_X, PixA_Y, PixA_CX, PixA_CY


        SFFTModule_dict['SpatialCoord'] = SpatialVariation

        # <*****> produce Iij <*****> #
        if KerSpType == 'Polynomial':
            
            _strdec = 'f8[:,:,:]' + '(i4[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def SpatialPoly(REF_ij, PixA_CX, PixA_CY, PixA_I, SPixA_Iij):
                
                for ij in nb.prange(Fij):
                    i, j = REF_ij[ij]
                    PixA_kpoly = np.power(PixA_CX, i) * np.power(PixA_CY, j)
                    SPixA_Iij[ij,:,:] = PixA_I * PixA_kpoly
                    
                return SPixA_Iij
            
            SFFTModule_dict['KerSpatial'] = SpatialPoly
        
        if KerSpType == 'B-Spline':
            
            _strdec = 'f8[:,:,:]' + '(i4[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def SpatialBSpline(REF_ij, KerSplBasisX, KerSplBasisY, PixA_I, SPixA_Iij):
                
                for ROW in nb.prange(N0):
                    for COL in nb.prange(N1):
                        for ij in range(Fij):
                            i, j = REF_ij[ij]
                            spl = KerSplBasisX[i,ROW] * KerSplBasisY[j,COL]
                            SPixA_Iij[ij,ROW,COL] = PixA_I[ROW,COL] * spl
                
                return SPixA_Iij
            
            SFFTModule_dict['KerSpatial'] = SpatialBSpline
        
        
        
        if SCALING_MODE == 'SEPARATE-VARYING':

            if ScaSpType == 'Polynomial':

                _strdec = 'f8[:,:,:]' + '(i4[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def ScaSpPoly(ScaREF_ij, PixA_CX, PixA_CY, PixA_I, ScaSPixA_Iij):
                    
                    for ij in nb.prange(Fij):
                        i, j = ScaREF_ij[ij]
                        poly = np.power(PixA_CX, i) * np.power(PixA_CY, j)
                        ScaSPixA_Iij[ij,:,:] = PixA_I * poly
                    
                    return ScaSPixA_Iij

                SFFTModule_dict['ScaSpatial'] = ScaSpPoly


            if ScaSpType == 'B-Spline':

                _strdec = 'f8[:,:,:]' + '(i4[:,:], f8[:,:], f8[:,:], f8[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def ScaSpBSpline(ScaREF_ij, ScaSplBasisX, ScaSplBasisY, PixA_I, ScaSPixA_Iij):
                    
                    for ROW in nb.prange(N0):
                        for COL in nb.prange(N1):
                            for ij in nb.prange(Fij):
                                i, j = ScaREF_ij[ij]
                                spl = ScaSplBasisX[i,ROW] * ScaSplBasisY[j,COL]
                                ScaSPixA_Iij[ij,ROW,COL] = PixA_I[ROW,COL] * spl
                    
                    return ScaSPixA_Iij
    
                SFFTModule_dict['ScaSpatial'] = ScaSpBSpline
        
    
    
        # <*****> produce Tpq <*****> #
        if BkgSpType == 'Polynomial':

            _strdec = 'f8[:,:,:]' + '(i4[:,:], f8[:,:], f8[:,:], f8[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def BkgSpaPoly(REF_pq, PixA_CX, PixA_CY, SPixA_Tpq):
                
                for pq in nb.prange(Fpq):
                    p, q = REF_pq[pq]
                    poly_bterm = np.power(PixA_CX, p) * np.power(PixA_CY, q)
                    SPixA_Tpq[pq,:,:] = poly_bterm
                
                return SPixA_Tpq

            SFFTModule_dict['BkgSpatial'] = BkgSpaPoly
        
        if BkgSpType == 'B-Spline':

            _strdec = 'f8[:,:,:]' + '(i4[:,:], f8[:,:], f8[:,:], f8[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def BkgSpaBSpline(REF_pq, BkgSplBasisX, BkgSplBasisY, SPixA_Tpq):
                
                for ROW in nb.prange(N0):
                    for COL in nb.prange(N1):
                        for pq in range(Fpq):
                            p, q = REF_pq[pq]
                            spl_kterm = BkgSplBasisX[p][ROW] * BkgSplBasisY[q][COL]
                            SPixA_Tpq[pq,ROW,COL] = spl_kterm
                
                return SPixA_Tpq

            SFFTModule_dict['BkgSpatial'] = BkgSpaBSpline


        # ************************************ Constuct Linear System ************************************ #

        # <*****> OMEGA & GAMMA & PSI & PHI & THETA & DELTA <*****> #
        if SCALING_MODE in ['ENTANGLED', 'SEPARATE-CONSTANT']:

            if not MINIMIZE_GPU_MEMORY_USAGE:
                
                # ** Hadamard Product [OMEGA]
                # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
                
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG(SREF_iji0j0, SPixA_FIij, SPixA_CFIij, HpOMG):
                    
                    for i8j8ij in nb.prange(FOMG):
                        i8j8, ij = SREF_iji0j0[i8j8ij]
                        HpOMG[i8j8ij,:,:] = SPixA_FIij[i8j8] * SPixA_CFIij[ij]
                        
                    return HpOMG
                
                
                SFFTModule_dict['HadProd_OMG'] = HadProd_OMG

                # ** Fill Linear-System [OMEGA]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], f8[:,:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_OMG(SREF_ijab, REF_ab, PreOMG, LHMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fijab):
                            
                            # ** analyze index
                            i8j8, a8b8 = SREF_ijab[ROW]
                            ij, ab = SREF_ijab[COL]
                            
                            a8, b8 = REF_ab[a8b8]
                            a, b = REF_ab[ab]
                            idx = i8j8 * Fij + ij
                            
                            # ** define Mod_N0(rho), Mod_N1(eps)
                            tmp = int( np.fmod(float(a8), float(N0)) )
                            if tmp < 0.: tmp += N0
                            MODa8 = tmp

                            tmp = int( np.fmod(float(b8), float(N1)) )
                            if tmp < 0.: tmp += N1
                            MODb8 = tmp
                            
                            tmp = int( np.fmod(float(-a), float(N0)) )
                            if tmp < 0.: tmp += N0
                            MOD_a = tmp
                            
                            tmp = int( np.fmod(float(-b), float(N1)) )
                            if tmp < 0.: tmp += N1
                            MOD_b = tmp
                            
                            tmp = int( np.fmod(float(a8-a), float(N0)) )
                            if tmp < 0.: tmp += N0
                            MODa8_a = tmp
                            
                            tmp = int( np.fmod(float(b8-b), float(N1)) )
                            if tmp < 0.: tmp += N1
                            MODb8_b = tmp
                            
                            # ** fill linear system [A-component]
                            if ((a8 != 0 or b8 != 0) and (a != 0 or b != 0)):
                                LHMAT[ROW, COL] = - PreOMG[idx, MODa8, MODb8] - PreOMG[idx, MOD_a, MOD_b] + PreOMG[idx, MODa8_a, MODb8_b] + PreOMG[idx, 0, 0]    
                                                  
                            if ((a8 == 0 and b8 == 0) and (a != 0 or b != 0)):
                                LHMAT[ROW, COL] = PreOMG[idx, MOD_a, MOD_b] - PreOMG[idx, 0, 0]
                                
                            if ((a8 != 0 or b8 != 0) and (a == 0 and b == 0)):
                                LHMAT[ROW, COL] = PreOMG[idx, MODa8, MODb8] - PreOMG[idx, 0, 0]
                                
                            if ((a8 == 0 and b8 == 0) and (a == 0 and b == 0)):
                                LHMAT[ROW, COL] = PreOMG[idx, 0, 0]
                                
                    return LHMAT              
                            
                SFFTModule_dict['FillLS_OMG'] = FillLS_OMG


                # ** Hadamard Product [GAMMA]
                # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
                
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_GAM(SREF_ijpq, SPixA_FIij, SPixA_CFTpq, HpGAM):
                    
                    for i8j8pq in nb.prange(FGAM):
                        i8j8, pq = SREF_ijpq[i8j8pq]
                        HpGAM[i8j8pq,:,:] = SPixA_FIij[i8j8] * SPixA_CFTpq[pq]
                        
                    return HpGAM
                
                SFFTModule_dict['HadProd_GAM'] = HadProd_GAM


                # ** Fill Linear-System [GAMMA]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], f8[:,:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_GAM(SREF_ijab, REF_ab, PreGAM, LHMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fpq):
                            
                            # ** analyze index
                            i8j8, a8b8 = SREF_ijab[ROW]
                            pq = COL
                            
                            a8, b8 = REF_ab[a8b8]
                            idx = i8j8 * Fpq + pq
                            cCOL = Fijab + COL         # add offset
                            
                            # ** define Mod_N0(rho), Mod_N1(eps)
                            tmp = int( np.fmod(float(a8), float(N0)) )
                            if tmp < 0.: tmp += N0
                            MODa8 = tmp
                            
                            tmp = int( np.fmod(float(b8), float(N1)) )
                            if tmp < 0.: tmp += N1
                            MODb8 = tmp
                            
                            # ** fill linear system [B-component]
                            if (a8 != 0 or b8 != 0):
                                LHMAT[ROW, cCOL] = PreGAM[idx, MODa8, MODb8] - PreGAM[idx, 0, 0]
                                
                            if (a8 == 0 and b8 == 0):
                                LHMAT[ROW, cCOL] = PreGAM[idx, 0, 0]
                                
                    return LHMAT
                
                SFFTModule_dict['FillLS_GAM'] = FillLS_GAM

                # ** Hadamard Product [PSI]
                # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
                
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_PSI(SREF_pqij, SPixA_CFIij, SPixA_FTpq, HpPSI):
                    
                    for p8q8ij in nb.prange(FPSI):
                        p8q8, ij = SREF_pqij[p8q8ij]
                        HpPSI[p8q8ij,:,:] = SPixA_FTpq[p8q8] * SPixA_CFIij[ij]
                        
                    return HpPSI
                
                SFFTModule_dict['HadProd_PSI'] = HadProd_PSI

                # ** Fill Linear-System [PSI]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], f8[:,:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_PSI(SREF_ijab, REF_ab, PrePSI, LHMAT):
                    
                    for ROW in nb.prange(Fpq):
                        for COL in nb.prange(Fijab):
                            
                            # ** analyze index
                            cROW = Fijab + ROW      # add offset
                            p8q8 = ROW
                            
                            ij, ab = SREF_ijab[COL]
                            a, b = REF_ab[ab]
                            idx = p8q8 * Fij + ij
                            
                            # ** define Mod_N0(rho), Mod_N1(eps)
                            tmp = int( np.fmod(float(-a), float(N0)) )
                            if tmp < 0.: tmp += N0
                            MOD_a = tmp
                            
                            tmp = int( np.fmod(float(-b), float(N1)) )
                            if tmp < 0.: tmp += N1
                            MOD_b = tmp
                            
                            # ** fill linear system [B#-component]
                            if (a != 0 or b != 0):
                                LHMAT[cROW, COL] = PrePSI[idx, MOD_a, MOD_b] - PrePSI[idx, 0, 0]
                                
                            if (a == 0 and b == 0):
                                LHMAT[cROW, COL] = PrePSI[idx, 0, 0]
                                
                    return LHMAT
                
                SFFTModule_dict['FillLS_PSI'] = FillLS_PSI


                # ** Hadamard Product [PHI]
                # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_PHI(SREF_pqp0q0, SPixA_FTpq, SPixA_CFTpq, HpPHI):
                    
                    for p8q8pq in nb.prange(FPHI):
                        p8q8, pq = SREF_pqp0q0[p8q8pq]
                        HpPHI[p8q8pq,:,:] = SPixA_FTpq[p8q8] * SPixA_CFTpq[pq]
                        
                    return HpPHI
                
                SFFTModule_dict['HadProd_PHI'] = HadProd_PHI

                # ** Fill Linear-System [PHI]
                _strdec = 'f8[:,:](f8[:,:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_PHI(PrePHI, LHMAT):
                    
                    for ROW in nb.prange(Fpq):
                        for COL in nb.prange(Fpq):
                            
                            # ** analyze index
                            cROW = Fijab + ROW      # add offset
                            cCOL = Fijab + COL      # add offset
                            
                            p8q8 = ROW
                            pq = COL
                            idx = p8q8 * Fpq + pq
                            
                            # ** fill linear system [C-component]
                            LHMAT[cROW, cCOL] = PrePHI[idx, 0, 0]
                            
                    return LHMAT
                
                SFFTModule_dict['FillLS_PHI'] = FillLS_PHI


            if MINIMIZE_GPU_MEMORY_USAGE:
                
                # ** Hadamard Product [𝛀]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG(SREF_iji0j0, SPixA_FIij, SPixA_CFIij, cIdx, cHpOMG):
                    
                    i8j8, ij = SREF_iji0j0[cIdx]
                    cHpOMG[:,:] = SPixA_FIij[i8j8] * SPixA_CFIij[ij]
                        
                    return cHpOMG

                SFFTModule_dict['HadProd_OMG'] = HadProd_OMG



                # ** Fill Linear-System [OMEGA]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], i4, f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_OMG(SREF_ijab, REF_ab, cIdx, cPreOMG, LHMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fijab):
                            
                            # *** analysze index
                            i8j8, a8b8 = SREF_ijab[ROW]
                            ij, ab = SREF_ijab[COL]
                            
                            a8, b8 = REF_ab[a8b8]
                            a, b = REF_ab[ab]
                            idx = i8j8 * Fij + ij
                            
                            if (idx == cIdx):
                                
                                # *** define Mod_N0(rho), Mod_N1(eps)
                                tmp = int(np.fmod(float(a8), float(N0)))
                                if tmp < 0.0: tmp += N0
                                MODa8 = tmp
                                
                                tmp = int(np.fmod(float(b8), float(N1)))
                                if tmp < 0.0: tmp += N1
                                MODb8 = tmp

                                tmp = int(np.fmod(float(-a), float(N0)))
                                if tmp < 0.0: tmp += N0
                                MOD_a = tmp

                                tmp = int(np.fmod(float(-b), float(N1)))
                                if tmp < 0.0: tmp += N1
                                MOD_b = tmp

                                tmp = int(np.fmod(float(a8-a), float(N0)))
                                if tmp < 0.0: tmp += N0
                                MODa8_a = tmp

                                tmp = int(np.fmod(float(b8-b), float(N1)))
                                if tmp < 0.0: tmp += N1
                                MODb8_b = tmp
                                


                                # *** fill linear system [A-component]
                                if ((a8 != 0 or b8 != 0) and (a != 0 or b != 0)):
                                    LHMAT[ROW, COL] = - cPreOMG[MODa8, MODb8] \
                                                      - cPreOMG[MOD_a, MOD_b] \
                                                      + cPreOMG[MODa8_a, MODb8_b] \
                                                      + cPreOMG[0, 0]   # NOTE UPDATE

                                if ((a8 == 0 and b8 == 0) and (a != 0 or b != 0)):
                                    LHMAT[ROW, COL] = cPreOMG[MOD_a, MOD_b] - cPreOMG[0, 0]   # NOTE UPDATE

                                if ((a8 != 0 or b8 != 0) and (a == 0 and b == 0)):
                                    LHMAT[ROW, COL] = cPreOMG[MODa8, MODb8] - cPreOMG[0, 0]   # NOTE UPDATE

                                if ((a8 == 0 and b8 == 0) and (a == 0 and b == 0)):
                                    LHMAT[ROW, COL] = cPreOMG[0, 0]   # NOTE UPDATE
                                    
                    return LHMAT
                
                SFFTModule_dict['FillLS_OMG'] = FillLS_OMG

                # ** Hadamard Product [GAMMA]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_GAM(SREF_ijpq, SPixA_FIij, SPixA_CFTpq, cIdx, cHpGAM):
                    
                    i8j8, pq = SREF_ijpq[cIdx]
                    cHpGAM[:,:] = SPixA_FIij[i8j8] * SPixA_CFTpq[pq]
                        
                    return cHpGAM
                
                SFFTModule_dict['HadProd_GAM'] = HadProd_GAM

                # ** Fill Linear-System [GAMMA]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], i4, f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_GAM(SREF_ijab, REF_ab, cIdx, cPreGAM, LHMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fpq):
                            
                            # *** analyze index
                            i8j8, a8b8 = SREF_ijab[ROW]
                            pq = COL
                            
                            a8, b8 = REF_ab[a8b8]
                            idx = i8j8 * Fpq + pq
                            cCOL = Fijab + COL
                            
                            if (idx == cIdx):
                                
                                # ** define Mod_N0(rho), Mod_N1(eps)
                                tmp = int(np.fmod(float(a8), float(N0)))
                                if tmp < 0.0: tmp += N0
                                MODa8 = tmp
                                
                                tmp = int(np.fmod(float(b8), float(N1)))
                                if tmp < 0.0: tmp += N1
                                MODb8 = tmp
                                
                                # ** fill linear system [B-component]
                                if (a8 != 0 or b8 != 0):
                                    LHMAT[ROW, cCOL] = cPreGAM[MODa8, MODb8] - cPreGAM[0, 0]
                                    
                                if (a8 == 0 and b8 == 0):
                                    LHMAT[ROW, cCOL] = cPreGAM[0, 0]
                                    
                    return LHMAT
                                
                SFFTModule_dict['FillLS_GAM'] = FillLS_GAM


                # ** Hadamard Product [PSI]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HamProd_PSI(SREF_pqij, SPixA_CFIij, SPixA_FTpq, cIdx, cHpPSI):
                    
                    p8q8, ij = SREF_pqij[cIdx]
                    cHpPSI[:,:] = SPixA_FTpq[p8q8] * SPixA_CFIij[ij]
                    
                    return cHpPSI
                
                SFFTModule_dict['HadProd_PSI'] = HamProd_PSI

                # ** Fill Linear-System [PSI]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], i4, f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_PSI(SREF_ijab, REF_ab, cIdx, cPrePSI, LHMAT):
                    
                    for ROW in nb.prange(Fpq):
                        for COL in nb.prange(Fijab):
                            
                            # *** analyze index
                            cROW = Fijab + ROW
                            p8q8 = ROW
                            
                            ij, ab = SREF_ijab[COL]
                            a, b = REF_ab[ab]
                            idx = p8q8 * Fij + ij
                            
                            if (idx == cIdx):
                                
                                # ** define Mod_N0(rho), Mod_N1(eps)
                                tmp = int(np.fmod(float(-a), float(N0)))
                                if tmp < 0.0: tmp += N0
                                MOD_a = tmp
                                
                                tmp = int(np.fmod(float(-b), float(N1)))
                                if tmp < 0.0: tmp += N1
                                MOD_b = tmp
                                
                                # ** fill linear system [B#-component]
                                if (a != 0 or b != 0):
                                    LHMAT[cROW, COL] = cPrePSI[MOD_a, MOD_b] - cPrePSI[0, 0]
                                    
                                if (a == 0 and b == 0):
                                    LHMAT[cROW, COL] = cPrePSI[0, 0]
                                    
                    return LHMAT
            
                SFFTModule_dict['FillLS_PSI'] = FillLS_PSI

                # ** Hadamard Product [PHI]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HamProd_PHI(SREF_pqp0q0, SPixA_FTpq, SPixA_CFTpq, cIdx, cHpPHI):
                    
                    p8q8, pq = SREF_pqp0q0[cIdx]
                    cHpPHI[:,:] = SPixA_FTpq[p8q8] * SPixA_CFTpq[pq]
                    
                    return cHpPHI

                SFFTModule_dict['HadProd_PHI'] = HamProd_PHI

                # ** Fill Linear-System [PHI]
                _strdec = 'f8[:,:](i4, f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_PHI(cIdx, cPrePHI, LHMAT):
                    
                    for ROW in nb.prange(Fpq):
                        for COL in nb.prange(Fpq):
                            
                            cROW = Fijab + ROW     # add offset
                            cCOL = Fijab + COL     # add offset
                            
                            p8q8 = ROW
                            pq = COL
                            idx = p8q8 * Fpq + pq
                            
                            if idx == cIdx:
                                LHMAT[cROW, cCOL] = cPrePHI[0, 0]
                            
                    return LHMAT
                
                SFFTModule_dict['FillLS_PHI'] = FillLS_PHI





            # ** Hadamard Product [THETA]
            _strdec = 'c16[:,:,:](c16[:,:,:], c16[:,:], c16[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def HadProd_THE(SPixA_FIij, PixA_CFJ, HpTHE):
                
                for i8j8 in nb.prange(FTHE):
                    HpTHE[i8j8,:,:] = PixA_CFJ * SPixA_FIij[i8j8]
                    
                return HpTHE
                
            SFFTModule_dict['HadProd_THE'] = HadProd_THE

            # ** Fill Linear-System [THETA]  
            _strdec = 'f8[:](i4[:,:], i4[:,:], f8[:,:,:], f8[:])'
            @nb.njit(_strdec, parallel=True)
            def FillLS_THE(SREF_ijab, REF_ab, PreTHE, RHb):
                
                for ROW in nb.prange(Fijab):
                    
                    # ** analyze index
                    i8j8, a8b8 = SREF_ijab[ROW]
                    a8, b8 = REF_ab[a8b8]
                    idx = i8j8
                    
                    # ** define Mod_N0(rho), Mod_N1(eps)
                    tmp = int(np.fmod(float(a8), float(N0)))
                    if tmp < 0.0: tmp += N0
                    MODa8 = tmp
                    
                    tmp = int(np.fmod(float(b8), float(N1)))
                    if tmp < 0.0: tmp += N1
                    MODb8 = tmp
                    
                    # ** fill linear system [D-component]
                    if (a8 != 0 or b8 != 0):
                        RHb[ROW] = PreTHE[idx, MODa8, MODb8] - PreTHE[idx, 0, 0]
                        
                    if (a8 == 0 and b8 == 0):
                        RHb[ROW] = PreTHE[idx, 0, 0]
                        
                return RHb
        
            SFFTModule_dict['FillLS_THE'] = FillLS_THE

            # ** Hadamard Product [DELTA]
            _strdec = 'c16[:,:,:](c16[:,:,:], c16[:,:], c16[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def HadProd_DEL(SPixA_FTpq, PixA_CFJ, HpDEL):
                
                for p8q8 in nb.prange(FDEL):
                    HpDEL[p8q8, :, :] = PixA_CFJ * SPixA_FTpq[p8q8]
                    
                return HpDEL
                
            SFFTModule_dict['HadProd_DEL'] = HadProd_DEL

            # ** Fill Linear-System [DELTA]
            _strdec = 'f8[:](f8[:,:,:], f8[:])'
            @nb.njit(_strdec, parallel=True)
            def FillLS_DEL(PreDEL, RHb):
                
                for ROW in nb.prange(Fpq):
                    cROW = Fijab + ROW
                    idx = ROW
                    
                    # ** Fill Linear System [E-component]
                    RHb[cROW] = PreDEL[idx, 0, 0]
                    
                return RHb
            
            SFFTModule_dict['FillLS_DEL'] = FillLS_DEL





        if SCALING_MODE == 'SEPARATE-VARYING':
            ###assert MINIMIZE_GPU_MEMORY_USAGE # Force  
            
            if not MINIMIZE_GPU_MEMORY_USAGE:
                
                # ** Hadamard Product [OMEGA_11]
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG11(SREF_iji0j0, SPixA_FIij, SPixA_CFIij, HpOMG11):
                    
                    for cIdx in nb.prange(FOMG):                    
                        i8j8, ij = SREF_iji0j0[cIdx]
                        HpOMG11[cIdx,:,:] = SPixA_FIij[i8j8] * SPixA_CFIij[ij]
                    
                    return HpOMG11
                
                SFFTModule_dict['HadProd_OMG11'] = HadProd_OMG11

                # ** Hadamard Product [OMEGA_01]
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG01(SREF_iji0j0, ScaSPixA_FIij, SPixA_CFIij, HpOMG01):
                    
                    for cIdx in nb.prange(FOMG):
                        i8j8, ij = SREF_iji0j0[cIdx]
                        HpOMG01[cIdx,:,:] = ScaSPixA_FIij[i8j8] * SPixA_CFIij[ij]
                    
                    return HpOMG01
                
                SFFTModule_dict['HadProd_OMG01'] = HadProd_OMG01

                # ** Hadamard Product [OMEGA_10]
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG10(SREF_iji0j0, SPixA_FIij, ScaSPixA_CFIij, HpOMG10):
                    
                    for cIdx in nb.prange(FOMG):
                        i8j8, ij = SREF_iji0j0[cIdx]
                        HpOMG10[cIdx,:,:] = SPixA_FIij[i8j8] * ScaSPixA_CFIij[ij]
                    
                    return HpOMG10
                
                SFFTModule_dict['HadProd_OMG10'] = HadProd_OMG10

                # ** Hadamard Product [OMEGA_00]  # TODO: redundant, only one element used.
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG00(SREF_iji0j0, ScaSPixA_FIij, ScaSPixA_CFIij, HpOMG00):
                    
                    for cIdx in nb.prange(FOMG):
                        i8j8, ij = SREF_iji0j0[cIdx]
                        HpOMG00[cIdx, :,:] = ScaSPixA_FIij[i8j8] * ScaSPixA_CFIij[ij]
                    
                    return HpOMG00
                
                SFFTModule_dict['HadProd_OMG00'] = HadProd_OMG00
                
                
                # ** Fill Linear-System [OMEGA]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_OMG(SREF_ijab, REF_ab, PreOMG11, PreOMG01, PreOMG10, PreOMG00, LHMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fijab):
                            
                            # ** analyze index
                            i8j8, a8b8 = SREF_ijab[ROW]
                            ij, ab = SREF_ijab[COL]
                            
                            a8, b8 = REF_ab[a8b8]
                            a, b = REF_ab[ab]
                            idx = i8j8 * Fij + ij
                            
                                
                            # ** define Mod_N0(rho), Mod_N1(eps)
                            tmp = int(np.fmod(float(a8), float(N0)))
                            if tmp < 0.0: tmp += N0
                            MODa8 = tmp
                            
                            tmp = int(np.fmod(float(b8), float(N1)))
                            if tmp < 0.0: tmp += N1
                            MODb8 = tmp
                            
                            tmp = int(np.fmod(float(-a), float(N0)))
                            if tmp < 0.0: tmp += N0
                            MOD_a = tmp
                            
                            tmp = int(np.fmod(float(-b), float(N1)))
                            if tmp < 0.0: tmp += N1
                            MOD_b = tmp
                            
                            tmp = int(np.fmod(float(a8-a), float(N0)))
                            if tmp < 0.0: tmp += N0
                            MODa8_a = tmp
                            
                            tmp = int(np.fmod(float(b8-b), float(N1)))
                            if tmp < 0.0: tmp += N1
                            MODb8_b = tmp
                            
                            # ** fill linear system [A-component]
                            if ((a8 != 0 or b8 != 0) and (a != 0 or b != 0)):
                                LHMAT[ROW, COL] = - PreOMG11[idx, MODa8, MODb8] \
                                                    - PreOMG11[idx, MOD_a, MOD_b] \
                                                    + PreOMG11[idx, MODa8_a, MODb8_b] \
                                                    + PreOMG11[idx, 0, 0]
                                                    
                            if ((a8 == 0 and b8 == 0) and (a != 0 or b != 0)):
                                LHMAT[ROW, COL] = PreOMG01[idx, MOD_a, MOD_b] - PreOMG01[idx, 0, 0]
                                
                            if ((a8 != 0 or b8 != 0) and (a == 0 and b == 0)):
                                LHMAT[ROW, COL] = PreOMG10[idx, MODa8, MODb8] - PreOMG10[idx, 0, 0]
                                
                            if ((a8 == 0 and b8 == 0) and (a == 0 and b == 0)):
                                LHMAT[ROW, COL] = PreOMG00[idx, 0, 0]
                
                    return LHMAT
    
                SFFTModule_dict['FillLS_OMG'] = FillLS_OMG
    
    
    
                    # ** Hadamard Product [GAMMA_1]
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_GAM1(SREF_ijpq, SPixA_FIij, SPixA_CFTpq, HpGAM1):
                    
                    for cIdx in nb.prange(FGAM):
                        i8j8, pq = SREF_ijpq[cIdx]
                        HpGAM1[cIdx,:,:] = SPixA_FIij[i8j8] * SPixA_CFTpq[pq]
                    
                    return HpGAM1
                
                SFFTModule_dict['HadProd_GAM1'] = HadProd_GAM1

                # ** Hadamard Product [GAMMA_0]  # TODO: redundant, only one element used.
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_GAM0(SREF_ijpq, ScaSPixA_FIij, SPixA_CFTpq, HpGAM0):
                    
                    for cIdx in nb.prange(FGAM):
                        i8j8, pq = SREF_ijpq[cIdx]
                        HpGAM0[cIdx,:,:] = ScaSPixA_FIij[i8j8] * SPixA_CFTpq[pq]
                    
                    return HpGAM0
                
                SFFTModule_dict['HadProd_GAM0'] = HadProd_GAM0

                # ** Fill Linear-System [GAMMA]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], f8[:,:,:], f8[:,:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_GAM(SREF_ijab, REF_ab, PreGAM1, PreGAM0, LHMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fpq):
                            
                            # ** analyze index
                            i8j8, a8b8 = SREF_ijab[ROW]
                            pq = COL
                            
                            a8, b8 = REF_ab[a8b8]
                            idx = i8j8 * Fpq + pq
                            cCOL = Fijab + COL
                            
                            # ** define Mod_N0(rho), Mod_N1(eps)
                            tmp = int(np.fmod(float(a8), float(N0)))
                            if tmp < 0.0: tmp += N0
                            MODa8 = tmp
                            
                            tmp = int(np.fmod(float(b8), float(N1)))
                            if tmp < 0.0: tmp += N1
                            MODb8 = tmp
                            
                            # ** fill linear system [B-component]
                            if (a8 != 0 or b8 != 0):
                                LHMAT[ROW, cCOL] = PreGAM1[idx, MODa8, MODb8] - PreGAM1[idx, 0, 0]
                                
                            if (a8 == 0 and b8 == 0):
                                LHMAT[ROW, cCOL] = PreGAM0[idx, 0, 0]
                                    
                    return LHMAT

                SFFTModule_dict['FillLS_GAM'] = FillLS_GAM
    
    
    
                    # ** Hadamard Product [PSI_1]
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HamProd_PSI1(SREF_pqij, SPixA_CFIij, SPixA_FTpq, HpPSI1):
                    
                    for cIdx in nb.prange(FPSI):
                        p8q8, ij = SREF_pqij[cIdx]
                        HpPSI1[cIdx,:,:] = SPixA_FTpq[p8q8] * SPixA_CFIij[ij]
                    
                    return HpPSI1

                SFFTModule_dict['HadProd_PSI1'] = HamProd_PSI1

                # ** Hadamard Product [PSI_0]  # TODO: redundant, only one element used.
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HamProd_PSI0(SREF_pqij, ScaSPixA_CFIij, SPixA_FTpq, HpPSI0):
                    
                    for cIdx in nb.prange(FPSI):
                        p8q8, ij = SREF_pqij[cIdx]
                        HpPSI0[cIdx,:,:] = SPixA_FTpq[p8q8] * ScaSPixA_CFIij[ij]
                    
                    return HpPSI0
                
                SFFTModule_dict['HadProd_PSI0'] = HamProd_PSI0


                # ** Fill Linear-System [PSI]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], f8[:,:,:], f8[:,:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_PSI(SREF_ijab, REF_ab, PrePSI1, PrePSI0, LHMAT):
                    
                    for ROW in nb.prange(Fpq):
                        for COL in nb.prange(Fijab):
                            
                            # ** analyze index
                            cROW = Fijab + ROW
                            p8q8 = ROW
                            
                            ij, ab = SREF_ijab[COL]
                            a, b = REF_ab[ab]
                            idx = p8q8 * Fij + ij

                            
                            # ** define Mod_N0(rho), Mod_N1(eps)
                            tmp = int(np.fmod(float(-a), float(N0)))
                            if tmp < 0.0: tmp += N0
                            MOD_a = tmp
                            
                            tmp = int(np.fmod(float(-b), float(N1)))
                            if tmp < 0.0: tmp += N1
                            MOD_b = tmp
                            
                            # ** fill linear system [B-component]
                            if (a != 0 or b != 0):
                                LHMAT[cROW, COL] = PrePSI1[idx, MOD_a, MOD_b] - PrePSI1[idx, 0, 0]
                                
                            if (a == 0 and b == 0):
                                LHMAT[cROW, COL] = PrePSI0[idx, 0, 0]

                    return LHMAT                
                
                SFFTModule_dict['FillLS_PSI'] = FillLS_PSI




                # ** Hadamard Product [PHI]
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HamProd_PHI(SREF_pqp0q0, SPixA_FTpq, SPixA_CFTpq, HpPHI):
                    
                    for cIdx in nb.prange(FPHI):
                        p8q8, pq = SREF_pqp0q0[cIdx]
                        HpPHI[cIdx,:,:] = SPixA_FTpq[p8q8] * SPixA_CFTpq[pq]
                    
                    return HpPHI
                
                SFFTModule_dict['HadProd_PHI'] = HamProd_PHI

                # ** Fill Linear-System [PHI]
                _strdec = 'f8[:,:](f8[:,:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_PHI(PrePHI, LHMAT):
                    
                    for ROW in nb.prange(Fpq):
                        for COL in nb.prange(Fpq):
                            
                            # ** analyze index
                            cROW = Fijab + ROW
                            cCOL = Fijab + COL
                            
                            p8q8 = ROW
                            pq = COL
                            idx = p8q8 * Fpq + pq
                            
                            # ** fill linear system [C-component]
                            LHMAT[cROW, cCOL] = PrePHI[idx, 0, 0]
                                
                    return LHMAT
                
                SFFTModule_dict['FillLS_PHI'] = FillLS_PHI
                
            
            if MINIMIZE_GPU_MEMORY_USAGE:
            
                # ** Hadamard Product [OMEGA_11]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG11(SREF_iji0j0, SPixA_FIij, SPixA_CFIij, cIdx, cHpOMG11):
                    
                    i8j8, ij = SREF_iji0j0[cIdx]
                    cHpOMG11[:,:] = SPixA_FIij[i8j8] * SPixA_CFIij[ij]
                    
                    return cHpOMG11
                
                SFFTModule_dict['HadProd_OMG11'] = HadProd_OMG11

                # ** Hadamard Product [OMEGA_01]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG01(SREF_iji0j0, ScaSPixA_FIij, SPixA_CFIij, cIdx, cHpOMG01):
                    
                    i8j8, ij = SREF_iji0j0[cIdx]
                    cHpOMG01[:,:] = ScaSPixA_FIij[i8j8] * SPixA_CFIij[ij]
                    
                    return cHpOMG01
                
                SFFTModule_dict['HadProd_OMG01'] = HadProd_OMG01

                # ** Hadamard Product [OMEGA_10]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG10(SREF_iji0j0, SPixA_FIij, ScaSPixA_CFIij, cIdx, cHpOMG10):
                    
                    i8j8, ij = SREF_iji0j0[cIdx]
                    cHpOMG10[:,:] = SPixA_FIij[i8j8] * ScaSPixA_CFIij[ij]
                    
                    return cHpOMG10
                
                SFFTModule_dict['HadProd_OMG10'] = HadProd_OMG10

                # ** Hadamard Product [OMEGA_00]  # TODO: redundant, only one element used.
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG00(SREF_iji0j0, ScaSPixA_FIij, ScaSPixA_CFIij, cIdx, cHpOMG00):
                    
                    i8j8, ij = SREF_iji0j0[cIdx]
                    cHpOMG00[:,:] = ScaSPixA_FIij[i8j8] * ScaSPixA_CFIij[ij]
                    
                    return cHpOMG00
                
                SFFTModule_dict['HadProd_OMG00'] = HadProd_OMG00


                # ** Fill Linear-System [OMEGA]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], i4, f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_OMG(SREF_ijab, REF_ab, cIdx, cPreOMG11, cPreOMG01, cPreOMG10, cPreOMG00, LHMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fijab):
                            
                            # ** analyze index
                            i8j8, a8b8 = SREF_ijab[ROW]
                            ij, ab = SREF_ijab[COL]
                            
                            a8, b8 = REF_ab[a8b8]
                            a, b = REF_ab[ab]
                            idx = i8j8 * Fij + ij
                            
                            
                            if (idx == cIdx):
                                
                                # ** define Mod_N0(rho), Mod_N1(eps)
                                tmp = int(np.fmod(float(a8), float(N0)))
                                if tmp < 0.0: tmp += N0
                                MODa8 = tmp
                                
                                tmp = int(np.fmod(float(b8), float(N1)))
                                if tmp < 0.0: tmp += N1
                                MODb8 = tmp
                                
                                tmp = int(np.fmod(float(-a), float(N0)))
                                if tmp < 0.0: tmp += N0
                                MOD_a = tmp
                                
                                tmp = int(np.fmod(float(-b), float(N1)))
                                if tmp < 0.0: tmp += N1
                                MOD_b = tmp
                                
                                tmp = int(np.fmod(float(a8-a), float(N0)))
                                if tmp < 0.0: tmp += N0
                                MODa8_a = tmp
                                
                                tmp = int(np.fmod(float(b8-b), float(N1)))
                                if tmp < 0.0: tmp += N1
                                MODb8_b = tmp
                                
                                # ** fill linear system [A-component]
                                if ((a8 != 0 or b8 != 0) and (a != 0 or b != 0)):
                                    LHMAT[ROW, COL] = - cPreOMG11[MODa8, MODb8] \
                                                      - cPreOMG11[MOD_a, MOD_b] \
                                                      + cPreOMG11[MODa8_a, MODb8_b] \
                                                      + cPreOMG11[0, 0]
                                                      
                                if ((a8 == 0 and b8 == 0) and (a != 0 or b != 0)):
                                    LHMAT[ROW, COL] = cPreOMG01[MOD_a, MOD_b] - cPreOMG01[0, 0]
                                    
                                if ((a8 != 0 or b8 != 0) and (a == 0 and b == 0)):
                                    LHMAT[ROW, COL] = cPreOMG10[MODa8, MODb8] - cPreOMG10[0, 0]
                                    
                                if ((a8 == 0 and b8 == 0) and (a == 0 and b == 0)):
                                    LHMAT[ROW, COL] = cPreOMG00[0, 0]
                
                    return LHMAT
    
                SFFTModule_dict['FillLS_OMG'] = FillLS_OMG


                # ** Hadamard Product [GAMMA_1]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_GAM1(SREF_ijpq, SPixA_FIij, SPixA_CFTpq, cIdx, cHpGAM1):
                    
                    i8j8, pq = SREF_ijpq[cIdx]
                    cHpGAM1[:,:] = SPixA_FIij[i8j8] * SPixA_CFTpq[pq]
                    
                    return cHpGAM1
                
                SFFTModule_dict['HadProd_GAM1'] = HadProd_GAM1

                # ** Hadamard Product [GAMMA_0]  # TODO: redundant, only one element used.
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_GAM0(SREF_ijpq, ScaSPixA_FIij, SPixA_CFTpq, cIdx, cHpGAM0):
                    
                    i8j8, pq = SREF_ijpq[cIdx]
                    cHpGAM0[:,:] = ScaSPixA_FIij[i8j8] * SPixA_CFTpq[pq]
                    
                    return cHpGAM0
                
                SFFTModule_dict['HadProd_GAM0'] = HadProd_GAM0

                # ** Fill Linear-System [GAMMA]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], i4, f8[:,:], f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_GAM(SREF_ijab, REF_ab, cIdx, cPreGAM1, cPreGAM0, LHMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fpq):
                            
                            # ** analyze index
                            i8j8, a8b8 = SREF_ijab[ROW]
                            pq = COL
                            
                            a8, b8 = REF_ab[a8b8]
                            idx = i8j8 * Fpq + pq
                            cCOL = Fijab + COL
                            
                            if (idx == cIdx):
                                
                                # ** define Mod_N0(rho), Mod_N1(eps)
                                tmp = int(np.fmod(float(a8), float(N0)))
                                if tmp < 0.0: tmp += N0
                                MODa8 = tmp
                                
                                tmp = int(np.fmod(float(b8), float(N1)))
                                if tmp < 0.0: tmp += N1
                                MODb8 = tmp
                                
                                # ** fill linear system [B-component]
                                if (a8 != 0 or b8 != 0):
                                    LHMAT[ROW, cCOL] = cPreGAM1[MODa8, MODb8] - cPreGAM1[0, 0]
                                    
                                if (a8 == 0 and b8 == 0):
                                    LHMAT[ROW, cCOL] = cPreGAM0[0, 0]
                                    
                    return LHMAT

                SFFTModule_dict['FillLS_GAM'] = FillLS_GAM

                # ** Hadamard Product [PSI_1]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HamProd_PSI1(SREF_pqij, SPixA_CFIij, SPixA_FTpq, cIdx, cHpPSI1):
                    
                    p8q8, ij = SREF_pqij[cIdx]
                    cHpPSI1[:,:] = SPixA_FTpq[p8q8] * SPixA_CFIij[ij]
                    
                    return cHpPSI1

                SFFTModule_dict['HadProd_PSI1'] = HamProd_PSI1

                # ** Hadamard Product [PSI_0]  # TODO: redundant, only one element used.
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HamProd_PSI0(SREF_pqij, ScaSPixA_CFIij, SPixA_FTpq, cIdx, cHpPSI0):
                    
                    p8q8, ij = SREF_pqij[cIdx]
                    cHpPSI0[:,:] = SPixA_FTpq[p8q8] * ScaSPixA_CFIij[ij]
                    
                    return cHpPSI0
                
                SFFTModule_dict['HadProd_PSI0'] = HamProd_PSI0


                # ** Fill Linear-System [PSI]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], i4, f8[:,:], f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_PSI(SREF_ijab, REF_ab, cIdx, cPrePSI1, cPrePSI0, LHMAT):
                    
                    for ROW in nb.prange(Fpq):
                        for COL in nb.prange(Fijab):
                            
                            # ** analyze index
                            cROW = Fijab + ROW
                            p8q8 = ROW
                            
                            ij, ab = SREF_ijab[COL]
                            a, b = REF_ab[ab]
                            idx = p8q8 * Fij + ij
                            
                            if (idx == cIdx):
                                
                                # ** define Mod_N0(rho), Mod_N1(eps)
                                tmp = int(np.fmod(float(-a), float(N0)))
                                if tmp < 0.0: tmp += N0
                                MOD_a = tmp
                                
                                tmp = int(np.fmod(float(-b), float(N1)))
                                if tmp < 0.0: tmp += N1
                                MOD_b = tmp
                                
                                # ** fill linear system [B-component]
                                if (a != 0 or b != 0):
                                    LHMAT[cROW, COL] = cPrePSI1[MOD_a, MOD_b] - cPrePSI1[0, 0]
                                    
                                if (a == 0 and b == 0):
                                    LHMAT[cROW, COL] = cPrePSI0[0, 0]

                    return LHMAT                
                
                SFFTModule_dict['FillLS_PSI'] = FillLS_PSI

                # ** Hadamard Product [PHI]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HamProd_PHI(SREF_pqp0q0, SPixA_FTpq, SPixA_CFTpq, cIdx, cHpPHI):
                    
                    p8q8, pq = SREF_pqp0q0[cIdx]
                    cHpPHI[:,:] = SPixA_FTpq[p8q8] * SPixA_CFTpq[pq]
                    
                    return cHpPHI
                
                SFFTModule_dict['HadProd_PHI'] = HamProd_PHI

                # ** Fill Linear-System [PHI]
                _strdec = 'f8[:,:](i4, f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_PHI(cIdx, cPrePHI, LHMAT):
                    
                    for ROW in nb.prange(Fpq):
                        for COL in nb.prange(Fpq):
                            
                            # ** analyze index
                            cROW = Fijab + ROW
                            cCOL = Fijab + COL
                            
                            p8q8 = ROW
                            pq = COL
                            idx = p8q8 * Fpq + pq
                            
                            # ** fill linear system [C-component]
                            if idx == cIdx:
                                LHMAT[cROW, cCOL] = cPrePHI[0, 0]
                                
                    return LHMAT
                
                SFFTModule_dict['FillLS_PHI'] = FillLS_PHI


            # ** Hadamard Product [THETA_1]
            _strdec = 'c16[:,:,:](c16[:,:,:], c16[:,:], c16[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def HamProd_THE1(SPixA_FIij, PixA_CFJ, HpTHE1):
                
                for i8j8 in nb.prange(FTHE):
                    HpTHE1[i8j8,:,:] = PixA_CFJ * SPixA_FIij[i8j8]
                    
                return HpTHE1
            
            SFFTModule_dict['HadProd_THE1'] = HamProd_THE1


            # ** Hadamard Product [THETA_0]  # TODO: redundant, only one element used.
            _strdec = 'c16[:,:,:](c16[:,:,:], c16[:,:], c16[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def HamProd_THE0(ScaSPixA_FIij, PixA_CFJ, HpTHE0):
                
                for i8j8 in nb.prange(FTHE):
                    HpTHE0[i8j8,:,:] = PixA_CFJ * ScaSPixA_FIij[i8j8]
                    
                return HpTHE0
            
            SFFTModule_dict['HadProd_THE0'] = HamProd_THE0


            # ** Fill Linear-System [THETA]  
            _strdec = 'f8[:](i4[:,:], i4[:,:], f8[:,:,:], f8[:,:,:], f8[:])'
            @nb.njit(_strdec, parallel=True)
            def FillLS_THE(SREF_ijab, REF_ab, PreTHE1, PreTHE0, RHb):
                
                for ROW in nb.prange(Fijab):
                    
                    # ** analyze index
                    i8j8, a8b8 = SREF_ijab[ROW]
                    a8, b8 = REF_ab[a8b8]
                    idx = i8j8
                    
                    # ** define Mod_N0(rho), Mod_N1(eps)
                    tmp = int(np.fmod(float(a8), float(N0)))
                    if tmp < 0.0: tmp += N0
                    MODa8 = tmp
                    
                    tmp = int(np.fmod(float(b8), float(N1)))
                    if tmp < 0.0: tmp += N1
                    MODb8 = tmp
                    
                    # ** fill linear system [D-component]
                    if (a8 != 0 or b8 != 0):
                        RHb[ROW] = PreTHE1[idx, MODa8, MODb8] - PreTHE1[idx, 0, 0]
                        
                    if (a8 == 0 and b8 == 0):
                        RHb[ROW] = PreTHE0[idx, 0, 0]
                        
                return RHb
            
            SFFTModule_dict['FillLS_THE'] = FillLS_THE



            # ** Hadamard Product [DELTA]
            _strdec = 'c16[:,:,:](c16[:,:,:], c16[:,:], c16[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def HamProd_DEL(SPixA_FTpq, PixA_CFJ, HpDEL):
                
                for p8q8 in nb.prange(FDEL):
                    HpDEL[p8q8,:,:] = PixA_CFJ * SPixA_FTpq[p8q8]
                    
                return HpDEL
            
            SFFTModule_dict['HadProd_DEL'] = HamProd_DEL

            # ** Fill Linear-System [DELTA]
            _strdec = 'f8[:](f8[:,:,:], f8[:])'
            @nb.njit(_strdec, parallel=True)
            def FillLS_DEL(PreDEL, RHb):
                
                for ROW in nb.prange(Fpq):
                    
                    # ** analyze index
                    cROW = Fijab + ROW
                    idx = ROW
                    
                    # ** Fill Linear System [E-component]
                    RHb[cROW] = PreDEL[idx, 0, 0]
                    
                return RHb
            
            SFFTModule_dict['FillLS_DEL'] = FillLS_DEL


        # <*****> regularize matrix <*****> #
        if REGULARIZE_KERNEL:
            _strdec = 'i4[:,:](i4[:,:], i4[:], i4[:])'
            @nb.njit(_strdec, parallel=True)
            def Fill_LAPMAT_NonDiagonal(LAPMAT, RRF, CCF):
                
                for ROW in nb.prange(Fab):
                    for COL in nb.prange(Fab):
                        
                        if (ROW != COL):
                            r1 = RRF[ROW]
                            c1 = CCF[ROW]
                            r2 = RRF[COL]
                            c2 = CCF[COL]
                            
                            if (r2 == r1-1 and c2 == c1):
                                LAPMAT[ROW, COL] = -1
                                
                            if (r2 == r1+1 and c2 == c1):
                                LAPMAT[ROW, COL] = -1
                                
                            if (r2 == r1 and c2 == c1-1):
                                LAPMAT[ROW, COL] = -1
                                
                            if (r2 == r1 and c2 == c1+1):
                                LAPMAT[ROW, COL] = -1
                                
                return LAPMAT

            SFFTModule_dict['fill_lapmat_nondiagonal'] = Fill_LAPMAT_NonDiagonal

            c0 = w0*L1+w1
            _strdec = 'i4[:,:](i4[:,:], i4[:,:])'
            @nb.njit(_strdec, parallel=True)
            def Fill_iREGMAT(iREGMAT, LTLMAT):
                
                for ROW in nb.prange(Fab):
                    for COL in nb.prange(Fab):
                        
                        if (ROW != c0 and COL != c0):
                            iREGMAT[ROW, COL] = LTLMAT[ROW, COL] + LTLMAT[COL, ROW] \
                                                - LTLMAT[c0, ROW] - LTLMAT[c0, COL] \
                                                - LTLMAT[ROW, c0] - LTLMAT[COL, c0] \
                                                + 2*LTLMAT[c0, c0]
                                                
                        if (ROW != c0 and COL == c0):
                            iREGMAT[ROW, COL] = LTLMAT[ROW, c0] + LTLMAT[c0, ROW] \
                                                - 2*LTLMAT[c0, c0]
                                                                        
                        if (ROW == c0 and COL != c0):
                            iREGMAT[ROW, COL] = LTLMAT[COL, c0] + LTLMAT[c0, COL] \
                                                - 2*LTLMAT[c0, c0]
                                                
                        if (ROW == c0 and COL == c0):
                            iREGMAT[ROW, COL] = 2*LTLMAT[c0, c0]            
            
                return iREGMAT
            
            SFFTModule_dict['fill_iregmat'] = Fill_iREGMAT



            if SCALING_MODE in ['ENTANGLED', 'SEPARATE-CONSTANT']:

                SCALE2 = SCALE**2
                _strdec = 'f8[:,:](i4[:,:], f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def Fill_REGMAT(iREGMAT, SSTMAT, REGMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fijab):
                            k = int(ROW // Fab)
                            c = int(ROW % Fab)
                            k8 = int(COL // Fab)
                            c8 = int(COL % Fab)
                            
                            REGMAT[ROW, COL] = SCALE2 * SSTMAT[k, k8] * iREGMAT[c, c8]
                
                    return REGMAT
                
                SFFTModule_dict['fill_regmat'] = Fill_REGMAT
            
            
            if SCALING_MODE == 'SEPARATE-VARYING':
                
                c0 = w0*L1+w1
                SCALE2 = SCALE**2
                _strdec = 'f8[:,:](i4[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def Fill_REGMAT(iREGMAT, SSTMAT, CSSTMAT, DSSTMAT, REGMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fijab):
                            k = int(ROW // Fab)
                            c = int(ROW % Fab)
                            k8 = int(COL // Fab)
                            c8 = int(COL % Fab)
                            
                            if (c != c0 and c8 != c0):
                                REGMAT[ROW, COL] = SCALE2 * SSTMAT[k, k8] * iREGMAT[c, c8]
                                
                            if (c != c0 and c8 == c0):
                                REGMAT[ROW, COL] = SCALE2 * CSSTMAT[k, k8] * iREGMAT[c, c8]
                                
                            if (c == c0 and c8 != c0):
                                REGMAT[ROW, COL] = SCALE2 * CSSTMAT[k8, k] * iREGMAT[c, c8]
                                
                            if (c == c0 and c8 == c0):
                                REGMAT[ROW, COL] = SCALE2 * DSSTMAT[k, k8] * iREGMAT[c, c8]
                                
                    return REGMAT
                
                SFFTModule_dict['fill_regmat'] = Fill_REGMAT

        # <*****> Tweak Linear-System & Restore Solution <*****> #
        if SCALING_MODE == 'SEPARATE-CONSTANT':

            if KerSpType == 'Polynomial':
                
                _strdec = 'Tuple((f8[:,:], f8[:]))' + '(f8[:,:], f8[:], i4[:], f8[:,:], f8[:])'
                @nb.njit(_strdec, parallel=True)
                def TweakLSPoly(LHMAT, RHb, PresIDX, LHMAT_tweaked, RHb_tweaked):
                    
                    for ROW in nb.prange(NEQt):
                        
                        RHb_tweaked[ROW] = RHb[PresIDX[ROW]]
                        
                        for COL in nb.prange(NEQt):
                            LHMAT_tweaked[ROW, COL] = LHMAT[PresIDX[ROW], PresIDX[COL]]

                    return LHMAT_tweaked, RHb_tweaked
                
                SFFTModule_dict['TweakLS'] = TweakLSPoly

            if KerSpType == 'B-Spline':
                
                _strdec = 'Tuple((f8[:,:], f8[:]))' + '(f8[:,:], f8[:], i4[:], i4[:], f8[:,:], f8[:])'
                @nb.njit(_strdec, parallel=True)
                def TweakLSBSpline(LHMAT, RHb, PresIDX, ij00, LHMAT_tweaked, RHb_tweaked):
                    keyIdx = ij00[0]
                    
                    for ROW in nb.prange(NEQt):
                        for COL in nb.prange(NEQt):

                            if (ROW == keyIdx) and (COL != keyIdx) and (COL < NEQt):
                                cum1 = 0.
                                for ij in nb.prange(Fij):
                                    ridx = ij00[ij]
                                    PresCOL = PresIDX[COL]
                                    cum1 += LHMAT[ridx, PresCOL]

                                LHMAT_tweaked[ROW, COL] = cum1
                
                
                            if (ROW != keyIdx) and (COL == keyIdx):
                                cum2 = 0.
                                for ij in nb.prange(Fij):
                                    PresROW = PresIDX[ROW]
                                    cidx = ij00[ij]
                                    cum2 += LHMAT[PresROW, cidx]
                                    
                                LHMAT_tweaked[ROW, COL] = cum2
                                
                                
                            if (ROW == keyIdx) and (COL == keyIdx):
                                cum3 = 0.
                                for ij in nb.prange(Fij):
                                    for i8j8 in nb.prange(Fij):
                                        ridx = ij00[ij]
                                        cidx = ij00[i8j8]
                                        cum3 += LHMAT[ridx, cidx]
                                        
                                LHMAT_tweaked[ROW, COL] = cum3
                                
                                
                            
                            if (ROW != keyIdx) and (COL != keyIdx):
                                PresROW = PresIDX[ROW]
                                PresCOL = PresIDX[COL]
                                LHMAT_tweaked[ROW, COL] = LHMAT[PresROW, PresCOL]
                                
                    return LHMAT_tweaked, RHb_tweaked    
    
                SFFTModule_dict['TweakLS'] = TweakLSBSpline
            
            
            
            
            _strdec = 'f8[:](f8[:], i4[:], f8[:])'
            @nb.njit(_strdec, parallel=True)
            def RestoreSolution(Solution_tweaked, PresIDX, Solution):
                for ROW in nb.prange(NEQt):
                    PresROW = PresIDX[ROW]
                    Solution[PresROW] = Solution_tweaked[ROW]
                return Solution
            
            SFFTModule_dict['Restore_Solution'] = RestoreSolution




        if (SCALING_MODE == 'SEPARATE-VARYING') and (NEQt < NEQ):
            
            _strdec = 'Tuple((f8[:,:], f8[:]))' + '(f8[:,:], f8[:], i4[:], f8[:,:], f8[:])'
            @nb.njit(_strdec, parallel=True)
            def TweakLS(LHMAT, RHb, PresIDX, LHMAT_tweaked, RHb_tweaked):

                for ROW in nb.prange(NEQt):
                    PresROW = PresIDX[ROW]
                    RHb_tweaked[ROW] = RHb[PresROW]

                    for COL in nb.prange(NEQt):
                        PresCOL = PresIDX[COL]
                        LHMAT_tweaked[ROW, COL] = LHMAT[PresROW, PresCOL]

                return LHMAT_tweaked, RHb_tweaked
            
            SFFTModule_dict['TweakLS'] = TweakLS
            

            _strdec = 'f8[:](f8[:], i4[:], f8[:])'            
            @nb.njit(_strdec, parallel=True)
            def RestoreSolution(Solution_tweaked, PresIDX, Solution):
                
                for ROW in nb.prange(NEQt):
                    PresROW = PresIDX[ROW]
                    Solution[PresROW] = Solution_tweaked[ROW]

                return Solution

            SFFTModule_dict['Restore_Solution'] = RestoreSolution
        
        # ************************************ Construct Difference ************************************ #

        # <*****> Construct difference in Fourier space <*****> #
        if SCALING_MODE in ['ENTANGLED', 'SEPARATE-CONSTANT']:

            # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
            _strdec = 'c16[:,:](i4[:,:], i4[:,:], c16[:], c16[:,:,:], c16[:,:,:], c16[:,:,:], c16[:], c16[:,:,:], c16[:,:], c16[:,:])'
            @nb.njit(_strdec, parallel=True)
            def Construct_FDIFF(SREF_ijab, REF_ab, a_ijab, SPixA_FIij, Kab_Wla, Kab_Wmb, b_pq, SPixA_FTpq, PixA_FJ, PixA_FDIFF):
            
                ZERO_C = 0.0 + 0.0j
                ONE_C = 1.0 + 0.0j
                SCALE_C = SCALE + 0.0j
                
                for ROW in nb.prange(N0):
                    for COL in nb.prange(N1):
                        
                        PVAL = ZERO_C
                        PVAL_FKab = ZERO_C
                        
                        for ab in range(Fab):
                            a, b = REF_ab[ab]
                            
                            if (a == 0) and (b == 0):
                                PVAL_FKab = SCALE_C
                                
                            if (a != 0) or (b != 0):
                                PVAL_FKab = SCALE_C * ((Kab_Wla[w0 + a, ROW, COL] * Kab_Wmb[w1 + b, ROW, COL]) - ONE_C)
            
                            
                            for ij in range(Fij):
                                ijab = ij * Fab + ab
                                PVAL += (a_ijab[ijab] * SPixA_FIij[ij, ROW, COL]) * PVAL_FKab
                                
                        for pq in range(Fpq):
                            PVAL += b_pq[pq] * SPixA_FTpq[pq, ROW, COL]
                            
                        PixA_FDIFF[ROW, COL] = PixA_FJ[ROW, COL] - PVAL
                        
                return PixA_FDIFF
            
            SFFTModule_dict['Construct_FDIFF'] = Construct_FDIFF

        if SCALING_MODE == 'SEPARATE-VARYING':

            # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
            _strdec = 'c16[:,:](i4[:,:], i4[:,:], c16[:], c16[:,:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:], c16[:], c16[:,:,:], c16[:,:], c16[:,:])'
            @nb.njit(_strdec, parallel=True)
            def Construct_FDIFF(SREF_ijab, REF_ab, a_ijab, SPixA_FIij, ScaSPixA_FIij, Kab_Wla, Kab_Wmb, b_pq, SPixA_FTpq, PixA_FJ, PixA_FDIFF):
                
                ZERO_C = 0.0 + 0.0j
                ONE_C = 1.0 + 0.0j
                SCALE_C = SCALE + 0.0j
                
                for ROW in nb.prange(N0):
                    for COL in nb.prange(N1):
                        
                        PVAL = ZERO_C
                        PVAL_FKab = ZERO_C
                        
                        for ab in range(Fab):
                            a, b = REF_ab[ab]
                            
                            if (a == 0 and b == 0):
                                PVAL_FKab = SCALE_C
                                
                                for ij in range(Fij):
                                    ijab = ij * Fab + ab
                                    PVAL += (a_ijab[ijab] * ScaSPixA_FIij[ij, ROW, COL]) * PVAL_FKab
                                    
                                    
                            if (a != 0 or b != 0):
                                PVAL_FKab = SCALE_C * ((Kab_Wla[w0 + a, ROW, COL] * Kab_Wmb[w1 + b, ROW, COL]) - ONE_C)
            
                                for ij in range(Fij):
                                    ijab = ij * Fab + ab
                                    PVAL += (a_ijab[ijab] * SPixA_FIij[ij, ROW, COL]) * PVAL_FKab
                                    
                        
                        for pq in range(Fpq):
                            PVAL += b_pq[pq] * SPixA_FTpq[pq, ROW, COL]
                            
                        PixA_FDIFF[ROW, COL] = PixA_FJ[ROW, COL] - PVAL
                        
                return PixA_FDIFF    
                            
            SFFTModule_dict['Construct_FDIFF'] = Construct_FDIFF
            
            
            
        # ************************************ Construct Coadd ************************************ #

        # <*****> Construct difference in Fourier space <*****> #
        if SCALING_MODE in ['ENTANGLED', 'SEPARATE-CONSTANT']:

            # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
            _strdec = 'c16[:,:](i4[:,:], i4[:,:], c16[:], c16[:,:,:], c16[:,:,:], c16[:,:,:], c16[:], c16[:,:,:], c16[:,:], f8, f8, c16[:,:])'
            @nb.njit(_strdec, parallel=True)
            def Construct_FCOADD(SREF_ijab, REF_ab, a_ijab, SPixA_FIij, Kab_Wla, Kab_Wmb, b_pq, SPixA_FTpq, PixA_FJ, wI, wJ, PixA_FCOADD):
            
                ZERO_C = 0.0 + 0.0j
                ONE_C = 1.0 + 0.0j
                SCALE_C = SCALE + 0.0j
                
                wI_C = wI + 0.0j
                wJ_C = wJ + 0.0j
                
                for ROW in nb.prange(N0):
                    for COL in nb.prange(N1):
                        
                        PVAL = ZERO_C
                        PVAL_FKab = ZERO_C
                        
                        for ab in range(Fab):
                            a, b = REF_ab[ab]
                            
                            if (a == 0) and (b == 0):
                                PVAL_FKab = SCALE_C
                                
                            if (a != 0) or (b != 0):
                                PVAL_FKab = SCALE_C * ((Kab_Wla[w0 + a, ROW, COL] * Kab_Wmb[w1 + b, ROW, COL]) - ONE_C)
            
                            
                            for ij in range(Fij):
                                ijab = ij * Fab + ab
                                PVAL += (a_ijab[ijab] * SPixA_FIij[ij, ROW, COL]) * PVAL_FKab
                                
                        for pq in range(Fpq):
                            PVAL += b_pq[pq] * SPixA_FTpq[pq, ROW, COL]
                            
                        PixA_FCOADD[ROW, COL] = wI_C*PixA_FJ[ROW, COL] + wJ_C*PVAL
                        
                return PixA_FCOADD
            
            SFFTModule_dict['Construct_FCOADD'] = Construct_FCOADD

        if SCALING_MODE == 'SEPARATE-VARYING':

            # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
            _strdec = 'c16[:,:](i4[:,:], i4[:,:], c16[:], c16[:,:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:], c16[:], c16[:,:,:], c16[:,:], f8, f8, c16[:,:])'
            @nb.njit(_strdec, parallel=True)
            def Construct_FCOADD(SREF_ijab, REF_ab, a_ijab, SPixA_FIij, ScaSPixA_FIij, Kab_Wla, Kab_Wmb, b_pq, SPixA_FTpq, PixA_FJ, wI, wJ, PixA_FCOADD):
                
                ZERO_C = 0.0 + 0.0j
                ONE_C = 1.0 + 0.0j
                SCALE_C = SCALE + 0.0j
                
                wI_C = wI + 0.0j
                wJ_C = wJ + 0.0j
                
                for ROW in nb.prange(N0):
                    for COL in nb.prange(N1):
                        
                        PVAL = ZERO_C
                        PVAL_FKab = ZERO_C
                        
                        for ab in range(Fab):
                            a, b = REF_ab[ab]
                            
                            if (a == 0 and b == 0):
                                PVAL_FKab = SCALE_C
                                
                                for ij in range(Fij):
                                    ijab = ij * Fab + ab
                                    PVAL += (a_ijab[ijab] * ScaSPixA_FIij[ij, ROW, COL]) * PVAL_FKab
                                    
                                    
                            if (a != 0 or b != 0):
                                PVAL_FKab = SCALE_C * ((Kab_Wla[w0 + a, ROW, COL] * Kab_Wmb[w1 + b, ROW, COL]) - ONE_C)
            
                                for ij in range(Fij):
                                    ijab = ij * Fab + ab
                                    PVAL += (a_ijab[ijab] * SPixA_FIij[ij, ROW, COL]) * PVAL_FKab
                                    
                        
                        for pq in range(Fpq):
                            PVAL += b_pq[pq] * SPixA_FTpq[pq, ROW, COL]
                            
                        PixA_FCOADD[ROW, COL] = wI_C*PixA_FJ[ROW, COL] + wJ_C*PVAL
                        
                return PixA_FCOADD    
                            
            SFFTModule_dict['Construct_FCOADD'] = Construct_FCOADD


        SFFTConfig = (SFFTParam_dict, SFFTModule_dict)
        if VERBOSE_LEVEL in [1, 2]:
            logger.info('--//--//--//--//-- EXIT SFFT COMPILATION --//--//--//--//-- ')

        return SFFTConfig
    
    
    
class SingleSFFTConfigure:
    @staticmethod
    def SSN(NX, NY, KerHW=8, KerSpType='Polynomial', KerSpDegree=2, KerIntKnotX=[], KerIntKnotY=[], \
        SEPARATE_SCALING=True, ScaSpType='Polynomial', ScaSpDegree=0, ScaIntKnotX=[], ScaIntKnotY=[], \
        BkgSpType='Polynomial', BkgSpDegree=2, BkgIntKnotX=[], BkgIntKnotY=[], \
        REGULARIZE_KERNEL=False, IGNORE_LAPLACIAN_KERCENT=True, XY_REGULARIZE=None, WEIGHT_REGULARIZE=None, \
        LAMBDA_REGULARIZE=1e-6, BACKEND_4SUBTRACT='Numpy', MAX_THREADS_PER_BLOCK=8, \
        MINIMIZE_GPU_MEMORY_USAGE=False, NUM_CPU_THREADS_4SUBTRACT=8, logger=None, VERBOSE_LEVEL=2):

        """
        # Compile Functions for SFFT
        #
        # Arguments:
        # -NX: Image size along X (pix)                                                        | e.g., 1024
        # -NY: Image size along Y (pix)                                                        | e.g., 1024
        # -KerHW: Kernel half width for compilation                                            | e.g., 8
        #
        # -KerSpType: Spatial varaition type of matching kernel                                | ['Polynomial', 'B-Spline']
        # -KerSpDegree: Polynomial/B-Spline degree of kernel spatial varaition                 | [0, 1, 2, 3]
        # -KerIntKnotX: Internal knots of kernel B-Spline spatial varaition along X            | e.g., [256., 512., 768.]
        # -KerIntKnotY: Internal knots of kernel B-Spline spatial varaition along Y            | e.g., [256., 512., 768.]
        #
        # -SEPARATE_SCALING: separate convolution scaling or entangled with matching kernel?   | [True, False]
        # -ScaSpType: Spatial varaition type of convolution scaling                            | ['Polynomial', 'B-Spline']
        # -ScaSpDegree: Polynomial/B-Spline degree of convolution scaling                      | [0, 1, 2, 3]
        # -ScaIntKnotX: Internal knots of scaling B-Spline spatial varaition along X           | e.g., [256., 512., 768.]
        # -ScaIntKnotY: Internal knots of scaling B-Spline spatial varaition along Y           | e.g., [256., 512., 768.]
        #
        # -BkgSpType: Spatial varaition type of differential background                        | ['Polynomial', 'B-Spline']
        # -BkgSpDegree: Polynomial/B-Spline degree of background spatial varaition             | [0, 1, 2, 3]
        # -BkgIntKnotX: Internal knots of background B-Spline spatial varaition along X        | e.g., [256., 512., 768.]
        # -BkgIntKnotY: Internal knots of background B-Spline spatial varaition along Y        | e.g., [256., 512., 768.]
        # 
        # -REGULARIZE_KERNEL: Regularize matching kernel by applying penalty on                | [True, False]
        #    kernel's second derivates using Laplacian matrix
        # -IGNORE_LAPLACIAN_KERCENT: zero out the rows of Laplacian matrix                     | [True, False]
        #    corresponding the kernel center pixels by zeros. 
        #    If True, the regularization will not impose any penalty 
        #    on a delta-function-like matching kernel
        # -XY_REGULARIZE: The coordinates at which the matching kernel regularized.            | e.g., np.array([[64., 64.], 
        #    Numpy array of (x, y) with shape (N_points, 2),                                   |                 [256., 256.]]) 
        #    where x in (0.5, NX+0.5) and y in (0.5, NY+0.5)
        # -WEIGHT_REGULARIZE: The weights of the coordinates sampled for regularization.       | e.g., np.array([1.0, 2.0, ...])
        #    Numpy array of weights with shape (N_points)
        #    -WEIGHT_REGULARIZE = None means uniform weights of 1.0
        # -LAMBDA_REGULARIZE: Tunning paramater lambda for regularization                      | e.g., 1e-6
        #    it controls the strength of penalty on kernel overfitting
        #
        # -BACKEND_4SUBTRACT: The backend with which you perform SFFT subtraction              | ['Cupy', 'Numpy']
        # -MAX_THREADS_PER_BLOCK: Maximum Threads per Block for CUDA configuration             | e.g., 8
        # -MINIMIZE_GPU_MEMORY_USAGE: Minimize the GPU Memory Usage?                           | [True, False]
        # -NUM_CPU_THREADS_4SUBTRACT: The number of CPU threads for Numpy-SFFT subtraction     | e.g., 8
        #
        # -VERBOSE_LEVEL: The level of verbosity, can be 0/1/2: QUIET/NORMAL/FULL              | [0, 1, 2]
        #
        """
        
        if BACKEND_4SUBTRACT == 'Numpy':

            SFFTConfig = SingleSFFTConfigure_Numpy.SSCN(NX=NX, NY=NY, KerHW=KerHW, KerSpType=KerSpType, \
                KerSpDegree=KerSpDegree, KerIntKnotX=KerIntKnotX, KerIntKnotY=KerIntKnotY, \
                SEPARATE_SCALING=SEPARATE_SCALING, ScaSpType=ScaSpType, ScaSpDegree=ScaSpDegree, \
                ScaIntKnotX=ScaIntKnotX, ScaIntKnotY=ScaIntKnotY, BkgSpType=BkgSpType, \
                BkgSpDegree=BkgSpDegree, BkgIntKnotX=BkgIntKnotX, BkgIntKnotY=BkgIntKnotY, \
                REGULARIZE_KERNEL=REGULARIZE_KERNEL, IGNORE_LAPLACIAN_KERCENT=IGNORE_LAPLACIAN_KERCENT, \
                XY_REGULARIZE=XY_REGULARIZE, WEIGHT_REGULARIZE=WEIGHT_REGULARIZE, LAMBDA_REGULARIZE=LAMBDA_REGULARIZE, \
                MAX_THREADS_PER_BLOCK=MAX_THREADS_PER_BLOCK, MINIMIZE_GPU_MEMORY_USAGE=MINIMIZE_GPU_MEMORY_USAGE, \
                VERBOSE_LEVEL=VERBOSE_LEVEL, logger=logger)
        
        if BACKEND_4SUBTRACT == 'Cupy':
            print('MeLOn ERROR: Use the other configure class for Cupy!')
            SFFTConfig = None
        
        return SFFTConfig
    
    


class ElementalSFFTSubtract_Numpy:
    @staticmethod
    def ESSN(PixA_I, PixA_J, SFFTConfig, wI=1., wJ=1., SFFTSolution=None, Subtract=False, Coadd=False, NUM_CPU_THREADS_4SUBTRACT=8, \
        logger=None, VERBOSE_LEVEL=2, fft_type='mkl'):
        
        import scipy.linalg as linalg
        def solver(A, b):
            lu_piv = linalg.lu_factor(A, overwrite_a=False, check_finite=True)
            x = linalg.lu_solve(lu_piv, b)
            return x
        
        # def solver(A, b):
        #     return np.linalg.solve(A, b)
        
        
        def Create_BSplineBasis(N, IntKnot, BSplineDegree):
            BSplineBasis = []
            PixCoord = (1.0+np.arange(N))/N
            Knot = np.concatenate(([0.5]*(BSplineDegree+1), IntKnot, [N+0.5]*(BSplineDegree+1)))/N
            Nc = len(IntKnot) + BSplineDegree + 1    # number of control points/coeffcients
            for idx in range(Nc):
                Coeff = (np.arange(Nc) == idx).astype(float)
                BaseFunc = BSpline(t=Knot, c=Coeff, k=BSplineDegree, extrapolate=False)
                BSplineBasis.append(BaseFunc(PixCoord))
            BSplineBasis = np.array(BSplineBasis)
            return BSplineBasis
        
        def Create_BSplineBasis_Req(N, IntKnot, BSplineDegree, ReqCoord):
            BSplineBasis_Req = []
            Knot = np.concatenate(([0.5]*(BSplineDegree+1), IntKnot, [N+0.5]*(BSplineDegree+1)))/N
            Nc = len(IntKnot) + BSplineDegree + 1    # number of control points/coeffcients
            for idx in range(Nc):
                Coeff = (np.arange(Nc) == idx).astype(float)
                BaseFunc = BSpline(t=Knot, c=Coeff, k=BSplineDegree, extrapolate=False)
                BSplineBasis_Req.append(BaseFunc(ReqCoord))
            BSplineBasis_Req = np.array(BSplineBasis_Req)
            return BSplineBasis_Req
        
        ta = time.time()
        # * Read SFFT parameters
        SFFTParam_dict, SFFTModule_dict = SFFTConfig

        KerHW = SFFTParam_dict['KerHW']
        KerSpType = SFFTParam_dict['KerSpType']
        KerSpDegree = SFFTParam_dict['KerSpDegree']
        KerIntKnotX = SFFTParam_dict['KerIntKnotX']
        KerIntKnotY = SFFTParam_dict['KerIntKnotY']

        SEPARATE_SCALING = SFFTParam_dict['SEPARATE_SCALING']
        if SEPARATE_SCALING:
            ScaSpType = SFFTParam_dict['ScaSpType']
            ScaSpDegree = SFFTParam_dict['ScaSpDegree']
            ScaIntKnotX = SFFTParam_dict['ScaIntKnotX']
            ScaIntKnotY = SFFTParam_dict['ScaIntKnotY']

        BkgSpType = SFFTParam_dict['BkgSpType']
        BkgSpDegree = SFFTParam_dict['BkgSpDegree']
        BkgIntKnotX = SFFTParam_dict['BkgIntKnotX']
        BkgIntKnotY = SFFTParam_dict['BkgIntKnotY']
        
        REGULARIZE_KERNEL = SFFTParam_dict['REGULARIZE_KERNEL']
        IGNORE_LAPLACIAN_KERCENT = SFFTParam_dict['IGNORE_LAPLACIAN_KERCENT']
        XY_REGULARIZE = SFFTParam_dict['XY_REGULARIZE']
        WEIGHT_REGULARIZE = SFFTParam_dict['WEIGHT_REGULARIZE']
        LAMBDA_REGULARIZE = SFFTParam_dict['LAMBDA_REGULARIZE']

        MAX_THREADS_PER_BLOCK = SFFTParam_dict['MAX_THREADS_PER_BLOCK']
        MINIMIZE_GPU_MEMORY_USAGE = SFFTParam_dict['MINIMIZE_GPU_MEMORY_USAGE']

        SCALING_MODE = None
        if not SEPARATE_SCALING:
            SCALING_MODE = 'ENTANGLED'
        elif ScaSpDegree == 0:
            SCALING_MODE = 'SEPARATE-CONSTANT'
        else: SCALING_MODE = 'SEPARATE-VARYING'
        assert SCALING_MODE is not None

        if VERBOSE_LEVEL in [1, 2]:
            logger.info('--//--//--//--//-- TRIGGER SFFT EXECUTION [Numpy] --//--//--//--//-- ')

            if KerSpType == 'Polynomial':
                logger.info('---//--- Polynomial Kernel | KerSpDegree %d | KerHW %d ---//---' %(KerSpDegree, KerHW))
                if not SEPARATE_SCALING:
                    logger.info('---//--- [ENTANGLED] Polynomial Scaling | KerSpDegree %d ---//---' %KerSpDegree)

            if KerSpType == 'B-Spline':
                logger.info('---//--- B-Spline Kernel | Internal Knots %d,%d | KerSpDegree %d | KerHW %d ---//---' \
                      %(len(KerIntKnotX), len(KerIntKnotY), KerSpDegree, KerHW))
                if not SEPARATE_SCALING: 
                    logger.info('---//--- [ENTANGLED] B-Spline Scaling | Internal Knots %d,%d | KerSpDegree %d ---//---' \
                          %(len(KerIntKnotX), len(KerIntKnotY), KerSpDegree))
            
            if SEPARATE_SCALING:
                if ScaSpType == 'Polynomial':
                    logger.info('---//--- [SEPARATE] Polynomial Scaling | ScaSpDegree %d ---//---' %ScaSpDegree)
                
                if ScaSpType == 'B-Spline':
                    logger.info('---//--- [SEPARATE] B-Spline Scaling | Internal Knots %d,%d | ScaSpDegree %d ---//---' \
                          %(len(ScaIntKnotX), len(ScaIntKnotY), ScaSpDegree))
            
            if BkgSpType == 'Polynomial':
                logger.info('---//--- Polynomial Background | BkgSpDegree %d ---//---' %BkgSpDegree)
            
            if BkgSpType == 'B-Spline':
                logger.info('---//--- B-Spline Background | Internal Knots %d,%d | BkgSpDegree %d ---//---' \
                    %(len(BkgIntKnotX), len(BkgIntKnotY), BkgSpDegree))


        if VERBOSE_LEVEL in [1, 2]:
            logger.info('MeLOn CheckPoint: Start SFFT-EXECUTION Preliminary Steps')


        N0 = SFFTParam_dict['N0']               # a.k.a, NX
        N1 = SFFTParam_dict['N1']               # a.k.a, NY
        w0 = SFFTParam_dict['w0']               # a.k.a, KerHW
        w1 = SFFTParam_dict['w1']               # a.k.a, KerHW
        DK = SFFTParam_dict['DK']               # a.k.a, KerSpDegree
        DB = SFFTParam_dict['DB']               # a.k.a, BkgSpDegree
        if SEPARATE_SCALING: 
            DS = SFFTParam_dict['DS']           # a.k.a, PolyScaSpDegree

        SCALE = SFFTParam_dict['SCALE']
        SCALE_L = SFFTParam_dict['SCALE_L']

        L0 = SFFTParam_dict['L0']
        L1 = SFFTParam_dict['L1']
        Fab = SFFTParam_dict['Fab']
        Fi = SFFTParam_dict['Fi']
        Fj = SFFTParam_dict['Fj']
        Fij = SFFTParam_dict['Fij']
        Fp = SFFTParam_dict['Fp']
        Fq = SFFTParam_dict['Fq']
        Fpq = SFFTParam_dict['Fpq']

        if SCALING_MODE == 'SEPARATE-VARYING':
            ScaFi = SFFTParam_dict['ScaFi']
            ScaFj = SFFTParam_dict['ScaFj']
            ScaFij = SFFTParam_dict['ScaFij']
        Fijab = SFFTParam_dict['Fijab']
        
        FOMG = SFFTParam_dict['FOMG']
        FGAM = SFFTParam_dict['FGAM']
        FTHE = SFFTParam_dict['FTHE']
        FPSI = SFFTParam_dict['FPSI']
        FPHI = SFFTParam_dict['FPHI']
        FDEL = SFFTParam_dict['FDEL']
        
        NEQ = SFFTParam_dict['NEQ']
        NEQt = SFFTParam_dict['NEQt']

        # check input image size 
        assert PixA_I.shape == (N0, N1) and PixA_J.shape == (N0, N1)

        # * Define First-order MultiIndex Reference
        if KerSpType == 'Polynomial':
            REF_ij = np.array([(i, j) for i in range(DK+1) for j in range(DK+1-i)]).astype(np.int32)
        if KerSpType == 'B-Spline':
            REF_ij = np.array([(i, j) for i in range(Fi) for j in range(Fj)]).astype(np.int32)

        if BkgSpType == 'Polynomial':
            REF_pq = np.array([(p, q) for p in range(DB+1) for q in range(DB+1-p)]).astype(np.int32)
        if BkgSpType == 'B-Spline':
            REF_pq = np.array([(p, q) for p in range(Fp) for q in range(Fq)]).astype(np.int32)
        REF_ab = np.array([(a_pos-w0, b_pos-w1) for a_pos in range(L0) for b_pos in range(L1)]).astype(np.int32)



        if SCALING_MODE == 'SEPARATE-VARYING':
            
            if ScaSpType == 'Polynomial':
                ScaREF_ij = np.array(
                    [(i, j) for i in range(DS+1) for j in range(DS+1-i)] \
                    + [(-1, -1)] * (Fij - ScaFij)
                ).astype(np.int32)

            if ScaSpType == 'B-Spline':
                ScaREF_ij = np.array(
                    [(i, j) for i in range(ScaFi) for j in range(ScaFj)] \
                    + [(-1, -1)] * (Fij - ScaFij)
                ).astype(np.int32)

        # * Define Second-order MultiIndex Reference
        SREF_iji0j0 = np.array([(ij, i0j0) for ij in range(Fij) for i0j0 in range(Fij)]).astype(np.int32)
        SREF_pqp0q0 = np.array([(pq, p0q0) for pq in range(Fpq) for p0q0 in range(Fpq)]).astype(np.int32)
        SREF_ijpq = np.array([(ij, pq) for ij in range(Fij) for pq in range(Fpq)]).astype(np.int32)
        SREF_pqij = np.array([(pq, ij) for pq in range(Fpq) for ij in range(Fij)]).astype(np.int32)
        SREF_ijab = np.array([(ij, ab) for ij in range(Fij) for ab in range(Fab)]).astype(np.int32)

        # * Indices related to Constant Scaling Case
        ij00 = np.arange(w0 * L1 + w1, Fijab, Fab).astype(np.int32)

        t0 = time.time()
        # * Read input images as C-order arrays
        if not PixA_I.flags['C_CONTIGUOUS']:
            PixA_I = np.ascontiguousarray(PixA_I, np.float64)
        else: PixA_I = np.array(PixA_I.astype(np.float64))
        
        if not PixA_J.flags['C_CONTIGUOUS']:
            PixA_J = np.ascontiguousarray(PixA_J, np.float64)
        else: PixA_J = np.array(PixA_J.astype(np.float64))
        dt0 = time.time() - t0
        
        if VERBOSE_LEVEL in [2]:
            logger.info('/////   a   ///// Read Input Images  (%.4fs)' %dt0)
        
        # * Symbol Convention Notes
        #   X (x) / Y (y) ----- pixel row / column index
        #   CX (cx) / CY (cy) ----- ScaledFortranCoord of pixel (x, y) center   
        #   e.g. pixel (x, y) = (3, 5) corresponds (cx, cy) = (4.0/N0, 6.0/N1)
        #   NOTE cx / cy is literally \mathtt{x} / \mathtt{y} in SFFT paper.
        #   NOTE Without special definition, MeLOn convention refers to X (x) / Y (y) as FortranCoord.

        # * Get Spatial Coordinates
        t1 = time.time()
        PixA_X = np.zeros((N0, N1), dtype=np.int32)      # row index, [0, N0)
        PixA_Y = np.zeros((N0, N1), dtype=np.int32)      # column index, [0, N1)
        PixA_CX = np.zeros((N0, N1), dtype=np.float64)   # coordinate.x
        PixA_CY = np.zeros((N0, N1), dtype=np.float64)   # coordinate.y 

        _func = SFFTModule_dict['SpatialCoord']
        PixA_X, PixA_Y, PixA_CX, PixA_CY = _func(PixA_X=PixA_X, PixA_Y=PixA_Y, PixA_CX=PixA_CX, PixA_CY=PixA_CY)


        # <*****> produce Iij <*****> #
        SPixA_Iij = np.zeros((Fij, N0, N1), dtype=np.float64)
        if KerSpType == 'Polynomial':
            _func = SFFTModule_dict['KerSpatial']
            SPixA_Iij = _func(REF_ij=REF_ij, PixA_CX=PixA_CX, PixA_CY=PixA_CY, PixA_I=PixA_I, SPixA_Iij=SPixA_Iij)
        
        if KerSpType == 'B-Spline':
            KerSplBasisX = Create_BSplineBasis(N=N0, IntKnot=KerIntKnotX, BSplineDegree=DK).astype(np.float64)
            KerSplBasisY = Create_BSplineBasis(N=N1, IntKnot=KerIntKnotY, BSplineDegree=DK).astype(np.float64)

            _func = SFFTModule_dict['KerSpatial']
            SPixA_Iij = _func(REF_ij=REF_ij, KerSplBasisX=KerSplBasisX, KerSplBasisY=KerSplBasisY, PixA_I=PixA_I, SPixA_Iij=SPixA_Iij)



        if fft_type == 'numba':
            #1 it/s (same with fastmath=True)
            
            from rocket_fft import numpy_like
            numpy_like()
            
            fft2d_func = SFFTModule_dict['fft2d']
            fft3d_func = SFFTModule_dict['fft3d']
            ifft2d_func = SFFTModule_dict['ifft2d']
        
        elif fft_type == 'pyfftw':
            #3 it/s
            #~1.6 it/s (with dask)

            import pyfftw
            pyfftw.config.NUM_THREADS = NUM_CPU_THREADS_4SUBTRACT
            pyfftw.interfaces.cache.enable()
            pyfftw.interfaces.cache.set_keepalive_time(10)

            fft2d_func = pyfftw.interfaces.dask_fft.fft2
            fft3d_func = pyfftw.interfaces.dask_fft.fft2
            ifft2d_func = pyfftw.interfaces.dask_fft.ifft2
            
        elif fft_type == 'mkl':
            #~3.5 it/s
            
            from mkl_fft._numpy_fft import fft2, ifft2
            
            fft2d_func = fft2
            fft3d_func = fft2
            ifft2d_func = ifft2
            



        if SCALING_MODE == 'SEPARATE-VARYING':

            if ScaSpType == 'Polynomial':
                ScaSPixA_Iij = np.zeros((Fij, N0, N1), dtype=np.float64)
                _func = SFFTModule_dict['ScaSpatial']
                ScaSPixA_Iij = _func(ScaREF_ij=ScaREF_ij, PixA_CX=PixA_CX, PixA_CY=PixA_CY, PixA_I=PixA_I, ScaSPixA_Iij=ScaSPixA_Iij)

            if ScaSpType == 'B-Spline':
                ScaSplBasisX = Create_BSplineBasis(N=N0, IntKnot=ScaIntKnotX, BSplineDegree=DS).astype(np.float64)
                ScaSplBasisY = Create_BSplineBasis(N=N1, IntKnot=ScaIntKnotY, BSplineDegree=DS).astype(np.float64)

                ScaSPixA_Iij = np.zeros((Fij, N0, N1), dtype=np.float64)
                _func = SFFTModule_dict['ScaSpatial']
                ScaSPixA_Iij = _func(ScaREF_ij=ScaREF_ij, ScaSplBasisX=ScaSplBasisX, ScaSplBasisY=ScaSplBasisY, PixA_I=PixA_I, ScaSPixA_Iij=ScaSPixA_Iij)

        del PixA_I

        # <*****> produce Tpq <*****> #
        SPixA_Tpq = np.zeros((Fpq, N0, N1), dtype=np.float64)
        
        if BkgSpType == 'Polynomial':
            _func = SFFTModule_dict['BkgSpatial']
            SPixA_Tpq = _func(REF_pq=REF_pq, PixA_CX=PixA_CX, PixA_CY=PixA_CY, SPixA_Tpq=SPixA_Tpq)
        
        if BkgSpType == 'B-Spline':
            BkgSplBasisX = Create_BSplineBasis(N=N0, IntKnot=BkgIntKnotX, BSplineDegree=DB).astype(np.float64)
            BkgSplBasisY = Create_BSplineBasis(N=N1, IntKnot=BkgIntKnotY, BSplineDegree=DB).astype(np.float64)

            _func = SFFTModule_dict['BkgSpatial']
            SPixA_Tpq = _func(REF_pq=REF_pq, BkgSplBasisX=BkgSplBasisX, BkgSplBasisY=BkgSplBasisY, SPixA_Tpq=SPixA_Tpq)

        
        dt1 = time.time() - t1
        t2 = time.time()
        
        if VERBOSE_LEVEL in [2]:
            logger.info('/////   b   ///// Spatial Polynomial (%.4fs)' %dt1)


        # * Make DFT of J, Iij, Tpq and their conjugates
        PixA_FJ = np.empty((N0, N1), dtype=np.complex128)
        PixA_FJ[:, :] = PixA_J.astype(np.complex128)
        # PixA_FJ[:, :] = pyfftw.interfaces.numpy_fft.fft2(PixA_FJ)
        PixA_FJ[:, :] = fft2d_func(PixA_FJ)
        PixA_FJ[:, :] *= SCALE

        SPixA_FIij = np.empty((Fij, N0, N1), dtype=np.complex128)
        SPixA_FIij[:, :, :] = SPixA_Iij.astype(np.complex128)
        for k in range(Fij): 
            # SPixA_FIij[k: k+1] = pyfftw.interfaces.numpy_fft.fft2(SPixA_FIij[k: k+1])
            SPixA_FIij[k: k+1] = fft3d_func(SPixA_FIij[k: k+1])
        SPixA_FIij[:, :] *= SCALE

        SPixA_FTpq = np.empty((Fpq, N0, N1), dtype=np.complex128)
        SPixA_FTpq[:, :, :] = SPixA_Tpq.astype(np.complex128)
        for k in range(Fpq): 
            # SPixA_FTpq[k: k+1] = pyfftw.interfaces.numpy_fft.fft2(SPixA_FTpq[k: k+1])
            SPixA_FTpq[k: k+1] = fft3d_func(SPixA_FTpq[k: k+1])
        SPixA_FTpq[:, :] *= SCALE

        del PixA_J
        del SPixA_Iij
        del SPixA_Tpq

        PixA_CFJ = np.conj(PixA_FJ)
        SPixA_CFIij = np.conj(SPixA_FIij)
        SPixA_CFTpq = np.conj(SPixA_FTpq)



        if SCALING_MODE == 'SEPARATE-VARYING':
            # TODO: this variable can be too GPU memory consuming
            ScaSPixA_FIij = np.empty((Fij, N0, N1), dtype=np.complex128)
            ScaSPixA_FIij[:, :, :] = ScaSPixA_Iij.astype(np.complex128)
            for k in range(ScaFij):
                # ScaSPixA_FIij[k: k+1] = pyfftw.interfaces.numpy_fft.fft2(ScaSPixA_FIij[k: k+1])
                ScaSPixA_FIij[k: k+1] = fft3d_func(ScaSPixA_FIij[k: k+1])
            ScaSPixA_FIij[:, :] *= SCALE

            del ScaSPixA_Iij
            ScaSPixA_CFIij = np.conj(ScaSPixA_FIij)
        
        dt2 = time.time() - t2
        dta = time.time() - ta
        
        if VERBOSE_LEVEL in [2]:
            logger.info('/////   c   ///// DFT-%d             (%.4fs)' %(1 + Fij + Fpq, dt2))

        if VERBOSE_LEVEL in [1, 2]:
            logger.info('MeLOn CheckPoint: SFFT-EXECUTION Preliminary Steps takes [%.4fs]' %dta)

        # * Consider The Major Sources of the Linear System 
        #    OMEGA_i8j8ij   &    DELTA_i8j8pq    ||   THETA_i8j8
        #    PSI_p8q8ij     &    PHI_p8q8pq      ||   DELTA_p8q8
        #
        # - Remarks
        #    a. They have consistent form: Greek(rho, eps) = PreGreek(Mod_N0(rho), Mod_N1(eps))
        #    b. PreGreek = s * Re[DFT(HpGreek)], where HpGreek = GLH * GRH and s is some real-scale
        #    c. Considering the subscripted variables, HpGreek/PreGreek is Complex/Real 3D with shape (F_Greek, N0, N1)
        
        if SFFTSolution is not None:
            Solution = SFFTSolution 
            Solution = np.array(Solution.astype(np.float64))
            a_ijab = Solution[: Fijab]
            b_pq = Solution[Fijab: ]



        if SFFTSolution is None:
            if VERBOSE_LEVEL in [1, 2]:
                logger.info('MeLOn CheckPoint: Start SFFT-EXECUTION Establish & Solve Linear System')
            
            tb = time.time()

            LHMAT = np.empty((NEQ, NEQ), dtype=np.float64)
            RHb = np.empty(NEQ, dtype=np.float64)

            t3 = time.time()
            if SCALING_MODE in ['ENTANGLED', 'SEPARATE-CONSTANT']:

                if not MINIMIZE_GPU_MEMORY_USAGE:

                    # <*****> Establish Linear System through OMEGA <*****> #
                    
                    # a. Hadamard Product for OMEGA [HpOMG]
                    _func = SFFTModule_dict['HadProd_OMG']
                    HpOMG = np.empty((FOMG, N0, N1), dtype=np.complex128)
                    HpOMG = _func(SREF_iji0j0=SREF_iji0j0, SPixA_FIij=SPixA_FIij, SPixA_CFIij=SPixA_CFIij, HpOMG=HpOMG)

                    # b. PreOMG = SCALE * Re[DFT(HpOMG)]
                    for k in range(FOMG):
                        HpOMG[k: k+1] = pyfftw.interfaces.numpy_fft.fft2(HpOMG[k: k+1])
                    HpOMG *= SCALE

                    PreOMG = np.empty((FOMG, N0, N1), dtype=np.float64)
                    PreOMG[:, :, :] = HpOMG.real
                    PreOMG[:, :, :] *= SCALE
                    del HpOMG

                    # c. Fill Linear System with PreOMG
                    _func = SFFTModule_dict['FillLS_OMG']
                    LHMAT = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, PreOMG=PreOMG, LHMAT=LHMAT)
                    del PreOMG

                    dt3 = time.time() - t3
                    t4 = time.time()

                    # <*****> Establish Linear System through GAMMA <*****> #

                    # a. Hadamard Product for GAMMA [HpGAM]
                    _func = SFFTModule_dict['HadProd_GAM']
                    HpGAM = np.empty((FGAM, N0, N1), dtype=np.complex128)
                    HpGAM = _func(SREF_ijpq=SREF_ijpq, SPixA_FIij=SPixA_FIij, SPixA_CFTpq=SPixA_CFTpq, HpGAM=HpGAM)

                    # b. PreGAM = 1 * Re[DFT(HpGAM)]
                    for k in range(FGAM):
                        HpGAM[k: k+1] = fft2d_func(HpGAM[k: k+1])
                    HpGAM *= SCALE

                    PreGAM = np.empty((FGAM, N0, N1), dtype=np.float64)
                    PreGAM[:, :, :] = HpGAM.real
                    del HpGAM

                    # c. Fill Linear System with PreGAM
                    _func = SFFTModule_dict['FillLS_GAM']
                    LHMAT = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, PreGAM=PreGAM, LHMAT=LHMAT)
                    del PreGAM
                    
                    dt4 = time.time() - t4
                    t5 = time.time()
                    
                    # <*****> Establish Linear System through PSI <*****> #

                    # a. Hadamard Product for PSI [HpPSI]
                    _func = SFFTModule_dict['HadProd_PSI']
                    HpPSI = np.empty((FPSI, N0, N1), dtype=np.complex128)
                    HpPSI = _func(SREF_pqij=SREF_pqij, SPixA_CFIij=SPixA_CFIij, SPixA_FTpq=SPixA_FTpq, HpPSI=HpPSI)
                    
                    # b. PrePSI = 1 * Re[DFT(HpPSI)]
                    for k in range(FPSI):
                        HpPSI[k: k+1] = fft2d_func(HpPSI[k: k+1])
                    HpPSI *= SCALE

                    PrePSI = np.empty((FPSI, N0, N1), dtype=np.float64)
                    PrePSI[:, :, :] = HpPSI.real
                    del HpPSI

                    # c. Fill Linear System with PrePSI
                    _func = SFFTModule_dict['FillLS_PSI']
                    LHMAT = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, PrePSI=PrePSI, LHMAT=LHMAT)
                    del PrePSI

                    dt5 = time.time() - t5
                    t6 = time.time()

                    # <*****> Establish Linear System through PHI <*****> #

                    # a. Hadamard Product for PHI  [HpPHI]
                    _func = SFFTModule_dict['HadProd_PHI']
                    HpPHI = np.empty((FPHI, N0, N1), dtype=np.complex128)
                    HpPHI = _func(SREF_pqp0q0=SREF_pqp0q0, SPixA_FTpq=SPixA_FTpq, SPixA_CFTpq=SPixA_CFTpq, HpPHI=HpPHI)

                    # b. PrePHI = SCALE_L * Re[DFT(HpPHI)]
                    for k in range(FPHI):
                        HpPHI[k: k+1] = fft2d_func(HpPHI[k: k+1])
                    HpPHI *= SCALE

                    PrePHI = np.empty((FPHI, N0, N1), dtype=np.float64)
                    PrePHI[:, :, :] = HpPHI.real
                    PrePHI[:, :, :] *= SCALE_L
                    del HpPHI

                    # c. Fill Linear System with PrePHI
                    _func = SFFTModule_dict['FillLS_PHI']
                    LHMAT = _func(PrePHI=PrePHI, LHMAT=LHMAT)
                    del PrePHI




                if MINIMIZE_GPU_MEMORY_USAGE:

                    # <*****> Establish Linear System through OMEGA <*****> #

                    for cIdx in range(FOMG):

                        # a. Hadamard Product for OMEGA [HpOMG]
                        _func = SFFTModule_dict['HadProd_OMG']
                        cHpOMG = np.empty((N0, N1), dtype=np.complex128)
                        cHpOMG = _func(SREF_iji0j0=SREF_iji0j0, SPixA_FIij=SPixA_FIij, SPixA_CFIij=SPixA_CFIij, cIdx=cIdx, cHpOMG=cHpOMG)
                        
                        # b. PreOMG = SCALE * Re[DFT(HpOMG)]
                        cHpOMG = fft2d_func(cHpOMG)
                        cHpOMG *= SCALE

                        cPreOMG = np.empty((N0, N1), dtype=np.float64)
                        cPreOMG[:, :] = cHpOMG.real
                        cPreOMG[:, :] *= SCALE
                        del cHpOMG

                        # c. Fill Linear System with PreOMG
                        _func = SFFTModule_dict['FillLS_OMG']
                        LHMAT = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, cIdx=cIdx, cPreOMG=cPreOMG, LHMAT=LHMAT)
                        del cPreOMG

                    dt3 = time.time() - t3
                    t4 = time.time()

                    # <*****> Establish Linear System through GAMMA <*****> #

                    for cIdx in range(FGAM):

                        # a. Hadamard Product for GAMMA [HpGAM]
                        _func = SFFTModule_dict['HadProd_GAM']
                        cHpGAM = np.empty((N0, N1), dtype=np.complex128)
                        cHpGAM = _func(SREF_ijpq=SREF_ijpq, SPixA_FIij=SPixA_FIij, SPixA_CFTpq=SPixA_CFTpq, cIdx=cIdx, cHpGAM=cHpGAM)

                        # b. PreGAM = 1 * Re[DFT(HpGAM)]
                        cHpGAM = fft2d_func(cHpGAM)
                        cHpGAM *= SCALE

                        cPreGAM = np.empty((N0, N1), dtype=np.float64)
                        cPreGAM[:, :] = cHpGAM.real
                        del cHpGAM

                        # c. Fill Linear System with PreGAM
                        _func = SFFTModule_dict['FillLS_GAM']
                        LHMAT = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, cIdx=cIdx, cPreGAM=cPreGAM, LHMAT=LHMAT)
                        del cPreGAM

                    dt4 = time.time() - t4
                    t5 = time.time()

                    # <*****> Establish Linear System through PSI <*****> #

                    for cIdx in range(FPSI):

                        # a. Hadamard Product for PSI [HpPSI]
                        _func = SFFTModule_dict['HadProd_PSI']
                        cHpPSI = np.empty((N0, N1), dtype=np.complex128)
                        cHpPSI = _func(SREF_pqij=SREF_pqij, SPixA_CFIij=SPixA_CFIij, SPixA_FTpq=SPixA_FTpq, cIdx=cIdx, cHpPSI=cHpPSI)

                        # b. PrePSI = 1 * Re[DFT(HpPSI)]
                        cHpPSI = fft2d_func(cHpPSI)
                        cHpPSI *= SCALE

                        cPrePSI = np.empty((N0, N1), dtype=np.float64)
                        cPrePSI[:, :] = cHpPSI.real
                        del cHpPSI

                        # c. Fill Linear System with PrePSI
                        _func = SFFTModule_dict['FillLS_PSI']
                        LHMAT = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, cIdx=cIdx, cPrePSI=cPrePSI, LHMAT=LHMAT)
                        del cPrePSI

                    dt5 = time.time() - t5
                    t6 = time.time()

                    # <*****> Establish Linear System through PHI <*****> #

                    for cIdx in range(FPHI):

                        # a. Hadamard Product for PHI  [HpPHI]
                        _func = SFFTModule_dict['HadProd_PHI']
                        cHpPHI = np.empty((N0, N1), dtype=np.complex128)
                        cHpPHI = _func(SREF_pqp0q0=SREF_pqp0q0, SPixA_FTpq=SPixA_FTpq, SPixA_CFTpq=SPixA_CFTpq, cIdx=cIdx, cHpPHI=cHpPHI)

                        # b. PrePHI = SCALE_L * Re[DFT(HpPHI)]
                        cHpPHI = fft2d_func(cHpPHI)
                        cHpPHI *= SCALE

                        cPrePHI = np.empty((N0, N1), dtype=np.float64)
                        cPrePHI[:, :] = cHpPHI.real
                        cPrePHI[:, :] *= SCALE_L
                        del cHpPHI

                        # c. Fill Linear System with PrePHI
                        _func = SFFTModule_dict['FillLS_PHI']
                        LHMAT = _func(cIdx=cIdx, cPrePHI=cPrePHI, LHMAT=LHMAT)
                        del cPrePHI
                
                dt6 = time.time() - t6
                t7 = time.time()

                # <*****> Establish Linear System through THETA & DELTA <*****> #

                # a1. Hadamard Product for THETA [HpTHE]
                _func = SFFTModule_dict['HadProd_THE']
                HpTHE = np.empty((FTHE, N0, N1), dtype=np.complex128)
                HpTHE = _func(SPixA_FIij=SPixA_FIij, PixA_CFJ=PixA_CFJ, HpTHE=HpTHE)

                # a2. Hadamard Product for DELTA [HpDEL]
                _func = SFFTModule_dict['HadProd_DEL']
                HpDEL = np.empty((FDEL, N0, N1), dtype=np.complex128)
                HpDEL = _func(SPixA_FTpq, PixA_CFJ, HpDEL)

                # b1. PreTHE = 1 * Re[DFT(HpTHE)]
                # b2. PreDEL = SCALE_L * Re[DFT(HpDEL)]
                for k in range(FTHE):
                    HpTHE[k: k+1] = fft2d_func(HpTHE[k: k+1])
                HpTHE[:, :, :] *= SCALE
                
                for k in range(FDEL):
                    HpDEL[k: k+1] = fft2d_func(HpDEL[k: k+1])
                HpDEL[:, :, :] *= SCALE

                PreTHE = np.empty((FTHE, N0, N1), dtype=np.float64)
                PreTHE[:, :, :] = HpTHE.real
                del HpTHE
                
                PreDEL = np.empty((FDEL, N0, N1), dtype=np.float64)
                PreDEL[:, :, :] = HpDEL.real
                PreDEL[:, :, :] *= SCALE_L
                del HpDEL
                
                # c1. Fill Linear System with PreTHE
                _func = SFFTModule_dict['FillLS_THE']
                RHb = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, PreTHE=PreTHE, RHb=RHb)
                del PreTHE

                # c2. Fill Linear System with PreDEL
                _func = SFFTModule_dict['FillLS_DEL']
                RHb = _func(PreDEL=PreDEL, RHb=RHb)
                del PreDEL



            if SCALING_MODE == 'SEPARATE-VARYING':
                ###assert MINIMIZE_GPU_MEMORY_USAGE

                if not MINIMIZE_GPU_MEMORY_USAGE:
                    
                    # <*****> Establish Linear System through OMEGA <*****> #
                    
                    # a11. Hadamard Product for OMEGA_11 [HpOMG11]
                    _func = SFFTModule_dict['HadProd_OMG11']
                    HpOMG11 = np.empty((FOMG, N0, N1), dtype=np.complex128)
                    HpOMG11 = _func(SREF_iji0j0=SREF_iji0j0, SPixA_FIij=SPixA_FIij, SPixA_CFIij=SPixA_CFIij, HpOMG11=HpOMG11)
                
                    # b11. PreOMG11 = SCALE * Re[DFT(HpOMG11)]
                    for k in tqdm( range(FOMG) ):
                        HpOMG11[k: k+1] = fft2d_func(HpOMG11[k: k+1])
                    HpOMG11 *= SCALE

                    PreOMG11 = np.empty((FOMG, N0, N1), dtype=np.float64)
                    PreOMG11[:, :, :] = HpOMG11.real
                    PreOMG11[:, :, :] *= SCALE
                    del HpOMG11
                    
                    if VERBOSE_LEVEL in [2]:
                        logger.info('/////   d   ///// Establish OMG11')
                    
                    
                    # a01. Hadamard Product for OMEGA_01 [HpOMG01]
                    _func = SFFTModule_dict['HadProd_OMG01']
                    HpOMG01 = np.empty((FOMG, N0, N1), dtype=np.complex128)
                    HpOMG01 = _func(SREF_iji0j0=SREF_iji0j0, ScaSPixA_FIij=ScaSPixA_FIij, SPixA_CFIij=SPixA_CFIij, HpOMG01=HpOMG01)
                    
                    # b01. PreOMG01 = SCALE * Re[DFT(HpOMG01)]
                    for k in tqdm( range(FOMG) ):
                        HpOMG01[k: k+1] = fft2d_func(HpOMG01[k: k+1])
                    HpOMG01 *= SCALE
                    
                    PreOMG01 = np.empty((FOMG, N0, N1), dtype=np.float64)
                    PreOMG01[:, :] = HpOMG01.real
                    PreOMG01[:, :] *= SCALE
                    del HpOMG01
                    
                    if VERBOSE_LEVEL in [2]:
                        logger.info('/////   d   ///// Establish OMG01')
                    
                    
                    # a10. Hadamard Product for OMEGA_10 [HpOMG10]
                    _func = SFFTModule_dict['HadProd_OMG10']
                    HpOMG10 = np.empty((FOMG, N0, N1), dtype=np.complex128)
                    HpOMG10 = _func(SREF_iji0j0=SREF_iji0j0, SPixA_FIij=SPixA_FIij, ScaSPixA_CFIij=ScaSPixA_CFIij, HpOMG10=HpOMG10)
                    
                    # b01. PreOMG10 = SCALE * Re[DFT(HpOMG10)]
                    for k in tqdm( range(FOMG) ):
                        HpOMG10[k: k+1] = fft2d_func(HpOMG10[k: k+1])
                    HpOMG10 *= SCALE
                    
                    PreOMG10 = np.empty((FOMG, N0, N1), dtype=np.float64)
                    PreOMG10[:, :] = HpOMG10.real
                    PreOMG10[:, :] *= SCALE
                    del HpOMG10
                    
                    if VERBOSE_LEVEL in [2]:
                        logger.info('/////   d   ///// Establish OMG10')
                    
                    
                    # a00. Hadamard Product for OMEGA_00 [HpOMG00]
                    _func = SFFTModule_dict['HadProd_OMG00']
                    HpOMG00 = np.empty((FOMG, N0, N1), dtype=np.complex128)
                    HpOMG00 = _func(SREF_iji0j0=SREF_iji0j0, ScaSPixA_FIij=ScaSPixA_FIij, ScaSPixA_CFIij=ScaSPixA_CFIij, HpOMG00=HpOMG00)
                    
                    # b00. PreOMG00 = SCALE * Re[DFT(HpOMG00)]
                    for k in tqdm( range(FOMG) ):
                        HpOMG00[k: k+1] = fft2d_func(HpOMG00[k: k+1])
                    HpOMG00 *= SCALE

                    PreOMG00 = np.empty((FOMG, N0, N1), dtype=np.float64)
                    PreOMG00[:, :] = HpOMG00.real
                    PreOMG00[:, :] *= SCALE
                    del HpOMG00
                    
                    if VERBOSE_LEVEL in [2]:
                        logger.info('/////   d   ///// Establish OMG00')
                    
                    
                    # c. Fill Linear System with PreOMG
                    _func = SFFTModule_dict['FillLS_OMG']
                    LHMAT = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, \
                        PreOMG11=PreOMG11, PreOMG01=PreOMG01, PreOMG10=PreOMG10, PreOMG00=PreOMG00, LHMAT=LHMAT)
                    
                    del PreOMG11
                    del PreOMG01
                    del PreOMG10
                    del PreOMG00
                    
                    dt3 = time.time() - t3
                    t4 = time.time()
                    
                    if VERBOSE_LEVEL in [2]:
                        logger.info('/////   d   ///// Establish OMG                       (%.4fs)' %dt3)
                    
                    
                    # <*****> Establish Linear System through GAMMA <*****> #

                    # a1. Hadamard Product for GAMMA_1 [HpGAM1]
                    _func = SFFTModule_dict['HadProd_GAM1']
                    HpGAM1 = np.empty((FGAM, N0, N1), dtype=np.complex128)
                    HpGAM1 = _func(SREF_ijpq=SREF_ijpq, SPixA_FIij=SPixA_FIij, SPixA_CFTpq=SPixA_CFTpq, HpGAM1=HpGAM1)

                    # b1. PreGAM1 = 1 * Re[DFT(HpGAM1)]
                    for k in range(FGAM):
                        HpGAM1[k: k+1] = fft2d_func(HpGAM1[k: k+1])
                    HpGAM1 *= SCALE

                    PreGAM1 = np.empty((FGAM, N0, N1), dtype=np.float64)
                    PreGAM1[:, :] = HpGAM1.real
                    del HpGAM1
                    
                    if VERBOSE_LEVEL in [2]:
                        logger.info('/////   e   ///// Establish GAM1')
                    
                    # a0. Hadamard Product for GAMMA_0 [HpGAM0]
                    _func = SFFTModule_dict['HadProd_GAM0']
                    HpGAM0 = np.empty((FGAM, N0, N1), dtype=np.complex128)
                    HpGAM0 = _func(SREF_ijpq=SREF_ijpq, ScaSPixA_FIij=ScaSPixA_FIij, SPixA_CFTpq=SPixA_CFTpq, HpGAM0=HpGAM0)

                    # b0. PreGAM0 = 1 * Re[DFT(HpGAM0)]
                    for k in range(FGAM):
                        HpGAM0[k: k+1] = fft2d_func(HpGAM0[k: k+1])
                    HpGAM0 *= SCALE

                    PreGAM0 = np.empty((FGAM, N0, N1), dtype=np.float64)
                    PreGAM0[:, :] = HpGAM0.real
                    del HpGAM0
                    
                    if VERBOSE_LEVEL in [2]:
                        logger.info('/////   e   ///// Establish GAM0')

                    # c. Fill Linear System with PreGAM
                    _func = SFFTModule_dict['FillLS_GAM']
                    LHMAT = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, PreGAM1=PreGAM1, PreGAM0=PreGAM0, LHMAT=LHMAT)
                    
                    del PreGAM1
                    del PreGAM0

                    dt4 = time.time() - t4
                    t5 = time.time()
                    
                    if VERBOSE_LEVEL in [2]:
                        logger.info('/////   e   ///// Establish GAM                       (%.4fs)' %dt4)
                        
                        
                        
                    # <*****> Establish Linear System through PSI <*****> #

                    # a1. Hadamard Product for PSI_1 [HpPSI1]
                    _func = SFFTModule_dict['HadProd_PSI1']
                    HpPSI1 = np.empty((FPSI, N0, N1), dtype=np.complex128)
                    HpPSI1 = _func(SREF_pqij=SREF_pqij, SPixA_CFIij=SPixA_CFIij, SPixA_FTpq=SPixA_FTpq, HpPSI1=HpPSI1)

                    # b1. PrePSI1 = 1 * Re[DFT(HpPSI1)]
                    for k in range(FPSI):
                        HpPSI1[k:k+1] = fft2d_func(HpPSI1[k:k+1])
                    HpPSI1 *= SCALE

                    PrePSI1 = np.empty((FPSI, N0, N1), dtype=np.float64)
                    PrePSI1[:, :] = HpPSI1.real
                    del HpPSI1
                    
                    if VERBOSE_LEVEL in [2]:
                        logger.info('/////   f   ///// Establish PSI1')

                    # a0. Hadamard Product for PSI_0 [HpPSI0]
                    _func = SFFTModule_dict['HadProd_PSI0']
                    HpPSI0 = np.empty((FPSI, N0, N1), dtype=np.complex128)
                    HpPSI0 = _func(SREF_pqij=SREF_pqij, ScaSPixA_CFIij=ScaSPixA_CFIij, SPixA_FTpq=SPixA_FTpq, HpPSI0=HpPSI0)

                    # b1. PrePSI0 = 1 * Re[DFT(HpPSI0)]
                    for k in range(FPSI):
                        HpPSI0[k:k+1] = fft2d_func(HpPSI0[k:k+1])
                    HpPSI0 *= SCALE

                    PrePSI0 = np.empty((FPSI, N0, N1), dtype=np.float64)
                    PrePSI0[:, :] = HpPSI0.real
                    del HpPSI0
                    
                    if VERBOSE_LEVEL in [2]:
                        logger.info('/////   f   ///// Establish PSI0')

                    # c. Fill Linear System with PrePSI
                    _func = SFFTModule_dict['FillLS_PSI']
                    LHMAT = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, PrePSI1=PrePSI1, PrePSI0=PrePSI0, LHMAT=LHMAT)
                    
                    del PrePSI1
                    del PrePSI0

                    dt5 = time.time() - t5
                    t6 = time.time()
                    
                    if VERBOSE_LEVEL in [2]:
                        logger.info('/////   f   ///// Establish PSI                       (%.4fs)' %dt5)

                    
                    
                    # <*****> Establish Linear System through PHI <*****> #

                    # a. Hadamard Product for PHI [HpPHI]
                    _func = SFFTModule_dict['HadProd_PHI']
                    HpPHI = np.empty((FPHI, N0, N1), dtype=np.complex128)
                    HpPHI = _func(SREF_pqp0q0=SREF_pqp0q0, SPixA_FTpq=SPixA_FTpq, SPixA_CFTpq=SPixA_CFTpq, HpPHI=HpPHI)

                    # b. PrePHI = SCALE_L * Re[DFT(HpPHI)]
                    for k in range(FPHI):
                        HpPHI[k: k+1] = fft2d_func(HpPHI[k: k+1])
                    HpPHI *= SCALE

                    PrePHI = np.empty((FPHI, N0, N1), dtype=np.float64)
                    PrePHI[:, :] = HpPHI.real
                    PrePHI[:, :] *= SCALE_L
                    del HpPHI

                    # c. Fill Linear System with PrePHI
                    _func = SFFTModule_dict['FillLS_PHI']
                    LHMAT = _func(PrePHI=PrePHI, LHMAT=LHMAT)
                    del PrePHI

                    dt6 = time.time() - t6
                    t7 = time.time()
                    
                    if VERBOSE_LEVEL in [2]:
                        logger.info('/////   g   ///// Establish PHI                       (%.4fs)' %dt6)


                
                if MINIMIZE_GPU_MEMORY_USAGE:

                    # <*****> Establish Linear System through OMEGA <*****> #

                    for cIdx in tqdm( range(FOMG) ):    
                        
                        # a11. Hadamard Product for OMEGA_11 [HpOMG11]
                        _func = SFFTModule_dict['HadProd_OMG11']
                        cHpOMG11 = np.empty((N0, N1), dtype=np.complex128)
                        cHpOMG11 = _func(SREF_iji0j0=SREF_iji0j0, SPixA_FIij=SPixA_FIij, SPixA_CFIij=SPixA_CFIij, cIdx=cIdx, cHpOMG11=cHpOMG11)
                        
                        # b11. PreOMG11 = SCALE * Re[DFT(HpOMG11)]
                        # cHpOMG11 = pyfftw.interfaces.numpy_fft.fft2(cHpOMG11)
                        cHpOMG11 = fft2d_func(cHpOMG11)
                        cHpOMG11 *= SCALE

                        cPreOMG11 = np.empty((N0, N1), dtype=np.float64)
                        cPreOMG11[:, :] = cHpOMG11.real
                        cPreOMG11[:, :] *= SCALE
                        del cHpOMG11

                        # a01. Hadamard Product for OMEGA_01 [HpOMG01]
                        _func = SFFTModule_dict['HadProd_OMG01']
                        cHpOMG01 = np.empty((N0, N1), dtype=np.complex128)
                        cHpOMG01 = _func(SREF_iji0j0=SREF_iji0j0, ScaSPixA_FIij=ScaSPixA_FIij, SPixA_CFIij=SPixA_CFIij, cIdx=cIdx, cHpOMG01=cHpOMG01)
                        
                        # b01. PreOMG01 = SCALE * Re[DFT(HpOMG01)]
                        # cHpOMG01 = pyfftw.interfaces.numpy_fft.fft2(cHpOMG01)
                        cHpOMG01 = fft2d_func(cHpOMG01)
                        cHpOMG01 *= SCALE

                        cPreOMG01 = np.empty((N0, N1), dtype=np.float64)
                        cPreOMG01[:, :] = cHpOMG01.real
                        cPreOMG01[:, :] *= SCALE
                        del cHpOMG01

                        # a10. Hadamard Product for OMEGA_10 [HpOMG10]
                        _func = SFFTModule_dict['HadProd_OMG10']
                        cHpOMG10 = np.empty((N0, N1), dtype=np.complex128)
                        cHpOMG10 = _func(SREF_iji0j0=SREF_iji0j0, SPixA_FIij=SPixA_FIij, ScaSPixA_CFIij=ScaSPixA_CFIij, cIdx=cIdx, cHpOMG10=cHpOMG10)
                        
                        # b10. PreOMG10 = SCALE * Re[DFT(HpOMG10)]
                        # cHpOMG10 = pyfftw.interfaces.numpy_fft.fft2(cHpOMG10)
                        cHpOMG10 = fft2d_func(cHpOMG10)
                        cHpOMG10 *= SCALE

                        cPreOMG10 = np.empty((N0, N1), dtype=np.float64)
                        cPreOMG10[:, :] = cHpOMG10.real
                        cPreOMG10[:, :] *= SCALE
                        del cHpOMG10

                        # a00. Hadamard Product for OMEGA_00 [HpOMG00]
                        _func = SFFTModule_dict['HadProd_OMG00']
                        cHpOMG00 = np.empty((N0, N1), dtype=np.complex128)
                        cHpOMG00 = _func(SREF_iji0j0=SREF_iji0j0, ScaSPixA_FIij=ScaSPixA_FIij, ScaSPixA_CFIij=ScaSPixA_CFIij, cIdx=cIdx, cHpOMG00=cHpOMG00)
                        
                        # b00. PreOMG00 = SCALE * Re[DFT(HpOMG00)]
                        # cHpOMG00 = pyfftw.interfaces.numpy_fft.fft2(cHpOMG00)
                        cHpOMG00 = fft2d_func(cHpOMG00)
                        cHpOMG00 *= SCALE

                        cPreOMG00 = np.empty((N0, N1), dtype=np.float64)
                        cPreOMG00[:, :] = cHpOMG00.real
                        cPreOMG00[:, :] *= SCALE
                        del cHpOMG00

                        # c. Fill Linear System with PreOMG
                        _func = SFFTModule_dict['FillLS_OMG']
                        LHMAT = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, cIdx=cIdx, \
                            cPreOMG11=cPreOMG11, cPreOMG01=cPreOMG01, cPreOMG10=cPreOMG10, cPreOMG00=cPreOMG00, LHMAT=LHMAT)
                        
                        del cPreOMG11
                        del cPreOMG01
                        del cPreOMG10
                        del cPreOMG00

                    dt3 = time.time() - t3
                    t4 = time.time()
                    
                    if VERBOSE_LEVEL in [2]:
                        logger.info('/////   d   ///// Establish OMG                       (%.4fs)' %dt3)


                    # <*****> Establish Linear System through GAMMA <*****> #

                    for cIdx in tqdm( range(FGAM) ):

                        # a1. Hadamard Product for GAMMA_1 [HpGAM1]
                        _func = SFFTModule_dict['HadProd_GAM1']
                        cHpGAM1 = np.empty((N0, N1), dtype=np.complex128)
                        cHpGAM1 = _func(SREF_ijpq=SREF_ijpq, SPixA_FIij=SPixA_FIij, SPixA_CFTpq=SPixA_CFTpq, cIdx=cIdx, cHpGAM1=cHpGAM1)

                        # b1. PreGAM1 = 1 * Re[DFT(HpGAM1)]
                        # cHpGAM1 = pyfftw.interfaces.numpy_fft.fft2(cHpGAM1)
                        cHpGAM1 = fft2d_func(cHpGAM1)
                        cHpGAM1 *= SCALE

                        cPreGAM1 = np.empty((N0, N1), dtype=np.float64)
                        cPreGAM1[:, :] = cHpGAM1.real
                        del cHpGAM1
                        
                        # a0. Hadamard Product for GAMMA_0 [HpGAM0]
                        _func = SFFTModule_dict['HadProd_GAM0']
                        cHpGAM0 = np.empty((N0, N1), dtype=np.complex128)
                        cHpGAM0 = _func(SREF_ijpq=SREF_ijpq, ScaSPixA_FIij=ScaSPixA_FIij, SPixA_CFTpq=SPixA_CFTpq, cIdx=cIdx, cHpGAM0=cHpGAM0)

                        # b0. PreGAM0 = 1 * Re[DFT(HpGAM0)]
                        # cHpGAM0 = pyfftw.interfaces.numpy_fft.fft2(cHpGAM0)
                        cHpGAM0 = fft2d_func(cHpGAM0)
                        cHpGAM0 *= SCALE

                        cPreGAM0 = np.empty((N0, N1), dtype=np.float64)
                        cPreGAM0[:, :] = cHpGAM0.real
                        del cHpGAM0

                        # c. Fill Linear System with PreGAM
                        _func = SFFTModule_dict['FillLS_GAM']
                        LHMAT = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, cIdx=cIdx, cPreGAM1=cPreGAM1, cPreGAM0=cPreGAM0, LHMAT=LHMAT)
                        
                        del cPreGAM1
                        del cPreGAM0

                    dt4 = time.time() - t4
                    t5 = time.time()
                    
                    if VERBOSE_LEVEL in [2]:
                        logger.info('/////   e   ///// Establish GAM                       (%.4fs)' %dt4)


                    # <*****> Establish Linear System through PSI <*****> #

                    for cIdx in tqdm( range(FPSI) ):

                        # a1. Hadamard Product for PSI_1 [HpPSI1]
                        _func = SFFTModule_dict['HadProd_PSI1']
                        cHpPSI1 = np.empty((N0, N1), dtype=np.complex128)
                        cHpPSI1 = _func(SREF_pqij=SREF_pqij, SPixA_CFIij=SPixA_CFIij, SPixA_FTpq=SPixA_FTpq, cIdx=cIdx, cHpPSI1=cHpPSI1)

                        # b1. PrePSI1 = 1 * Re[DFT(HpPSI1)]
                        # cHpPSI1 = pyfftw.interfaces.numpy_fft.fft2(cHpPSI1)
                        cHpPSI1 = fft2d_func(cHpPSI1)
                        cHpPSI1 *= SCALE

                        cPrePSI1 = np.empty((N0, N1), dtype=np.float64)
                        cPrePSI1[:, :] = cHpPSI1.real
                        del cHpPSI1

                        # a0. Hadamard Product for PSI_0 [HpPSI0]
                        _func = SFFTModule_dict['HadProd_PSI0']
                        cHpPSI0 = np.empty((N0, N1), dtype=np.complex128)
                        cHpPSI0 = _func(SREF_pqij=SREF_pqij, ScaSPixA_CFIij=ScaSPixA_CFIij, SPixA_FTpq=SPixA_FTpq, cIdx=cIdx, cHpPSI0=cHpPSI0)

                        # b1. PrePSI0 = 1 * Re[DFT(HpPSI0)]
                        # cHpPSI0 = pyfftw.interfaces.numpy_fft.fft2(cHpPSI0)
                        cHpPSI0 = fft2d_func(cHpPSI0)
                        cHpPSI0 *= SCALE

                        cPrePSI0 = np.empty((N0, N1), dtype=np.float64)
                        cPrePSI0[:, :] = cHpPSI0.real
                        del cHpPSI0

                        # c. Fill Linear System with PrePSI
                        _func = SFFTModule_dict['FillLS_PSI']
                        LHMAT = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, cIdx=cIdx, cPrePSI1=cPrePSI1, cPrePSI0=cPrePSI0, LHMAT=LHMAT)
                        
                        del cPrePSI1
                        del cPrePSI0

                    dt5 = time.time() - t5
                    t6 = time.time()
                    
                    if VERBOSE_LEVEL in [2]:
                        logger.info('/////   f   ///// Establish PSI                       (%.4fs)' %dt5)

                    # <*****> Establish Linear System through PHI <*****> #

                    for cIdx in tqdm( range(FPHI) ):
                        # a. Hadamard Product for PHI [HpPHI]
                        _func = SFFTModule_dict['HadProd_PHI']
                        cHpPHI = np.empty((N0, N1), dtype=np.complex128)
                        cHpPHI = _func(SREF_pqp0q0=SREF_pqp0q0, SPixA_FTpq=SPixA_FTpq, SPixA_CFTpq=SPixA_CFTpq, cIdx=cIdx, cHpPHI=cHpPHI)

                        # b. PrePHI = SCALE_L * Re[DFT(HpPHI)]
                        # cHpPHI = pyfftw.interfaces.numpy_fft.fft2(cHpPHI)
                        cHpPHI = fft2d_func(cHpPHI)
                        cHpPHI *= SCALE

                        cPrePHI = np.empty((N0, N1), dtype=np.float64)
                        cPrePHI[:, :] = cHpPHI.real
                        cPrePHI[:, :] *= SCALE_L
                        del cHpPHI

                        # c. Fill Linear System with PrePHI
                        _func = SFFTModule_dict['FillLS_PHI']
                        LHMAT = _func(cIdx=cIdx, cPrePHI=cPrePHI, LHMAT=LHMAT)
                        del cPrePHI

                dt6 = time.time() - t6
                t7 = time.time()
                
                if VERBOSE_LEVEL in [2]:
                    logger.info('/////   g   ///// Establish PHI                       (%.4fs)' %dt6)
                
                # <*****> Establish Linear System through THETA & DELTA <*****> #

                # a1. Hadamard Product for THETA_1 [HpTHE1]
                _func = SFFTModule_dict['HadProd_THE1']
                HpTHE1 = np.empty((FTHE, N0, N1), dtype=np.complex128)
                HpTHE1 = _func(SPixA_FIij=SPixA_FIij, PixA_CFJ=PixA_CFJ, HpTHE1=HpTHE1)

                # a0. Hadamard Product for THETA_0 [HpTHE0]
                _func = SFFTModule_dict['HadProd_THE0']
                HpTHE0 = np.empty((FTHE, N0, N1), dtype=np.complex128)
                HpTHE0 = _func(ScaSPixA_FIij=ScaSPixA_FIij, PixA_CFJ=PixA_CFJ, HpTHE0=HpTHE0)

                # x. Hadamard Product for DELTA [HpDEL]
                _func = SFFTModule_dict['HadProd_DEL']
                HpDEL = np.empty((FDEL, N0, N1), dtype=np.complex128)
                HpDEL = _func(SPixA_FTpq=SPixA_FTpq, PixA_CFJ=PixA_CFJ, HpDEL=HpDEL)

                # b1. PreTHE1 = 1 * Re[DFT(HpTHE1)]
                # b0. PreTHE0 = 1 * Re[DFT(HpTHE0)]
                # y. PreDEL = SCALE_L * Re[DFT(HpDEL)]

                for k in range(FTHE):
                    # HpTHE1[k: k+1] = pyfftw.interfaces.numpy_fft.fft2(HpTHE1[k: k+1])
                    HpTHE1[k: k+1] = fft3d_func(HpTHE1[k: k+1])
                HpTHE1[:, :, :] *= SCALE

                PreTHE1 = np.empty((FTHE, N0, N1), dtype=np.float64)
                PreTHE1[:, :, :] = HpTHE1.real
                del HpTHE1

                for k in range(FTHE):
                    # HpTHE0[k: k+1] = pyfftw.interfaces.numpy_fft.fft2(HpTHE0[k: k+1])
                    HpTHE0[k: k+1] = fft3d_func(HpTHE0[k: k+1])
                HpTHE0[:, :, :] *= SCALE

                PreTHE0 = np.empty((FTHE, N0, N1), dtype=np.float64)
                PreTHE0[:, :, :] = HpTHE0.real
                del HpTHE0

                for k in range(FDEL):
                    # HpDEL[k: k+1] = pyfftw.interfaces.numpy_fft.fft2(HpDEL[k: k+1])
                    HpDEL[k: k+1] = fft3d_func(HpDEL[k: k+1])
                HpDEL[:, :, :] *= SCALE

                PreDEL = np.empty((FDEL, N0, N1), dtype=np.float64)
                PreDEL[:, :, :] = HpDEL.real
                PreDEL[:, :, :] *= SCALE_L
                del HpDEL

                # c. Fill Linear System with PreTHE
                _func = SFFTModule_dict['FillLS_THE']
                RHb = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, PreTHE1=PreTHE1, PreTHE0=PreTHE0, RHb=RHb)
                
                del PreTHE1
                del PreTHE0

                # z. Fill Linear System with PreDEL
                _func = SFFTModule_dict['FillLS_DEL']
                RHb = _func(PreDEL=PreDEL, RHb=RHb)
                del PreDEL
                
            dt7 = time.time() - t7
            t8 = time.time()
            
            if VERBOSE_LEVEL in [2]:
                logger.info('/////   h   ///// Establish THE & DEL                 (%.4fs)' %dt7)

            # <*****> Regularize Linear System <*****> #

            if REGULARIZE_KERNEL:
                
                NREG = XY_REGULARIZE.shape[0]
                CX_REG = XY_REGULARIZE[:, 0]/N0
                CY_REG = XY_REGULARIZE[:, 1]/N1

                if KerSpType == 'Polynomial':

                    SPMAT = np.array([
                        CX_REG**i * CY_REG**j 
                        for i in range(DK+1) for j in range(DK+1-i)
                    ])

                if KerSpType == 'B-Spline':
                    KerSplBasisX_REG = Create_BSplineBasis_Req(N=N0, IntKnot=KerIntKnotX, BSplineDegree=DK, ReqCoord=CX_REG)
                    KerSplBasisY_REG = Create_BSplineBasis_Req(N=N1, IntKnot=KerIntKnotY, BSplineDegree=DK, ReqCoord=CY_REG)
                    
                    SPMAT = np.array([
                        KerSplBasisX_REG[i] * KerSplBasisY_REG[j]
                        for i in range(Fi) for j in range(Fj)
                    ])

                SPMAT = np.array(SPMAT, dtype=np.float64)

                if SCALING_MODE == 'SEPARATE-VARYING':

                    if ScaSpType == 'Polynomial':

                        ScaSPMAT = np.array([
                            CX_REG**i * CY_REG**j 
                            for i in range(DS+1) for j in range(DS+1-i)
                        ])

                    if ScaSpType == 'B-Spline':
                        ScaSplBasisX_REG = Create_BSplineBasis_Req(N=N0, IntKnot=ScaIntKnotX, BSplineDegree=DS, ReqCoord=CX_REG)
                        ScaSplBasisY_REG = Create_BSplineBasis_Req(N=N1, IntKnot=ScaIntKnotY, BSplineDegree=DS, ReqCoord=CY_REG)
                        
                        ScaSPMAT = np.array([
                            ScaSplBasisX_REG[i] * ScaSplBasisY_REG[j]
                            for i in range(ScaFi) for j in range(ScaFj)
                        ])
                    
                    # placeholder
                    if ScaFij < Fij:
                        ScaSPMAT = np.concatenate((
                            ScaSPMAT, 
                            np.zeros((Fij-ScaFij, NREG), dtype=np.float64)
                            ), axis=0
                        )

                    ScaSPMAT = np.array(ScaSPMAT, dtype=np.float64)

                if WEIGHT_REGULARIZE is None:
                    SSTMAT = np.matmul(SPMAT, SPMAT.T)/NREG   # symmetric
                    if SCALING_MODE == 'SEPARATE-VARYING':
                        CSSTMAT = np.matmul(SPMAT, ScaSPMAT.T)/NREG     # C: Cross
                        DSSTMAT = np.matmul(ScaSPMAT, ScaSPMAT.T)/NREG  # D: Double, symmetric

                if WEIGHT_REGULARIZE is not None:
                    # weighted average over regularization points
                    WSPMAT = np.diag(WEIGHT_REGULARIZE)
                    WSPMAT /= np.sum(WEIGHT_REGULARIZE)  # normalize to have unit sum
                    WSPMAT = np.array(WSPMAT, dtype=np.float64)

                    SSTMAT = np.matmul(np.matmul(SPMAT, WSPMAT), SPMAT.T)   # symmetric
                    if SCALING_MODE == 'SEPARATE-VARYING':
                        CSSTMAT = np.matmul(p.matmul(SPMAT, WSPMAT), ScaSPMAT.T)   # C: Cross
                        DSSTMAT = np.matmul(p.matmul(ScaSPMAT, WSPMAT), ScaSPMAT.T)   # D: Double, symmetric

                # Create Laplacian Matrix
                LAPMAT = np.zeros((Fab, Fab)).astype(np.int32)
                RR, CC = np.mgrid[0: L0, 0: L1]
                RRF = RR.flatten().astype(np.int32)
                CCF = CC.flatten().astype(np.int32)

                AdCOUNT = signal.correlate2d(
                    np.ones((L0, L1)), 
                    np.array([[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]]),
                    mode='same', 
                    boundary='fill',
                    fillvalue=0
                ).astype(np.int32)

                # fill diagonal elements
                KIDX = np.arange(Fab)
                LAPMAT[KIDX, KIDX] = AdCOUNT.flatten()[KIDX]

                LAPMAT = np.array(LAPMAT, dtype=np.int32)
                RRF = np.array(RRF, dtype=np.int32)
                CCF = np.array(CCF, dtype=np.int32)

                # fill non-diagonal 
                _func = SFFTModule_dict['fill_lapmat_nondiagonal']
                LAPMAT = _func(LAPMAT=LAPMAT, RRF=RRF, CCF=CCF)

                # zero-out kernel-center rows of laplacian matrix
                # FIXME: one can multiply user-defined weights of kernel pixels 
                if IGNORE_LAPLACIAN_KERCENT:
                    
                    LAPMAT[(w0-1)*L1+w1, :] = 0.0 
                    LAPMAT[w0*L1+w1-1, :] = 0.0
                    LAPMAT[w0*L1+w1, :] = 0.0
                    LAPMAT[w0*L1+w1+1, :] = 0.0
                    LAPMAT[(w0+1)*L1+w1, :] = 0.0

                LTLMAT = np.matmul(LAPMAT.T, LAPMAT)   # symmetric
                
                # Create iREGMAT
                iREGMAT = np.zeros((Fab, Fab), dtype=np.int32)
                _func = SFFTModule_dict['fill_iregmat']    
                iREGMAT = _func(iREGMAT=iREGMAT, LTLMAT=LTLMAT)

                # Create REGMAT
                REGMAT = np.zeros((NEQ, NEQ), dtype=np.float64)
                if SCALING_MODE in ['ENTANGLED', 'SEPARATE-CONSTANT']:
                    _func = SFFTModule_dict['fill_regmat']    
                    REGMAT = _func(iREGMAT=iREGMAT, SSTMAT=SSTMAT, REGMAT=REGMAT)

                if SCALING_MODE == 'SEPARATE-VARYING':
                    _func = SFFTModule_dict['fill_regmat']    
                    REGMAT = _func(iREGMAT=iREGMAT, SSTMAT=SSTMAT, CSSTMAT=CSSTMAT, DSSTMAT=DSSTMAT, REGMAT=REGMAT)

                # UPDATE LHMAT
                LHMAT += LAMBDA_REGULARIZE * REGMAT
            
            # <*****> Tweak Linear System <*****> #

            if SCALING_MODE == 'ENTANGLED' or (SCALING_MODE == 'SEPARATE-VARYING' and NEQt == NEQ):
                pass

            if SCALING_MODE == 'SEPARATE-CONSTANT':
                
                LHMAT_tweaked = np.empty((NEQt, NEQt), dtype=np.float64)
                RHb_tweaked = np.empty(NEQt, dtype=np.float64)

                PresIDX = np.setdiff1d(np.arange(NEQ), ij00[1:], assume_unique=True).astype(np.int32)
                assert np.all(PresIDX[:-1] < PresIDX[1:])
                assert PresIDX[ij00[0]] == ij00[0]
                PresIDX = np.array(PresIDX)


                if KerSpType == 'Polynomial':
                    _func = SFFTModule_dict['TweakLS']
                    LHMAT_tweaked, RHb_tweaked = _func(LHMAT=LHMAT, RHb=RHb, PresIDX=PresIDX, LHMAT_tweaked=LHMAT_tweaked, \
                                                       RHb_tweaked=RHb_tweaked)
                
                if KerSpType == 'B-Spline':
                    _func = SFFTModule_dict['TweakLS']
                    LHMAT_tweaked, RHb_tweaked = _func(LHMAT=LHMAT, RHb=RHb, PresIDX=PresIDX, ij00=ij00, \
                                                       LHMAT_tweaked=LHMAT_tweaked, RHb_tweaked=RHb_tweaked)
            
            
            
            if SCALING_MODE == 'SEPARATE-VARYING' and NEQt < NEQ:

                LHMAT_tweaked = np.empty((NEQt, NEQt), dtype=np.float64)
                RHb_tweaked = np.empty(NEQt, dtype=np.float64)

                PresIDX = np.setdiff1d(np.arange(NEQ), ij00[ScaFij:], assume_unique=True).astype(np.int32)
                assert np.all(PresIDX[:-1] < PresIDX[1:])
                assert PresIDX[ij00[0]] == ij00[0]
                PresIDX = np.array(PresIDX)

                
                _func = SFFTModule_dict['TweakLS']
                LHMAT_tweaked, RHb_tweaked = _func(LHMAT=LHMAT, RHb=RHb, PresIDX=PresIDX, LHMAT_tweaked=LHMAT_tweaked, \
                                                   RHb_tweaked=RHb_tweaked)
            
            # <*****> Solve Linear System & Restore Solution <*****> #

            if SCALING_MODE == 'ENTANGLED' or (SCALING_MODE == 'SEPARATE-VARYING' and NEQt == NEQ):
                Solution = solver(LHMAT, RHb)

            if SCALING_MODE == 'SEPARATE-CONSTANT':
                Solution_tweaked = solver(LHMAT_tweaked, RHb_tweaked)

                if KerSpType == 'Polynomial':
                    Solution = np.zeros(NEQ, dtype=np.float64)
                
                if KerSpType == 'B-Spline':
                    Solution = np.zeros(NEQ, dtype=np.float64)
                    Solution[ij00[1:]] = Solution_tweaked[ij00[0]]

                _func = SFFTModule_dict['Restore_Solution']
                Solution = _func(Solution_tweaked=Solution_tweaked, PresIDX=PresIDX, Solution=Solution)


            if SCALING_MODE == 'SEPARATE-VARYING' and NEQt < NEQ:
                Solution_tweaked = solver(LHMAT_tweaked, RHb_tweaked)

                Solution = np.zeros(NEQ, dtype=np.float64)
                _func = SFFTModule_dict['Restore_Solution']
                Solution = _func(Solution_tweaked=Solution_tweaked, PresIDX=PresIDX, Solution=Solution)

            a_ijab = Solution[: Fijab]
            b_pq = Solution[Fijab: ]

            dt8 = time.time() - t8
            dtb = time.time() - tb


            if VERBOSE_LEVEL in [2]:
                logger.info('/////   i   ///// Solve Linear System                 (%.4fs)' %dt8)

            if VERBOSE_LEVEL in [1, 2]:
                logger.info('MeLOn CheckPoint: SFFT-EXECUTION Establish & Solve Linear System takes [%.4fs]' %dtb)

        # <*****> Perform Subtraction/Coaddition  <*****> #
        PixA_OUT = None
        if Subtract or Coadd:
            tc = time.time()
            t9 = time.time()

            # Calculate Kab components
            Wl = np.exp((-2j*np.pi/N0) * PixA_X.astype(np.float64))    # row index l, [0, N0)
            Wm = np.exp((-2j*np.pi/N1) * PixA_Y.astype(np.float64))    # column index m, [0, N1)
            Kab_Wla = np.empty((L0, N0, N1), dtype=np.complex128)
            Kab_Wmb = np.empty((L1, N0, N1), dtype=np.complex128)

            if w0 == w1:
                wx = w0   # a little bit faster
                for aob in range(-wx, wx+1):
                    Kab_Wla[aob + wx] = Wl ** aob    # offset 
                    Kab_Wmb[aob + wx] = Wm ** aob    # offset 
            else:
                for a in range(-w0, w0+1): 
                    Kab_Wla[a + w0] = Wl ** a        # offset 
                for b in range(-w1, w1+1): 
                    Kab_Wmb[b + w1] = Wm ** b        # offset 
            
            dt9 = time.time() - t9
            t10 = time.time()
            

        if Subtract:
            # Construct Difference in Fourier Space
            if SCALING_MODE in ['ENTANGLED', 'SEPARATE-CONSTANT']:
                _func = SFFTModule_dict['Construct_FDIFF']  
                PixA_FDIFF = np.empty((N0, N1), dtype=np.complex128)
                PixA_FDIFF = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, a_ijab=a_ijab.astype(np.complex128), \
                    SPixA_FIij=SPixA_FIij, Kab_Wla=Kab_Wla, Kab_Wmb=Kab_Wmb, b_pq=b_pq.astype(np.complex128), \
                    SPixA_FTpq=SPixA_FTpq, PixA_FJ=PixA_FJ, PixA_FDIFF=PixA_FDIFF)
                
            if SCALING_MODE == 'SEPARATE-VARYING':
                _func = SFFTModule_dict['Construct_FDIFF']   
                PixA_FDIFF = np.empty((N0, N1), dtype=np.complex128)
                PixA_FDIFF = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, a_ijab=a_ijab.astype(np.complex128), \
                    SPixA_FIij=SPixA_FIij, ScaSPixA_FIij=ScaSPixA_FIij, Kab_Wla=Kab_Wla, Kab_Wmb=Kab_Wmb, b_pq=b_pq.astype(np.complex128), \
                    SPixA_FTpq=SPixA_FTpq, PixA_FJ=PixA_FJ, PixA_FDIFF=PixA_FDIFF)
            
            # Get Difference & Reconstructed Images
            PixA_DIFF = np.empty_like(PixA_FDIFF)
            # PixA_DIFF = SCALE_L * pyfftw.interfaces.numpy_fft.ifft2(PixA_FDIFF)
            PixA_DIFF = SCALE_L * ifft2d_func(PixA_FDIFF)
            PixA_DIFF = PixA_DIFF.real
            
            
            dt10 = time.time() - t10
            dtc = time.time() - tc

            if VERBOSE_LEVEL in [1, 2]:
                logger.info('MeLOn CheckPoint: SFFT-SUBTRACTION Perform Subtraction takes [%.4fs]' %dtc)
            
            if VERBOSE_LEVEL in [2]:
                logger.info('/////   j   ///// Calculate Kab         (%.4fs)' %dt9)
                logger.info('/////   k   ///// Construct DIFF        (%.4fs)' %dt10)
                
                
            PixA_OUT = PixA_DIFF.copy()
                
                
                
        if Coadd:
            # Construct Difference in Fourier Space
            if SCALING_MODE in ['ENTANGLED', 'SEPARATE-CONSTANT']:
                _func = SFFTModule_dict['Construct_FCOADD']  
                PixA_FCOADD = np.empty((N0, N1), dtype=np.complex128)
                PixA_FCOADD = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, a_ijab=a_ijab.astype(np.complex128), \
                    SPixA_FIij=SPixA_FIij, Kab_Wla=Kab_Wla, Kab_Wmb=Kab_Wmb, b_pq=b_pq.astype(np.complex128), \
                    SPixA_FTpq=SPixA_FTpq, PixA_FJ=PixA_FJ, wI=wI, wJ=wJ, PixA_FCOADD=PixA_FCOADD)
                
            if SCALING_MODE == 'SEPARATE-VARYING':
                _func = SFFTModule_dict['Construct_FCOADD']   
                PixA_FCOADD = np.empty((N0, N1), dtype=np.complex128)
                PixA_FCOADD = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, a_ijab=a_ijab.astype(np.complex128), \
                    SPixA_FIij=SPixA_FIij, ScaSPixA_FIij=ScaSPixA_FIij, Kab_Wla=Kab_Wla, Kab_Wmb=Kab_Wmb, b_pq=b_pq.astype(np.complex128), \
                    SPixA_FTpq=SPixA_FTpq, PixA_FJ=PixA_FJ, wI=wI, wJ=wJ, PixA_FCOADD=PixA_FCOADD)
            
            # Get Difference & Reconstructed Images
            PixA_COADD = np.empty_like(PixA_FCOADD)
            # PixA_COADD = SCALE_L * pyfftw.interfaces.numpy_fft.ifft2(PixA_FCOADD)
            PixA_COADD = SCALE_L * ifft2d_func(PixA_FCOADD)
            PixA_COADD = PixA_COADD.real
            
            
            dt10 = time.time() - t10
            dtc = time.time() - tc

            if VERBOSE_LEVEL in [1, 2]:
                logger.info('MeLOn CheckPoint: SFFT-COADDITION Perform COADDITION takes [%.4fs]' %dtc)
            
            if VERBOSE_LEVEL in [2]:
                logger.info('/////   j   ///// Calculate Kab         (%.4fs)' %dt9)
                logger.info('/////   k   ///// Construct COADD       (%.4fs)' %dt10)
                
            PixA_OUT = PixA_COADD.copy()
        

        if VERBOSE_LEVEL in [1, 2]:
            logger.info('--||--||--||--||-- EXIT SFFT [Numpy] --||--||--||--||--')

        return Solution, PixA_OUT
    


class ElementalSFFTSubtract:
    @staticmethod
    def ESN(PixA_I, PixA_J, SFFTConfig, SFFTSolution=None, Subtract=False, Coadd=False, \
        BACKEND_4SUBTRACT='Numpy', NUM_CPU_THREADS_4SUBTRACT=8, logger=None, VERBOSE_LEVEL=2):

        if BACKEND_4SUBTRACT == 'Numpy':
            Solution, PixA_DIFF = ElementalSFFTSubtract_Numpy.ESSN(PixA_I=PixA_I, PixA_J=PixA_J, \
                SFFTConfig=SFFTConfig, SFFTSolution=SFFTSolution, Subtract=Subtract, Coadd=Coadd, \
                logger=logger, VERBOSE_LEVEL=VERBOSE_LEVEL)

        if BACKEND_4SUBTRACT == 'Cupy':
            print('MeLOn ERROR: Use other class for cupy integration!')
        
        return Solution, PixA_DIFF
    
    

    
class GeneralSFFTSubtract:
    @staticmethod
    def GSN(PixA_I, PixA_J, PixA_mI, PixA_mJ, SFFTConfig, ContamMask_I=None, \
        BACKEND_4SUBTRACT='Numpy', NUM_CPU_THREADS_4SUBTRACT=8, logger=None, VERBOSE_LEVEL=2):

        """
        # Perform image subtraction on I & J with SFFT parameters solved from mI & mJ
        #
        # Arguments:
        # -PixA_I: Image I that will be convolved [NaN-Free]                                    
        # -PixA_J: Image J that won't be convolved [NaN-Free]                                   
        # -PixA_mI: Masked version of Image I. 'same' means it is identical with I [NaN-Free]  
        # -PixA_mJ: Masked version of Image J. 'same' means it is identical with J [NaN-Free]
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
            logger.info(SFFT_BANNER)
        
        # * Check Size of input images
        tmplst = [PixA_I.shape, PixA_J.shape, PixA_mI.shape, PixA_mI.shape]
        if len(set(tmplst)) > 1:
            logger.error('MeLOn ERROR: Input images should have same size!')
            raise Exception('MeLOn ERROR: Input images should have same size!')
        
        # * Subtraction Solution derived from input masked image-pair
        Solution = ElementalSFFTSubtract.ESN(PixA_I=PixA_mI, PixA_J=PixA_mJ, SFFTConfig=SFFTConfig, \
            SFFTSolution=None, Subtract=False, Coadd=False, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, logger=logger, VERBOSE_LEVEL=VERBOSE_LEVEL)[0]
            
        # * Subtraction of the input image-pair (use above solution)
        PixA_DIFF = ElementalSFFTSubtract.ESN(PixA_I=PixA_I, PixA_J=PixA_J, SFFTConfig=SFFTConfig, \
            SFFTSolution=Solution, Subtract=True, Coadd=False, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, logger=logger, VERBOSE_LEVEL=VERBOSE_LEVEL)[1]
        
        # * Identify propagated contamination region through convolving I
        ContamMask_CI = None
        if ContamMask_I is not None:
            tSolution = Solution.copy()
            Fpq = SFFTConfig[0]['Fpq']
            tSolution[-Fpq:] = 0.0

            _tmpI = ContamMask_I.astype(np.float64)
            _tmpJ = np.zeros(PixA_J.shape).astype(np.float64)
            _tmpD = ElementalSFFTSubtract.ESN(PixA_I=_tmpI, PixA_J=_tmpJ, SFFTConfig=SFFTConfig, \
                SFFTSolution=tSolution, Subtract=True, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
                NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, logger=logger, VERBOSE_LEVEL=VERBOSE_LEVEL)[1]
            
            FTHRESH = -0.001  # emperical value
            ContamMask_CI = _tmpD < FTHRESH
        
        return Solution, PixA_DIFF, ContamMask_CI
    
    
class BSpline_Packet:
    @staticmethod
    def BSP(FITS_REF, FITS_SCI, FITS_mREF, FITS_mSCI, FITS_DIFF=None, FITS_Solution=None, \
        ForceConv='REF', GKerHW=8, KerSpType='Polynomial', KerSpDegree=2, KerIntKnotX=[], KerIntKnotY=[], \
        SEPARATE_SCALING=True, ScaSpType='Polynomial', ScaSpDegree=0, ScaIntKnotX=[], ScaIntKnotY=[], \
        BkgSpType='Polynomial', BkgSpDegree=2, BkgIntKnotX=[], BkgIntKnotY=[], \
        REGULARIZE_KERNEL=False, IGNORE_LAPLACIAN_KERCENT=True, XY_REGULARIZE=None, 
        WEIGHT_REGULARIZE=None, LAMBDA_REGULARIZE=1e-6, BACKEND_4SUBTRACT='Cupy', \
        CUDA_DEVICE_4SUBTRACT='0', MAX_THREADS_PER_BLOCK=8, MINIMIZE_GPU_MEMORY_USAGE=False, \
        NUM_CPU_THREADS_4SUBTRACT=8, logger=None, ngpu=0, VERBOSE_LEVEL=2):
        
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
            logger.info('MeLOn CheckPoint: TRIGGER Function Compilations of SFFT-SUBTRACTION!')

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
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, logger=logger, VERBOSE_LEVEL=VERBOSE_LEVEL)

        if VERBOSE_LEVEL in [1, 2]:
            _message = 'FUNCTION COMPILATIONS OF SFFT-SUBTRACTION TAKES [%.3f s]' %(time.time() - Tcomp_start)
            logger.info('MeLOn Report: %s' %_message)

        # * Perform SFFT Subtraction
        if ConvdSide == 'REF':
            PixA_mI, PixA_mJ = PixA_mREF, PixA_mSCI
            if NaNmask_U is not None:
                PixA_I, PixA_J = PixA_REF.copy(), PixA_SCI.copy()
                PixA_I[NaNmask_U] = PixA_mI[NaNmask_U]
                PixA_J[NaNmask_U] = PixA_mJ[NaNmask_U]
            else: PixA_I, PixA_J = PixA_REF, PixA_SCI

        if ConvdSide == 'SCI':
            PixA_mI, PixA_mJ = PixA_mSCI, PixA_mREF
            if NaNmask_U is not None:
                PixA_I, PixA_J = PixA_SCI.copy(), PixA_REF.copy()
                PixA_I[NaNmask_U] = PixA_mI[NaNmask_U]
                PixA_J[NaNmask_U] = PixA_mJ[NaNmask_U]
            else: PixA_I, PixA_J = PixA_SCI, PixA_REF
        
        if VERBOSE_LEVEL in [0, 1, 2]:
            logger.info('MeLOn CheckPoint: TRIGGER SFFT-SUBTRACTION!')

        Tsub_start = time.time()
        _tmp = GeneralSFFTSubtract.GSN(PixA_I=PixA_I, PixA_J=PixA_J, PixA_mI=PixA_mI, PixA_mJ=PixA_mJ, \
            SFFTConfig=SFFTConfig, ContamMask_I=None, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, logger=logger, VERBOSE_LEVEL=VERBOSE_LEVEL)

        Solution, PixA_DIFF = _tmp[:2]
        if VERBOSE_LEVEL in [1, 2]:
            _message = 'SFFT-SUBTRACTION TAKES [%.3f s]' %(time.time() - Tsub_start)
            logger.info('MeLOn Report: %s' %_message)
        
        # * Modifications on the difference image
        #   a) when REF is convolved, DIFF = SCI - Conv(REF)
        #      PSF(DIFF) is coincident with PSF(SCI), transients on SCI are positive signal in DIFF
        #   b) when SCI is convolved, DIFF = Conv(SCI) - REF
        #      PSF(DIFF) is coincident with PSF(REF), transients on SCI are still positive signal in DIFF

        if NaNmask_U is not None:
            # ** Mask Union-NaN region
            PixA_DIFF[NaNmask_U] = np.nan
        
        if ConvdSide == 'SCI': 
            # ** Flip difference when science is convolved
            PixA_DIFF = -PixA_DIFF
        
        # * Save difference image
        if FITS_DIFF is not None:
            with fits.open(FITS_SCI) as hdl:
                hdl[0].data[:, :] = PixA_DIFF.T
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
                hdl.writeto(FITS_DIFF, overwrite=True)
                
                logger.info('Wrote difference image to file {}'.format(FITS_DIFF))
        
    
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
            
            logger.info('Wrote solution array to file {}'.format(FITS_Solution))
        
        return Solution, PixA_DIFF



class Read_SFFTSolution:

    def __init__(self):

        """
        # Notes on the representation of a spatially-varying kernel (SVK)
        # 
        # (1) SFFT Format: 
        #     SVK_xy = sum_ab (Ac_xyab * K_ab)
        #            = sum_ab (Ac_xyab * K_ab) | KERNEL: (a, b) != (0, 0)
        #              + Ac_xy00 * K_00        | SCALING: (a, b) == (0, 0)
        #
        #     > SFFT can have separate KERNEL / SCALING in Polynomial / B-Spline form.
        #       Polynomial: Ac_xyab = sum_ij (ac_ijab * x^i * y^j)
        #       B-Spline:   Ac_xyab = sum_ij (ac_ijab * BSP_i(x) * BSP_j(y))
        #
        #     > Use the following two dictionaries to store the parameters
        #       KERNEL: SfftKerDict[(i, j)][a, b] = ac_ijab, where (a, b) != (0, 0)
        #       SCALING: SfftScaDict[(i, j)] = ac_ij00
        #
        #     NOTE: K_ab is modified delta basis (see the SFFT paper).
        #           Ac (ac) is the symbol A (a) with circle hat in SFFT paper, note that ac = a/(N0*N1).
        #           BSP_i(x) is the i-th B-Spline base function along X
        #           BSP_j(x) is the j-th B-Spline base function along Y
        #
        #     NOTE: For "ENTANGLED" SCALING or "SEPARATE-CONSTANT" SCALING
        #           one can solely use SfftKerDict to represent SVK by allowing (a, b) == (0, 0)
        #           no additional variation form need to be involved.
        #    
        #     NOTE: Although (x, y) are the integer indices of a pixel in SFFT paper,
        #           we actually use a scaled image coordinate (i.e., ScaledFortranCoor) of
        #           pixel center in our implementation. Note that above equations allow for 
        #           all possible image coordinates (not just pixel center).
        #
        # (2) Standard Format:
        #     SVK_xy = sum_ab (B_xyab * D_ab)
        #
        #     > Standard have entangled KERNEL & SCALING
        #       Polynomial: B_xyab = sum_ij (b_ijab * x^i * y^j)
        #       B-Spline:   B_xyab = sum_ij (b_ijab * BSP_i(x) * BSP_j(y))
        #     
        #     > Use the dictionary to store the parameters
        #       StandardKerDict[(i, j)][a, b] = b_ijab
        #
        #     NOTE: D_ab is standard Cartesian delta basis. Likewise, 
        #           (x, y) is ScaledFortranCoor that allows for all possible image coordinates.
        #
        # P.S. Coordinate Example: Given Pixel Location at (row=3, column=5) in an image of size (1024, 2048)
        #      FortranCoor = (4.0, 6.0) | ScaledFortranCoor = (4.0/1024, 6.0/2048)
        #
        # P.S. In principle, a and b are in the range of [-w0, w0 + 1) and [-w1 , w1+ 1), respectively.
        #      However, Sfft_dict[(i, j)] or Standard_dict[(i, j)] has to be a matrix with 
        #      non-negative indices starting from 0. In practice, the indices were tweaked 
        #      as [a, b] > [a + w0, b + w1].
        #
        """

        pass
    
    def FromArray(self, Solution, KerSpType, N0, N1, DK, L0, L1, Fi, Fj, Fpq, \
        SEPARATE_SCALING, ScaSpType, DS, ScaFi, ScaFj):

        # Remarks on SCALING_MODE
        # SEPARATE_SCALING & ScaSpDegree >>>      SCALING_MODE
        #        N         &     any     >>>       'ENTANGLED'
        #        Y         &      0      >>>   'SEPARATE-CONSTANT'
        #        Y         &     > 0     >>>   'SEPARATE-VARYING'

        SCALING_MODE = None
        if not SEPARATE_SCALING:
            SCALING_MODE = 'ENTANGLED'
        elif DS == 0:
            SCALING_MODE = 'SEPARATE-CONSTANT'
        else: SCALING_MODE = 'SEPARATE-VARYING'
        assert SCALING_MODE is not None

        Fab = L0*L1
        w0, w1 = (L0-1)//2, (L1-1)//2

        REF_ab = np.array([
            (a_pos - w0, b_pos - w1) \
            for a_pos in range(L0) for b_pos in range(L1)
        ]).astype(int)

        if KerSpType == 'Polynomial':
            Fij = ((DK+1)*(DK+2))//2
            REF_ij = np.array([
                (i, j) \
                for i in range(DK+1) for j in range(DK+1-i)
            ]).astype(int)
        
        if KerSpType == 'B-Spline':
            Fij = Fi*Fj
            REF_ij = np.array([
                (i, j) \
                for i in range(Fi) for j in range(Fj)
            ]).astype(int)
        
        Fijab = Fij*Fab
        SREF_ijab = np.array([
            (ij, ab) \
            for ij in range(Fij) for ab in range(Fab)
        ]).astype(int)

        if SCALING_MODE == 'SEPARATE-VARYING':

            if ScaSpType == 'Polynomial':
                ScaFij = ((DS+1)*(DS+2))//2
                ScaREF_ij = np.array(
                    [(i, j) for i in range(DS+1) for j in range(DS+1-i)] \
                    + [(-1, -1)] * (Fij - ScaFij)
                ).astype(int)

            if ScaSpType == 'B-Spline':
                ScaFij = ScaFi*ScaFj
                ScaREF_ij = np.array(
                    [(i, j) for i in range(ScaFi) for j in range(ScaFj)] \
                    + [(-1, -1)] * (Fij - ScaFij)
                ).astype(int)

        a_ijab = Solution[:-Fpq]      # drop differential background
        ac_ijab = a_ijab / (N0*N1)    # convert a to ac

        SfftKerDict = {}
        if KerSpType == 'Polynomial':
            for i in range(DK+1):
                for j in range(DK+1-i):
                    SfftKerDict[(i, j)] = np.zeros((L0, L1)).astype(float)
        
        if KerSpType == 'B-Spline':
            for i in range(Fi):
                for j in range(Fj):
                    SfftKerDict[(i, j)] = np.zeros((L0, L1)).astype(float)
        
        SfftScaDict = None
        if SCALING_MODE == 'SEPARATE-VARYING':
            
            SfftScaDict = {}
            if KerSpType == 'Polynomial':
                for i in range(DS+1):
                    for j in range(DS+1-i):
                        SfftScaDict[(i, j)] = 0.0
            
            if KerSpType == 'B-Spline':
                for i in range(ScaFi):
                    for j in range(ScaFj):
                        SfftScaDict[(i, j)] = 0.0

        for idx in range(Fijab):
            ij, ab = SREF_ijab[idx]
            a, b = REF_ab[ab]
            i, j = REF_ij[ij]

            if SCALING_MODE in ['ENTANGLED', 'SEPARATE-CONSTANT']:
                SfftKerDict[(i, j)][a+w0, b+w1] = ac_ijab[idx]

            if SCALING_MODE == 'SEPARATE-VARYING':
                if a == 0 and b == 0:
                    SfftKerDict[(i, j)][a+w0, b+w1] = np.nan                    
                    i8, j8 = ScaREF_ij[ij]
                    if i8 != -1 and j8 != -1:
                        SfftScaDict[(i8, j8)] = ac_ijab[idx]
                else:
                    SfftKerDict[(i, j)][a+w0, b+w1] = ac_ijab[idx]
            
        return SfftKerDict, SfftScaDict

    def FromFITS(self, FITS_Solution):
        
        Solution = fits.getdata(FITS_Solution, ext=0)[0]
        phdr = fits.getheader(FITS_Solution, ext=0)
        
        KerSpType = phdr['KSPTYPE']
        N0, N1 = int(phdr['N0']), int(phdr['N1'])
        DK = int(phdr['DK'])
        
        L0, L1 = int(phdr['L0']), int(phdr['L1'])
        Fi, Fj = int(phdr['FI']), int(phdr['FJ'])
        Fpq = int(phdr['FPQ'])

        SEPARATE_SCALING = phdr['SEPSCA'] == 'True'
        ScaSpType, DS = None, None
        if SEPARATE_SCALING:
            ScaSpType = phdr['SSPTYPE']
            DS = int(phdr['SSPDEG'])

        ScaFi, ScaFj = None, None
        if SEPARATE_SCALING and DS > 0:
            ScaFi = int(phdr['SCAFI'])
            ScaFj = int(phdr['SCAFJ'])

        SfftKerDict, SfftScaDict = self.FromArray(Solution=Solution, KerSpType=KerSpType, \
            N0=N0, N1=N1, DK=DK, L0=L0, L1=L1, Fi=Fi, Fj=Fj, Fpq=Fpq, \
            SEPARATE_SCALING=SEPARATE_SCALING, ScaSpType=ScaSpType, DS=DS, ScaFi=ScaFi, ScaFj=ScaFj)

        return SfftKerDict, SfftScaDict



# class ConvKernel_Convertion:

#     """
#     # * Remarks on Convolution Theorem
#     #    NOTE on Symbols: L theoretically can be even / odd, recall w = (L-1)//2 & w' = L//2 and w+w' = L-1
#     #    a. To make use of FFT with convolution theorem, it is necessary to convert convolution-kernel 
#     #        with relatively small size by Circular-Shift & tail-Zero-padding (CSZ) to align the Image-Size.
#     #    b. If we have an image elementwise-multiplication FI * FK, where FI and FK are conjugate symmetric,
#     #        Now we want to get its convolution counterpart in real space, we can calculate the convolution kernel by iCSZ on IDFT(FK).
#     #        In this process, we have to check the lost weight by size truncation, if the lost weight is basically inappreciable, 
#     #        the equivalent convolution is a good approximation.
#     #
#     """

#     def CSZ(ConvKernel, N0, N1):
#         L0, L1 = ConvKernel.shape  
#         w0, w1 = (L0-1) // 2, (L1-1) // 2      
#         pd0, pd1 = N0 - L0, N1 - L1
#         TailZP = np.lib.pad(ConvKernel, ((0, pd0), (0, pd1)), 'constant', constant_values=(0, 0))    # Tail-Zero-Padding
#         KIMG_CSZ = np.roll(np.roll(TailZP, -w0, axis=0), -w1, axis=1)    # Circular-Shift
#         return KIMG_CSZ

#     def iCSZ(KIMG, L0, L1):
#         N0, N1 = KIMG.shape
#         w0, w1 = (L0-1) // 2, (L1-1) // 2
#         KIMG_iCSZ = np.roll(np.roll(KIMG, w1, axis=1), w0, axis=0)    # inverse Circular-Shift
#         ConvKernel = KIMG_iCSZ[:L0, :L1]    # Tail-Truncation 
#         lost_weight = 1.0 - np.sum(np.abs(ConvKernel)) / np.sum(np.abs(KIMG_iCSZ))
#         return ConvKernel, lost_weight




class ConvKernel_Convertion:

    """
    # * Remarks on Convolution Theorem
    #    NOTE on Symbols: L theoretically can be even / odd, recall w = (L-1)//2 & w' = L//2 and w+w' = L-1
    #    a. To make use of FFT with convolution theorem, it is necessary to convert convolution-kernel 
    #        with relatively small size by Circular-Shift & tail-Zero-padding (CSZ) to align the Image-Size.
    #    b. If we have an image elementwise-multiplication FI * FK, where FI and FK are conjugate symmetric,
    #        Now we want to get its convolution counterpart in real space, we can calculate the convolution kernel by iCSZ on IDFT(FK).
    #        In this process, we have to check the lost weight by size truncation, if the lost weight is basically inappreciable, 
    #        the equivalent convolution is a good approximation.
    #
    """

    def CSZ(ConvKernel, N0, N1):
        L0, L1 = ConvKernel.shape  
        w0, w1 = (L0-1) // 2, (L1-1) // 2      
        pd0, pd1 = N0 - L0, N1 - L1
        TailZP = np.lib.pad(ConvKernel, ((0, pd0), (0, pd1)), 'constant', constant_values=(0, 0))    # Tail-Zero-Padding
        KIMG_CSZ = np.roll(np.roll(TailZP, -w0, axis=0), -w1, axis=1)    # Circular-Shift
        return KIMG_CSZ

    def iCSZ(KIMG, L0, L1):
        N0, N1 = KIMG.shape
        w0, w1 = (L0-1) // 2, (L1-1) // 2
        KIMG_iCSZ = np.roll(np.roll(KIMG, w1, axis=1), w0, axis=0)    # inverse Circular-Shift
        ConvKernel = KIMG_iCSZ[:L0, :L1]    # Tail-Truncation 
        lost_weight = 1.0 - np.sum(np.abs(ConvKernel)) / np.sum(np.abs(KIMG_iCSZ))
        return ConvKernel, lost_weight


class BSpline_DeCorrelation:
    @staticmethod
    def BDC(MK_JLst, SkySig_JLst, MK_ILst=[], SkySig_ILst=[], MK_Fin=None, \
        KERatio=2.0, DENO_CLIP_RATIO=100000.0, logger=None, VERBOSE_LEVEL=2):
        
        """
        # * Remarks on Input
        #   i. Image-Stacking Mode: NumI = 0
        #      MK_Fin will not work, NumJ >= 2 and NOT all J's kernel are None
        #   ii. Image-Subtraction Mode: NumI & NumJ >= 1
        #       NOT all (I / J / Fin) kernel are None
        #
        # * Remarks on difference flip
        #   D = REF - SCI * K
        #   fD = SCI * K - REF = fREF - fSCI * K
        #   NOTE: fD and D have consistent decorrelation kernel
        #         as Var(REF) = Var(fREF) and Var(SCI) = Var(fSCI)
        #
        # * Remarks on DeCorrelation Kernel Size
        #   The DeCorrelation Kernel is derived in Fourier Space, but it is not proper to directly 
        #   perform DeCorrelation in Fourier Space (equivalently, perform a convolution with Kernel-Size = Image-Size).
        #   a. convolution (and resampling) process has very local effect, it is unnecesseary to use a large decorrelation kernel.
        #   b. if we use a large decorrelation kernel, you will find only very few engery distributed at outskirt regions, 
        #      however, the low weight can degrade the decorrelation convolution by the remote saturated pixels.
        #
        """
        
        # fft_func = pyfftw.interfaces.dask_fft.fft2
        # ifft_func = pyfftw.interfaces.dask_fft.ifft2
        # fft_func = np.fft.fft2
        # ifft_func = np.fft.ifft2
        fft_func = nfft.fft2
        ifft_func = nfft.ifft2

        NumI, NumJ = len(MK_ILst), len(MK_JLst)
        if NumI == 0: 
            Mode = 'Image-Stacking'
            if NumJ < 2: 
                _error_message = 'Image-Stacking Mode requires at least 2 J-images!'
                raise logger.error('MeLOn ERROR: %s' %_error_message)
            if np.sum([MKj is not None for MKj in MK_JLst]) == 0:
                _error_message = 'Image-Stacking Mode requires at least 1 not-None J-kernel!'
                raise logger.error('MeLOn ERROR: %s' %_error_message)

        if NumI >= 1:
            Mode = 'Image-Subtraction'
            if NumJ == 0: 
                _error_message = 'Image-Subtraction Mode requires at least 1 I-image & 1 J-image!'
                raise logger.error('MeLOn ERROR: %s' %_error_message)
            if np.sum([MK is not None for MK in MK_JLst+MK_ILst+[MK_Fin]]) == 0:
                _error_message = 'Image-Subtraction Mode requires at least 1 not-None J/I/Fin-kernel!'
                raise logger.error('MeLOn ERROR: %s' %_error_message)
        
        MK_Queue = MK_JLst.copy()
        if Mode == 'Image-Subtraction': MK_Queue += [MK_Fin] + MK_ILst
        L0_KDeCo = int(round(KERatio * np.max([MK.shape[0] for MK in MK_Queue if MK is not None])))
        L1_KDeCo = int(round(KERatio * np.max([MK.shape[1] for MK in MK_Queue if MK is not None])))
        if L0_KDeCo%2 == 0: L0_KDeCo += 1
        if L1_KDeCo%2 == 0: L1_KDeCo += 1

        if VERBOSE_LEVEL in [1, 2]:
            _message = 'DeCorrelation Kernel with size [%d, %d]' %(L0_KDeCo, L1_KDeCo)
            logger.info('MeLOn CheckPoint: %s' %_message)

        # trivial image size, just typically larger than the kernel size.
        N0 = 2 ** (math.ceil(np.log2(np.max([MK.shape[0] for MK in MK_Queue if MK is not None])))+1)
        N1 = 2 ** (math.ceil(np.log2(np.max([MK.shape[1] for MK in MK_Queue if MK is not None])))+1)
        
        # construct the DeNonimator (a real-positive map) in Fourier Space
        uMK = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).astype(float)
        def Get_JTerm(MKj, skysig):
            if MKj is not None: KIMG_CSZ = ConvKernel_Convertion.CSZ(MKj, N0, N1)
            if MKj is None: KIMG_CSZ = ConvKernel_Convertion.CSZ(uMK, N0, N1)
            kft = fft_func(KIMG_CSZ)
            kft2 = (np.conj(kft) * kft).real
            term = (skysig**2 * kft2) / NumJ**2
            return term
        
        if MK_Fin is not None: KIMG_CSZ = ConvKernel_Convertion.CSZ(MK_Fin, N0, N1)
        if MK_Fin is None: KIMG_CSZ = ConvKernel_Convertion.CSZ(uMK, N0, N1)
        kft = fft_func(KIMG_CSZ)
        kft2_Fin = (np.conj(kft) * kft).real

        def Get_ITerm(MKi, skysig):
            if MKi is not None: KIMG_CSZ = ConvKernel_Convertion.CSZ(MKi, N0, N1)
            if MKi is None: KIMG_CSZ = ConvKernel_Convertion.CSZ(uMK, N0, N1)
            kft = fft_func(KIMG_CSZ)
            kft2 = (np.conj(kft) * kft).real
            term = (skysig**2 * kft2 * kft2_Fin) / NumI**2 
            return term
        
        DeNo = 0.0
        for MKj, skysig in zip(MK_JLst, SkySig_JLst):
            DeNo += Get_JTerm(MKj, skysig)
        if Mode == 'Image-Subtraction':
            for MKi, skysig in zip(MK_ILst, SkySig_ILst):
                DeNo += Get_ITerm(MKi, skysig)

        # clipping to avoid too small denominator
        if VERBOSE_LEVEL in [2]:
            logger.info('MeLOn CheckPoint: Initial Max/Min [%.1f] in Denominator Map' \
                %(np.max(DeNo)/np.min(DeNo)))
        
        DENO_CLIP_THRESH = np.max(DeNo)/DENO_CLIP_RATIO
        DENO_CLIP_MASK = DeNo < DENO_CLIP_THRESH
        DeNo[DENO_CLIP_MASK] = DENO_CLIP_THRESH

        if VERBOSE_LEVEL in [2]:
            logger.info('MeLOn CheckPoint: DENOMINATOR CLIPPING TWEAKED [%s] PIXELS' \
                %('{:.2%}'.format(np.sum(DENO_CLIP_MASK)/(N0*N1))))

        FDeCo = np.sqrt(1.0 / DeNo)        # real & conjugate-symmetric
        DeCo = ifft_func(FDeCo).real    # no imaginary part
        KDeCo, lost_weight = ConvKernel_Convertion.iCSZ(DeCo, L0_KDeCo, L1_KDeCo)
        KDeCo = KDeCo / np.sum(KDeCo)      # rescale to have Unit kernel sum

        if VERBOSE_LEVEL in [1, 2]:
            _message = 'Tail-Truncation Lost-Weight [%.4f %s] (Absolute Percentage Error)' %(lost_weight*100, '%')
            logger.info('MeLOn CheckPoint: %s' %_message)

        return KDeCo
    
    
    
    
    
class BSpline_GridConvolve:
    def __init__(self, PixA_obj, AllocatedL, KerStack, \
        nan_fill_value=0.0, use_fft=False, normalize_kernel=True):

        """
        # * Remarks on Grid-wise Spave-Varying Convolution
        #   Allocated LabelMap has same shape with PixA_obj to show the image segmentation (Compact-Box!)
        #   Kernel-Stack gives corresponding convolution kernel for each label
        #   For each segment, we would extract Esegment according to the label with a extended boundary
        #   Perform convolution and then send the values within this segment to output image.
        
        # * A Typcical Example of AllocatedL & KerStack
        
        TiHW = 10
        N0, N1 = 1024, 1024

        lab = 0
        TiN = 2*TiHW+1
        XY_TiC = []
        AllocatedL = np.zeros((N0, N1), dtype=int)
        for xs in np.arange(0, N0, TiN):
            xe = np.min([xs+TiN, N0])
            for ys in np.arange(0, N1, TiN):
                ye = np.min([ys+TiN, N1])
                AllocatedL[xs: xe, ys: ye] = lab
                x_q = 0.5 + xs + (xe - xs)/2.0   # tile-center (x)
                y_q = 0.5 + ys + (ye - ys)/2.0   # tile-center (y)
                XY_TiC.append([x_q, y_q])
                lab += 1
        XY_TiC = np.array(XY_TiC)
        
        """

        PixA_in = PixA_obj.copy()
        PixA_in[np.isnan(PixA_in)] = nan_fill_value
        self.PixA_in = PixA_in
        
        self.AllocatedL = AllocatedL
        self.KerStack = KerStack
        self.use_fft = use_fft
        self.normalize_kernel = normalize_kernel


    def GSVC_CPU(self, nproc=32):
        
        N0, N1 = self.PixA_in.shape
        Nseg, L0, L1 = self.KerStack.shape
        w0 = int((L0-1)/2)
        w1 = int((L1-1)/2)
        IBx, IBy = w0+1, w1+1
        
        def func_conv(idx):
            Ker = self.KerStack[idx]
            lX, lY = np.where(self.AllocatedL == idx)
            xs, xe = lX.min(), lX.max()
            ys, ye = lY.min(), lY.max()

            xEs, xEe = max([0, xs-IBx]), min([N0-1, xe+IBx])
            yEs, yEe = max([0, ys-IBy]), min([N1-1, ye+IBy])
            PixA_Emini = self.PixA_in[xEs: xEe+1, yEs: yEe+1]
            
            xyrg = xs, xe+1, ys, ye+1
            if np.nansum(Ker) == 0.:
                return xyrg, 0.

            if not self.use_fft:
                _CPixA = convolve(PixA_Emini, Ker, boundary='fill', \
                    fill_value=0.0, normalize_kernel=self.normalize_kernel)
            else:
                _CPixA = convolve_fft(PixA_Emini, Ker, boundary='fill', \
                    fill_value=0.0, normalize_kernel=self.normalize_kernel)
            
            fragment = _CPixA[xs-xEs: (xs-xEs)+(xe+1-xs), ys-yEs: (ys-yEs)+(ye+1-ys)]
            return xyrg, fragment

        taskid_lst = np.arange(Nseg)
        # mydict = Multi_Proc.MP(taskid_lst=taskid_lst, func=func_conv, nproc=nproc, mode='mp')

        with WorkerPool(n_jobs=nproc) as pool:
            output = pool.map(func_conv, range(Nseg), progress_bar=True, progress_bar_style='rich')
            

        PixA_GRID_SVConv = np.zeros((N0, N1)).astype(float)
        for idx in taskid_lst:
            # xyrg, fragment = mydict[idx]
            xyrg, fragment = output[idx]
            PixA_GRID_SVConv[xyrg[0]: xyrg[1], xyrg[2]: xyrg[3]] = fragment
        
        return PixA_GRID_SVConv
    

    def GSVC_GPU(self, CUDA_DEVICE='0', CLEAN_GPU_MEMORY=False, nproc=32):
        
        import cupy as cp
        from cupyx.scipy.signal import fftconvolve, convolve2d
        device = cp.cuda.Device(int(CUDA_DEVICE))
        device.use()

        if CLEAN_GPU_MEMORY:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()

        N0, N1 = self.PixA_in.shape
        Nseg, L0, L1 = self.KerStack.shape
        w0 = int((L0-1)/2)
        w1 = int((L1-1)/2)
        IBx, IBy = w0+1, w1+1

        PixA_in_GPU = cp.array(self.PixA_in, dtype=np.float64)
        if not self.normalize_kernel:
            KerStack_GPU = cp.array(self.KerStack, dtype=np.float64)
        else:
            sums = np.sum(self.KerStack, axis=(1,2))
            KerStack_GPU = cp.array(self.KerStack / sums[:, np.newaxis, np.newaxis], dtype=np.float64)

        def func_getIdx(idx):
            lX, lY = np.where(self.AllocatedL == idx)
            xs, xe = lX.min(), lX.max()
            ys, ye = lY.min(), lY.max()
            xEs, xEe = max([0, xs-IBx]), min([N0-1, xe+IBx])
            yEs, yEe = max([0, ys-IBy]), min([N1-1, ye+IBy])
            return (xs, xe, ys, ye, xEs, xEe, yEs, yEe)

        taskid_lst = np.arange(Nseg)
        IdxDICT = Multi_Proc.MP(taskid_lst=taskid_lst, func=func_getIdx, nproc=nproc, mode='mp')

        PixA_GRID_SVConv_GPU = cp.zeros((N0, N1), dtype=np.float64)
        if self.use_fft:
            for idx in range(Nseg):
                Ker_GPU = KerStack_GPU[idx]
                xs, xe, ys, ye, xEs, xEe, yEs, yEe = IdxDICT[idx]
                PixA_Emini_GPU = PixA_in_GPU[xEs: xEe+1, yEs: yEe+1]
                _CPixA_GPU = fftconvolve(PixA_Emini_GPU, Ker_GPU, mode='same')  # as if boundary filled by 0.0
                fragment_GPU = _CPixA_GPU[xs-xEs: (xs-xEs)+(xe+1-xs), ys-yEs: (ys-yEs)+(ye+1-ys)]
                PixA_GRID_SVConv_GPU[xs: xe+1, ys: ye+1] = fragment_GPU
                
        if not self.use_fft:
            for idx in range(Nseg):
                Ker_GPU = KerStack_GPU[idx]
                xs, xe, ys, ye, xEs, xEe, yEs, yEe = IdxDICT[idx]
                PixA_Emini_GPU = PixA_in_GPU[xEs: xEe+1, yEs: yEe+1]
                _CPixA_GPU = convolve2d(PixA_Emini_GPU, Ker_GPU, mode='same', boundary='fill', fillvalue=0.0)
                fragment_GPU = _CPixA_GPU[xs-xEs: (xs-xEs)+(xe+1-xs), ys-yEs: (ys-yEs)+(ye+1-ys)]
                PixA_GRID_SVConv_GPU[xs: xe+1, ys: ye+1] = fragment_GPU
        PixA_GRID_SVConv = cp.asnumpy(PixA_GRID_SVConv_GPU)

        return PixA_GRID_SVConv
    


class BSpline_MatchingKernel:

    def __init__(self, XY_q, VERBOSE_LEVEL=2):
        self.XY_q = XY_q
        self.VERBOSE_LEVEL = VERBOSE_LEVEL

    def FromArray(self, Solution, KerSpType, KerIntKnotX, KerIntKnotY, N0, N1, DK, L0, L1, Fi, Fj, Fpq, \
        SEPARATE_SCALING, ScaSpType, ScaIntKnotX, ScaIntKnotY, DS, ScaFi, ScaFj):

        # convert requested coordinates
        sXY_q = self.XY_q.astype(float)   # input requested coordinates in FortranCoor [global convention]
        sXY_q[:, 0] /= N0                 # convert to ScaledFortranCoor [local convention]
        sXY_q[:, 1] /= N1                 # convert to ScaledFortranCoor [local convention]

        # read SFFTSolution as dictionaries
        SfftKerDict, SfftScaDict = Read_SFFTSolution().FromArray(Solution=Solution, KerSpType=KerSpType, \
            N0=N0, N1=N1, DK=DK, L0=L0, L1=L1, Fi=Fi, Fj=Fj, Fpq=Fpq, \
            SEPARATE_SCALING=SEPARATE_SCALING, ScaSpType=ScaSpType, DS=DS, ScaFi=ScaFi, ScaFj=ScaFj)

        # realize kernels at given coordinates        
        def Create_BSplineBasis_Req(N, IntKnot, BSplineDegree, ReqCoord):
            BSplineBasis_Req = []
            Knot = np.concatenate(([0.5]*(BSplineDegree+1), IntKnot, [N+0.5]*(BSplineDegree+1)))/N
            Nc = len(IntKnot) + BSplineDegree + 1    # number of control points/coeffcients
            for idx in range(Nc):
                Coeff = (np.arange(Nc) == idx).astype(float)
                BaseFunc = BSpline(t=Knot, c=Coeff, k=BSplineDegree, extrapolate=False)
                BSplineBasis_Req.append(BaseFunc(ReqCoord))
            BSplineBasis_Req = np.array(BSplineBasis_Req)
            return BSplineBasis_Req

        w0, w1 = (L0-1)//2, (L1-1)//2

        if KerSpType == 'Polynomial':

            KerBASE = np.array([
                sXY_q[:, 0]**i * sXY_q[:, 1]**j \
                for i in range(DK+1) for j in range(DK+1-i)
            ])  # (Fij, NPOINT)

            KerCOEFF = np.array([
                SfftKerDict[(i, j)] \
                for i in range(DK+1) for j in range(DK+1-i)
            ])  # (Fij, KerNX, KerNY)

            KerStack = np.tensordot(KerBASE, KerCOEFF, (0, 0))   # (NPOINT, KerNX, KerNY)
        
        if KerSpType == 'B-Spline':

            KerSplBasisX_q = Create_BSplineBasis_Req(N=N0, IntKnot=KerIntKnotX, BSplineDegree=DK, ReqCoord=sXY_q[:, 0])
            KerSplBasisY_q = Create_BSplineBasis_Req(N=N1, IntKnot=KerIntKnotY, BSplineDegree=DK, ReqCoord=sXY_q[:, 1])

            KerBASE = np.array([
                KerSplBasisX_q[i] * KerSplBasisY_q[j] \
                for i in range(Fi) for j in range(Fj)
            ])  # (Fij, NPOINT)

            KerCOEFF = np.array([
                SfftKerDict[(i, j)] \
                for i in range(Fi) for j in range(Fj)
            ])  # (Fij, KerNX, KerNY)

            KerStack = np.tensordot(KerBASE, KerCOEFF, (0, 0))   # (NPOINT, KerNX, KerNY)
            
        # convert specific coefficients to kernel pixels (Note that sfft uses modified delta basis)
        if SfftScaDict is None:
            KerCENT = KerStack[:, w0, w1].copy()
            KerCENT -= np.sum(KerStack, axis=(1,2)) - KerStack[:, w0, w1]
            KerStack[:, w0, w1] = KerCENT   # UPDATE

        if SfftScaDict is not None:

            if ScaSpType == 'Polynomial':

                ScaBASE = np.array([
                    sXY_q[:, 0]**i * sXY_q[:, 1]**j \
                    for i in range(DS+1) for j in range(DS+1-i)
                ])  # (ScaFij, NPOINT)

                ScaCOEFF = np.array([
                    SfftScaDict[(i, j)] \
                    for i in range(DS+1) for j in range(DS+1-i)
                ])  # (ScaFij)

                KerCENT = np.matmul(ScaCOEFF.reshape((1, -1)), ScaBASE)[0]   # (NPOINT)
                KerCENT -= np.nansum(KerStack, axis=(1,2))
                KerStack[:, w0, w1] = KerCENT  # UPDATE
            
            if ScaSpType == 'B-Spline':
            
                ScaSplBasisX_q = Create_BSplineBasis_Req(N=N0, IntKnot=ScaIntKnotX, BSplineDegree=DS, ReqCoord=sXY_q[:, 0])
                ScaSplBasisY_q = Create_BSplineBasis_Req(N=N1, IntKnot=ScaIntKnotY, BSplineDegree=DS, ReqCoord=sXY_q[:, 1])

                ScaBASE = np.array([
                    ScaSplBasisX_q[i] * ScaSplBasisY_q[j] \
                    for i in range(ScaFi) for j in range(ScaFj)
                ])  # (ScaFij, NPOINT)

                ScaCOEFF = np.array([
                    SfftScaDict[(i, j)] \
                    for i in range(ScaFi) for j in range(ScaFj)
                ])  # (ScaFij)

                KerCENT = np.matmul(ScaCOEFF.reshape((1, -1)), ScaBASE)[0]   # (NPOINT)
                KerCENT -= np.nansum(KerStack, axis=(1,2))
                KerStack[:, w0, w1] = KerCENT  # UPDATE
        
        return KerStack
    
    def FromFITS(self, FITS_Solution, logger):

        phdr = fits.getheader(FITS_Solution, ext=0)
        KerHW = phdr['KERHW']
        KerSpType = phdr['KSPTYPE']
        KerIntKnotX = [phdr['KIKX%d' %i] for i in range(phdr['NKIKX'])]
        KerIntKnotY = [phdr['KIKY%d' %i] for i in range(phdr['NKIKY'])]

        N0, N1 = int(phdr['N0']), int(phdr['N1'])
        DK = int(phdr['DK'])
        
        L0, L1 = int(phdr['L0']), int(phdr['L1'])
        Fi, Fj = int(phdr['FI']), int(phdr['FJ'])
        Fpq = int(phdr['FPQ'])

        SEPARATE_SCALING = phdr['SEPSCA'] == 'True'
        ScaSpType, DS = None, None
        ScaIntKnotX, ScaIntKnotY = None, None
        if SEPARATE_SCALING:
            ScaSpType = phdr['SSPTYPE']
            DS = int(phdr['SSPDEG'])
            ScaIntKnotX = [phdr['SIKX%d' %i] for i in range(phdr['NSIKX'])]
            ScaIntKnotY = [phdr['SIKY%d' %i] for i in range(phdr['NSIKY'])]

        ScaFi, ScaFj = None, None
        if SEPARATE_SCALING and DS > 0:
            ScaFi = int(phdr['SCAFI'])
            ScaFj = int(phdr['SCAFJ'])

        if self.VERBOSE_LEVEL in [1, 2]:
            logger.info('--//--//--//--//-- SFFT CONFIGURATION --//--//--//--//-- ')

            if KerSpType == 'Polynomial':
                logger.info('---//--- Polynomial Kernel | KerSpDegree %d | KerHW %d ---//---' %(DK, KerHW))
                if not SEPARATE_SCALING:
                    logger.info('---//--- [ENTANGLED] Polynomial Scaling | KerSpDegree %d ---//---' %DK)

            if KerSpType == 'B-Spline':
                logger.info('---//--- B-Spline Kernel | Internal Knots %d,%d | KerSpDegree %d | KerHW %d ---//---' \
                      %(len(KerIntKnotX), len(KerIntKnotY), DK, KerHW))
                if not SEPARATE_SCALING: 
                    logger.info('---//--- [ENTANGLED] B-Spline Scaling | Internal Knots %d,%d | KerSpDegree %d ---//---' \
                          %(len(KerIntKnotX), len(KerIntKnotY), DK))
            
            if SEPARATE_SCALING:
                if ScaSpType == 'Polynomial':
                    logger.info('---//--- [SEPARATE] Polynomial Scaling | ScaSpDegree %d ---//---' %DS)
                
                if ScaSpType == 'B-Spline':
                    logger.info('---//--- [SEPARATE] B-Spline Scaling | Internal Knots %d,%d | ScaSpDegree %d ---//---' \
                          %(len(ScaIntKnotX), len(ScaIntKnotY), DS))

        Solution = fits.getdata(FITS_Solution, ext=0)[0]
        KerStack = self.FromArray(Solution=Solution, KerSpType=KerSpType, \
            KerIntKnotX=KerIntKnotX, KerIntKnotY=KerIntKnotY, N0=N0, N1=N1, \
            DK=DK, L0=L0, L1=L1, Fi=Fi, Fj=Fj, Fpq=Fpq, SEPARATE_SCALING=SEPARATE_SCALING, \
            ScaSpType=ScaSpType, ScaIntKnotX=ScaIntKnotX, ScaIntKnotY=ScaIntKnotY, \
            DS=DS, ScaFi=ScaFi, ScaFj=ScaFj)
        
        return KerStack