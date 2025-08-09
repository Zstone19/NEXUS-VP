import sep
import warnings
import os
import subprocess
import shutil
import numpy as np
from astropy.io import fits
from scipy.stats import iqr
# version: Apr 22, 2024

# improved by Lauren Aldoroty (Duke Univ.)
__author__ = "Lei Hu <leihu@andrew.cmu.edu>"
__version__ = "v1.4"

class SEx_SkySubtract:
    @staticmethod
    def SSS(FITS_obj, FITS_skysub=None, FITS_sky=None, FITS_skyrms=None, mask_type='sextractor', SATUR_KEY='SATURATE', ESATUR_KEY='ESATUR', \
        BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=1.5, DETECT_MINAREA=5, DETECT_MAXAREA=0, \
        VERBOSE_LEVEL=2, MDIR=None, NCPU=1):

        """
        # Inputs & Outputs:

        -FITS_obj []                    # FITS file path of the input image 
 
        -FITS_skysub [None]             # FITS file path of the output sky-subtracted image

        -FITS_sky [None]                # FITS file path of the output sky image
        
        -FITS_skyrms [None]             # FITS file path of the output sky RMS image

        -ESATUR_KEY ['ESATUR']          # Keyword for the effective saturation level of sky-subtracted image
                                        # P.S. the value will be saved in the primary header of -FITS_skysub

        # Configurations for SExtractor:
        
        -SATUR_KEY ['SATURATE']         # SExtractor Parameter SATUR_KEY
                                        # i.e., keyword of the saturation level in the input image header

        -BACK_SIZE [64]                 # SExtractor Parameter BACK_SIZE

        -BACK_FILTERSIZE [3]            # SExtractor Parameter BACK_FILTERSIZE

        -DETECT_THRESH [1.5]            # SExtractor Parameter DETECT_THRESH

        -DETECT_MINAREA [5]             # SExtractor Parameter DETECT_MINAREA
        
        -DETECT_MAXAREA [0]             # SExtractor Parameter DETECT_MAXAREA

        # Miscellaneous
        
        -VERBOSE_LEVEL [2]              # The level of verbosity, can be [0, 1, 2]
                                        # 0/1/2: QUIET/NORMAL/FULL mode
        
        -MDIR [None]                    # Parent Directory for output files
                                        # PYSEx will generate a child directory with a random name under the paraent directory 
                                        # all output files are stored in the child directory

        # Returns:

            SKYDIP                      # The flux peak of the sky image (outliers rejected)

            SKYPEAK                     # The flux dip of the sky image (outliers rejected)
            
            PixA_skysub                 # Pixel Array of the sky-subtracted image 
            
            PixA_sky                    # Pixel Array of the sky image 
            
            PixA_skyrms                 # Pixel Array of the sky RMS image 

        """
                
        if mask_type == 'noisechisel':
            #Save temporary image
            mdir = os.path.dirname(FITS_obj) + '/temp/'
            os.makedirs(mdir, exist_ok=True)
            
            FITS_obj_temp = mdir + os.path.basename(FITS_obj)
            with fits.open(FITS_obj) as hdul:
                hdul[0].data[ hdul[0].data == 0. ] = np.nan
                hdul.writeto(FITS_obj_temp, overwrite=True)
            
            mask_fname = os.path.dirname(FITS_obj) + '/{}.noisechisel_mask.fits'.format(os.path.basename(FITS_obj).split('.')[0])

            nngb = 15
            mask_nonexist = True

            while mask_nonexist:          
                if nngb == 0:
                    print('NoiseChisel failed to generate mask. Using SExtractor.')
                    mask_type = 'sextractor'
                    mask_nonexist = False
                    continue

                try:
                    subprocess.run(['astnoisechisel', FITS_obj_temp, '-o', mask_fname, '-h', '0', '-N', str(NCPU), '--oneelempertile', '--rawoutput', '--minnumfalse', '1', '--interpnumngb={}'.format(nngb)], check=True, capture_output=True).stdout
                    mask_nonexist = False
                except:
                    print(nngb)
                    nngb -= 1
                    continue

            if nngb > 0:
                print('NoiseChisel mask generated: {}'.format(mask_fname))                
                seg = fits.getdata(mask_fname, ext=1).T
                DETECT_MASK = (seg > 0)
                
            os.remove(FITS_obj_temp)
            shutil.rmtree(mdir)



        if mask_type == 'sextractor':
            from sfft.utils.pyAstroMatic.PYSEx import PY_SEx

            # * Generate SExtractor OBJECT-MASK
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # NOTE: GAIN, SATURATE, ANALYSIS_THRESH, DEBLEND_MINCONT, BACKPHOTO_TYPE do not affect the detection mask.
                DETECT_MASK = PY_SEx.PS(FITS_obj=FITS_obj, SExParam=['X_IMAGE', 'Y_IMAGE'], GAIN_KEY='PHGAIN', SATUR_KEY=SATUR_KEY, \
                    BACK_TYPE='AUTO', BACK_SIZE=BACK_SIZE, BACK_FILTERSIZE=BACK_FILTERSIZE, DETECT_THRESH=DETECT_THRESH, \
                    ANALYSIS_THRESH=1.5, DETECT_MINAREA=DETECT_MINAREA, DETECT_MAXAREA=DETECT_MAXAREA, DEBLEND_MINCONT=0.005, \
                    BACKPHOTO_TYPE='GLOBAL', CHECKIMAGE_TYPE='OBJECTS', MDIR=MDIR, VERBOSE_LEVEL=VERBOSE_LEVEL)[1][0].astype(bool)
            
        
        # * Extract SExtractor SKY-MAP from the Unmasked Image
        PixA_obj = fits.getdata(FITS_obj, ext=0).T
        _PixA = PixA_obj.astype(np.float64, copy=True)    # default copy=True, just to emphasize
        _PixA[DETECT_MASK] = np.nan
        if not _PixA.flags['C_CONTIGUOUS']: _PixA = np.ascontiguousarray(_PixA)

        # NOTE: here we use faster sep package instead of SExtractor.
        sepbkg = sep.Background(_PixA, bw=BACK_SIZE, bh=BACK_SIZE, fw=BACK_FILTERSIZE, fh=BACK_FILTERSIZE)
        PixA_sky, PixA_skyrms = sepbkg.back(), sepbkg.rms()
        PixA_skysub = PixA_obj - PixA_sky

        # * Make simple statistics for the SKY-MAP
        Q1 = np.percentile(PixA_sky, 25)
        Q3 = np.percentile(PixA_sky, 75)
        IQR = iqr(PixA_sky)
        SKYDIP = Q1 - 1.5*IQR    # outlier rejected dip
        SKYPEAK = Q3 + 1.5*IQR   # outlier rejected peak
        
        if FITS_skysub is not None:
            with fits.open(FITS_obj) as hdl:
                hdl[0].header['SKYDIP'] = (SKYDIP, 'MeLOn: IQR-MINIMUM of SEx-SKY-MAP')
                hdl[0].header['SKYPEAK'] = (SKYPEAK, 'MeLOn: IQR-MAXIMUM of SEx-SKY-MAP')
                if SATUR_KEY in hdl[0].header:
                    ESATUR = float(hdl[0].header[SATUR_KEY]) - SKYPEAK    # use a conservative value
                    hdl[0].header[ESATUR_KEY] = (ESATUR, 'MeLOn: Effective SATURATE after SEx-SKY-SUB')
                hdl[0].data[:, :] = PixA_skysub.T
                hdl.writeto(FITS_skysub, overwrite=True)
        
        if FITS_sky is not None:
            with fits.open(FITS_obj) as hdl:
                hdl[0].header['SKYDIP'] = (SKYDIP, 'MeLOn: IQR-MINIMUM of SEx-SKY-MAP')
                hdl[0].header['SKYPEAK'] = (SKYPEAK, 'MeLOn: IQR-MAXIMUM of SEx-SKY-MAP')
                hdl[0].data[:, :] = PixA_sky.T
                hdl.writeto(FITS_sky, overwrite=True)
        
        if FITS_skyrms is not None:
            with fits.open(FITS_obj) as hdl:
                hdl[0].header['SKYDIP'] = (SKYDIP, 'MeLOn: IQR-MINIMUM of SEx-SKY-MAP')
                hdl[0].header['SKYPEAK'] = (SKYPEAK, 'MeLOn: IQR-MAXIMUM of SEx-SKY-MAP')
                hdl[0].data[:, :] = PixA_skyrms.T
                hdl.writeto(FITS_skyrms, overwrite=True)

        return SKYDIP, SKYPEAK, PixA_skysub, PixA_sky, PixA_skyrms