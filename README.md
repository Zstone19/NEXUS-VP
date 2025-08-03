# The NEXUS Variability Pipeline (NVP)

The NEXUS Variability Pipeline (NVP) is designed to obtain difference images from NEXUS mosaics, identify sources of variability, and extract said variability reliably. NVP utilizes difference images obtained via direct subtraction (due to the high quality and pointing of the images from JWST) and more complex difference imaging techniques such as SFFT and ZOGY.

# Features

## SFFT Subtraction
The mosaics obtained from the NEXUS program are particularly suited for the use of SFFT, which has proven to be useful for JWST NIRCam images. The NVP has a number of routines used to perform SFFT pipeline subtraction once the input mosaics are aligned:
- Sky Subtraction: The NVP has the ability to subtract a constant sky value from each of the two input images
- GPU/CPU use: The NVP has both CPU and CPU-accelerated versions of SFFT, useful for large mosaics or complex subtraction parameterizations
- Cross-Convolution: To ensure better subtraction, SFFT matches the PSFs of the two input images. To make this PSF matching smoother, the NVP allows for "cross-convolution" of the input images before analysis, where the REF image is convolved with the SCI PSF, and vice-versa.
- Source masking: SFFT requires a mask to be input, whose pixels are safe to perform difference imaging on, for example where the pixels are not saturated. The NVP contains methods to produce said masks, and exclude bright, saturated sources (mainly stars).
- Decorrelation: The NVP has implemented the decorrelation algorithm in SFFT to remove correlations induced by the cross-convolution and SFFT subtraction.
- SNR Maps: The NVP can also calculate the signal-to-noise ratio map of the entire image
- SNR statistics: The NVP will output auxiliary information about the distribution of SNR across the decorrelated difference image
- Cutouts: The NVP may split the input REF and SCI images into NxN pixel cutouts, where the size of the cutouts is specified by the user. Pre-processing steps can be performed before or after the image is split into cutouts. The NVP also allows the user to remove any empty space from each cutout. This is usually necessary, as SFFT subtraction can be computationally expensive.

## Source Extraction
- After using SExtractor on multiple bands in both the directly subtracted difference image and decorrelated SFFT difference image (in dual image mode), the NVP can then combine all SExtractor results into a single table.
- The NVP can use this output data table to make informative cutouts of the sources identified by SExtractor.
