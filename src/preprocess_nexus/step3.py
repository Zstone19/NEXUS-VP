########################################################################################################################################################################
# Step 3

# 1. Run Image3Pipeline
#  - Combine all images from the same filter
#  - Skip SkyMatch
#  - Use tweakreg, outlier_detection, and resample
#  - Tie astrometry to Subraru HSC - HEROES
#  - Group exposures of the same module together, instead of the same visit
#  - Set maskpt=.5 in outlier_detection
#  - Set pixel_scale=.03" and pixfrac=.8 in resample

# 2. Perform background subtraction using the same method as in Step 2

# 3. Rescale weight/error maps
#  - Use ratio between the sigma from gaussian fitting the background-dominated pixels in the data and sigma from the error/weight map
#  - Different scaling factors for regions covered by different number of exposures 