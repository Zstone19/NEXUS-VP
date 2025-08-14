import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.visualization import ImageNormalize, ZScaleInterval, AsinhStretch, ManualInterval
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import numpy as np
from astropy.convolution import convolve_fft
from scipy import stats

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.wcs.utils import skycoord_to_pixel

import sys
sys.path.append('/home/stone28/projects/WebbDiffImg/NEXUS-VP/src/psf/')
from psf_utils import estimate_psf_fwhm


# Set matplotlib parameters
import matplotlib as mpl
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.direction'] = 'in'

mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.direction'] = 'in'

mpl.rcParams["figure.autolayout"] = False

mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.format'] = 'pdf'


def summary_plot(maindir, ref_name, sci_name, 
                 ref_title=None, sci_title=None,
                 show_fwhm=False, scalebar=True, show_stats=True,
                 cutout_center=None, cutout_size=None, npx_boundary=30,
                 skysub=False, conv_ref=False, conv_sci=False, show_sat_sources=False,
                 output_fname=None, show=False):
    """Make a summary plot of the SFFT outputs. Contains 8 panels:
    1. Reference Image (+PSF)
    2. Science Image (+PSF)
    3. Cross-Convolved Reference Image (+PSF)
    4. Cross-Convolved Science Image (+PSF)
    5. Difference Image (+ Matching Kernel)
    6. Decorrelated Difference Image (+PSF and Decorrelation Kernel)
    7. Decorrelated Difference SNR Map
    8. SNR Statistics

    Parameters:
    -----------
    maindir : str
        Path to the main directory where the raw data, PSFs, and output are stored.
    ref_name : str
        Name of the reference image.
    sci_name : str
        Name of the science image.
    ref title : str, optional
        Title for the reference image.
    sci_title : str, optional
        Title for the science image.
    show_fwhm : bool, optional
        If True, the FWHM of the PSF is displayed on the plot.
    scalebar : bool, optional
        If True, a scale bar is displayed on the plot.
    cutout_center : tuple, optional
        Center coordinates for a cutout (RA, Dec) in degrees. If None, no cutout is made. 
        If given, an inset will be made in the top left and a red circle will be drawn 
        around the coordinates.
    cutout_size : int, optional
        Size of the cutout in pixels. Required if cutout_center is provided.
    skysub : bool, optional
        If sky subtraction was performed.
    crossconv : bool, optional
        If the images were cross-convolved.
    output_fname : str, optional
        Name of the output file. If None, the plot is not saved.
    show : bool, optional
        If True, the plot is displayed.
        
    Returns:
    --------
    None    

    """
    
    cmap = plt.get_cmap('Greys')
    cmap.set_bad(color='DodgerBlue', alpha=0.5)
    
    cmap_snr = plt.get_cmap('coolwarm')
    cmap_snr.set_bad(color='w', alpha=1)
    
    
    indir = maindir + 'input/'
    psfdir = maindir + 'psf/'
    outdir = maindir + 'output/'
    maskdir = maindir + 'mask/'
    
    if ('f200w' in ref_name) or ('F200W' in ref_name):
        ps = .031
    elif ('f444w' in ref_name) or ('F444W' in ref_name):
        ps = .063
    
    
    if skysub:
        fname_lref = indir + '{}.skysub.fits'.format(ref_name)
        fname_lsci = indir + '{}.skysub.fits'.format(sci_name)
    else:
        fname_lref = indir + '{}.fits'.format(ref_name)
        fname_lsci = indir + '{}.fits'.format(sci_name)

    fname_psf_lref = psfdir + '{}.psf.fits'.format(ref_name)
    fname_psf_lsci = psfdir + '{}.psf.fits'.format(sci_name)

    if conv_ref:
        fname_cc_lref = outdir + '{}.crossconvd.fits'.format(ref_name)
    else:
        if skysub:
            fname_cc_lref = indir + '{}.skysub.fits'.format(ref_name)
        else:
            fname_cc_lref = indir + '{}.fits'.format(ref_name)
            
            
    if conv_sci:
        fname_cc_lsci = outdir + '{}.crossconvd.fits'.format(sci_name)
    else:
        if skysub:
            fname_cc_lsci = indir + '{}.skysub.fits'.format(sci_name)
        else:
            fname_cc_lsci = indir + '{}.fits'.format(sci_name)
            
    
    fname_mask_lref = indir + '{}.maskin.fits'.format(ref_name)
    fname_mask_lsci = indir + '{}.maskin.fits'.format(sci_name)
    
    fname_satcat_r = maskdir + '{}_saturated_sources.cat'.format(ref_name)
    fname_satcat_s = maskdir + '{}_saturated_sources.cat'.format(sci_name)
    fname_satcat_rg = maskdir + '{}_saturated_sources_bright.cat'.format(ref_name)
    fname_satcat_sg = maskdir + '{}_saturated_sources_bright.cat'.format(sci_name)

    fname_diff = outdir + '{}.sfftdiff.fits'.format(sci_name)
    fname_diff_decorr = outdir + '{}.sfftdiff.decorr.fits'.format(sci_name)
    fname_diff_decorr_snr = outdir + '{}.sfftdiff.decorr.snr.fits'.format(sci_name)

    fname_ker_match = outdir + '{}.sfftmatchker.mean.fits'.format(sci_name)
    fname_ker_decorr = outdir + '{}.sfftdecorrker.mean.fits'.format(sci_name)
    fname_decorr_bkg = outdir + '{}.decorrbkg.fits'.format(sci_name)
    fname_decorr_sig10 = outdir + '{}.decorrsig10.fits'.format(sci_name)
    fname_decorr_sig100 = outdir + '{}.decorrsig100.fits'.format(sci_name)

    fname_stats = outdir + '{}.decorrstats.lt10.dat'.format(sci_name)

    
    if (cutout_center is not None) or show_sat_sources:
        wcs_ref = WCS(fits.open(fname_lref)[0].header)
    
    
    if cutout_center is not None:
        assert cutout_size is not None, "If cutout_center is provided, cutout_size must also be provided."
        
        shape = (cutout_size, cutout_size)
        obj_coord = SkyCoord(cutout_center[0], cutout_center[1], unit=(u.deg, u.deg))
    
    
    im_lref = fits.getdata(fname_lref, ext=0)
    im_lsci = fits.getdata(fname_lsci, ext=0)

    im_psf_lref = fits.getdata(fname_psf_lref, ext=0)[450:-450, 450:-450]
    im_psf_lsci = fits.getdata(fname_psf_lsci, ext=0)[450:-450, 450:-450]
    im_psf_lref /= np.nansum(im_psf_lref)
    im_psf_lsci /= np.nansum(im_psf_lsci)
    
    im_mask_lref = fits.getdata(fname_mask_lref, ext=0).astype(bool)
    im_mask_lsci = fits.getdata(fname_mask_lsci, ext=0).astype(bool)
    im_mask_both = im_mask_lref | im_mask_lsci
    
    im_mask_boundary = np.ones(im_lref.shape, dtype=bool)
    im_mask_boundary[npx_boundary:-npx_boundary, npx_boundary:-npx_boundary] = False

    im_cc_lref = fits.getdata(fname_cc_lref, ext=0)
    im_cc_lsci = fits.getdata(fname_cc_lsci, ext=0)
    
    if show_sat_sources:
        catdat_sat_r = Table.read(fname_satcat_r)
        catdat_sat_s = Table.read(fname_satcat_s)  
        catdat_sat_rg = Table.read(fname_satcat_rg)
        catdat_sat_sg = Table.read(fname_satcat_sg)      

    try:
        im_diff = fits.getdata(fname_diff, ext=0)
    except:
        im_diff = np.full(im_lref.shape, np.nan)
    
    try:
        im_diff_decorr = fits.getdata(fname_diff_decorr, ext=0)
    except:
        im_diff_decorr = np.full(im_diff.shape, np.nan)
    
    try:
        im_diff_decorr_snr = fits.getdata(fname_diff_decorr_snr, ext=0)
    except:
        im_diff_decorr_snr = np.full(im_diff.shape, np.nan)

    try:
        ker_match = fits.getdata(fname_ker_match, ext=0)
        ker_decorr = fits.getdata(fname_ker_decorr, ext=0)
        ker_match /= np.nansum(ker_match)
        ker_decorr /= np.nansum(ker_decorr)
    except:
        ker_match = np.full(im_psf_lref.shape, np.nan)
        ker_decorr = np.full(im_psf_lref.shape, np.nan)

    
    if show_stats:
        statsdat = Table.read(fname_stats, format='ascii')

        decorr_bkg = fits.getdata(fname_decorr_bkg, ext=0)
        decorr_sig10 = fits.getdata(fname_decorr_sig10, ext=0)
        decorr_sig100 = fits.getdata(fname_decorr_sig100, ext=0)
        mask = np.isnan(im_diff) | (im_lref == 0.) | (im_lsci == 0.)
        decorr_bkg = decorr_bkg[(~mask) | im_mask_both | (decorr_bkg != 0.) | im_mask_boundary].flatten()
        decorr_sig10 = decorr_sig10[(~mask) | im_mask_both | (decorr_sig10 != 0.) | im_mask_boundary].flatten()
        decorr_sig100 = decorr_sig100[(~mask) | im_mask_both | (decorr_sig100 != 0.) | im_mask_boundary].flatten()
    
    #Get rid of NaNs
    im_diff[np.isnan(im_diff)] = np.nan
    im_diff_decorr[np.isnan(im_diff_decorr)] = np.nan
    im_diff_decorr_snr[np.isnan(im_diff_decorr_snr)] = np.nan
    ker_match[np.isnan(ker_match)] = np.nan 
    ker_decorr[np.isnan(ker_decorr)] =  np.nan 
        
    #Set masked regions to NaN
    im_lref[im_mask_both] = np.nan
    im_lsci[im_mask_both] = np.nan
    im_cc_lref[im_mask_both] = np.nan
    im_cc_lsci[im_mask_both] = np.nan
    im_diff[im_mask_both | im_mask_boundary] = np.nan
    im_diff_decorr[im_mask_both | im_mask_boundary] = np.nan
    im_diff_decorr_snr[im_mask_both | im_mask_boundary] = np.nan
    
    im_diff_decorr[im_diff_decorr == 0.] = np.nan    
    im_diff_decorr_snr[im_diff_decorr_snr == 0.] = np.nan
    
    ########################################################################################################################################
    ########################################################################################################################################    
    ########################################################################################################################################

    fig = plt.figure(figsize=(24,12))

    gs = GridSpec(1, 2, figure=fig, wspace=.1)
    gs_l = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0], wspace=0.05, hspace=0.15)
    gs_r = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1], wspace=0.05, hspace=0.15)

    ax00 = plt.subplot(gs_l[0,0])
    ax01 = plt.subplot(gs_l[0,1])
    ax10 = plt.subplot(gs_l[1,0])
    ax11 = plt.subplot(gs_l[1,1])
    ax_l = np.array([[ax00, ax01], [ax10, ax11]], dtype=object)

    ax00 = plt.subplot(gs_r[0,0])
    ax01 = plt.subplot(gs_r[0,1])
    ax10 = plt.subplot(gs_r[1,0])
    ax11 = plt.subplot(gs_r[1,1])
    ax_r = np.array([[ax00, ax01], [ax10, ax11]], dtype=object)

    big_fsize = 25
    small_fsize = 22


    ###################################################################
    ###########################  LEFT  ################################
    ###################################################################
    #Images
    norm = ImageNormalize(im_lref, interval=ZScaleInterval())
    ax_l[0,0].imshow(im_lref, cmap=cmap, norm=norm, origin='lower', aspect='auto')
    ax_l[0,0].text(.05, .95, 'REF', fontsize=15, ha='left', va='top', transform=ax_l[0,0].transAxes, color='k', fontweight='bold')
    
    #norm = ImageNormalize(im_lsci, interval=ZScaleInterval())
    ax_l[0,1].imshow(im_lsci, cmap=cmap, norm=norm, origin='lower', aspect='auto')
    ax_l[0,1].text(.05, .95, 'SCI', fontsize=15, ha='left', va='top', transform=ax_l[0,1].transAxes, color='k', fontweight='bold')
    
    if cutout_center is not None:
        px = skycoord_to_pixel(obj_coord, wcs_ref)
        circ = plt.Circle((px[0], px[1]), 10, color='red', fill=False)
        ax_l[0,0].add_patch(circ)
        
        px = skycoord_to_pixel(obj_coord, wcs_ref)
        circ = plt.Circle((px[0], px[1]), 10, color='red', fill=False)
        ax_l[0,1].add_patch(circ)
    

    if conv_ref:
        #norm = ImageNormalize(im_cc_lref, interval=ZScaleInterval())
        ax_l[1,0].imshow(im_cc_lref, cmap=cmap, norm=norm, origin='lower', aspect='auto')
        
        if cutout_center is not None:
            px = skycoord_to_pixel(obj_coord, wcs_ref)
            circ = plt.Circle((px[0], px[1]), 10, color='red', fill=False)
            ax_l[1,0].add_patch(circ)
            
            
        if show_sat_sources:
            for cat in [catdat_sat_r, catdat_sat_s, catdat_sat_rg, catdat_sat_sg]:            
                for i in range(len(cat)):
                    ra = cat['ALPHA_J2000'][i]
                    dec = cat['DELTA_J2000'][i]
                    coord = SkyCoord(ra, dec, unit=(u.deg, u.deg))
                    px = skycoord_to_pixel(coord, wcs_ref)
                    circ = plt.Circle((px[0], px[1]), 20, color='r', fill=False, lw=1)
                    ax_l[1,0].add_patch(circ)
                
    else:
        ax_l[1,0].axis('off')
        
        

    if conv_sci:
        #norm = ImageNormalize(im_cc_lsci, interval=ZScaleInterval())
        ax_l[1,1].imshow(im_cc_lsci, cmap=cmap, norm=norm, origin='lower', aspect='auto')
        
        if cutout_center is not None:
            px = skycoord_to_pixel(obj_coord, wcs_ref)
            circ = plt.Circle((px[0], px[1]), 10, color='red', fill=False)
            ax_l[1,1].add_patch(circ)
            
            
        if show_sat_sources:
            for cat in [catdat_sat_r, catdat_sat_s, catdat_sat_rg, catdat_sat_sg]:            
                for i in range(len(cat)):
                    ra = cat['ALPHA_J2000'][i]
                    dec = cat['DELTA_J2000'][i]
                    coord = SkyCoord(ra, dec, unit=(u.deg, u.deg))
                    px = skycoord_to_pixel(coord, wcs_ref)
                    circ = plt.Circle((px[0], px[1]), 20, color='r', fill=False, lw=1)
                    ax_l[1,1].add_patch(circ)

    else:
        ax_l[1,1].axis('off')



    ###################################################################
    #PSFs
    w = '30%'
    h = '30%'

    fpeak = np.nanmax(im_psf_lref)
    interval = ManualInterval(0, fpeak/1e4)
    norm = ImageNormalize(im_psf_lref, interval=interval, stretch=AsinhStretch())

    #REF
    ax_i = inset_axes(ax_l[0,0], width=w, height=h, loc='lower left', borderpad=0)
    ax_i.imshow(im_psf_lref, cmap='magma', origin='lower', norm=norm)
    ax_i.set_xticks([])
    ax_i.set_yticks([])
    
    if show_fwhm:
        hdr = fits.open(fname_psf_lref)[0].header
        if 'FWHM_major' in hdr:          
            f1 = hdr['FWHM_major']
            f2 = hdr['FWHM_minor']
            fval_ref = (f1+f2)/2
        elif 'FWHM' in hdr:
            fval_ref = hdr['FWHM']

            
        ax_i.text(.05, .95, 'FWHM = {:.2f}"'.format(fval_ref), fontsize=10, ha='left', va='top', transform=ax_i.transAxes, color='yellow', fontweight='bold')
            

    #SCI
    ax_i = inset_axes(ax_l[0,1], width=w, height=h, loc='lower left', borderpad=0)
    ax_i.imshow(im_psf_lsci, cmap='magma', origin='lower', norm=norm)
    ax_i.set_xticks([])
    ax_i.set_yticks([])
    
    if show_fwhm:
        hdr = fits.open(fname_psf_lsci)[0].header
        if 'FWHM_major' in hdr:
            f1 = hdr['FWHM_major']
            f2 = hdr['FWHM_minor']
            fval_sci = (f1+f2)/2
        elif 'FWHM' in hdr:
            fval_sci = hdr['FWHM']
            
        ax_i.text(.05, .95, 'FWHM = {:.2f}"'.format(fval_sci), fontsize=10, ha='left', va='top', transform=ax_i.transAxes, color='yellow', fontweight='bold')


    # if conv_ref:
    #     #REF CCd
    #     new_psf_ref = convolve_fft(im_psf_lref, im_psf_lsci, boundary='fill', nan_treatment='fill', fill_value=0., normalize_kernel=True)
    #     norm = ImageNormalize(new_psf_ref, interval=ZScaleInterval())
    #     ax_i = inset_axes(ax_l[1,0], width=w, height=h, loc='lower left', borderpad=0)
    #     ax_i.imshow(new_psf_ref, cmap='magma', norm=norm, origin='lower')
    #     ax_i.set_xticks([])
    #     ax_i.set_yticks([])

    # if conv_sci:
    #     #SCI CCd
    #     new_psf_sci = convolve_fft(im_psf_lsci, im_psf_lref, boundary='fill', nan_treatment='fill', fill_value=0., normalize_kernel=True)
    #     norm = ImageNormalize(new_psf_sci, interval=ZScaleInterval())
    #     ax_i = inset_axes(ax_l[1,1], width=w, height=h, loc='lower left', borderpad=0)
    #     ax_i.imshow(new_psf_sci, cmap='magma', norm=norm, origin='lower')
    #     ax_i.set_xticks([])
    #     ax_i.set_yticks([])


    ###################################################################
    #Arrows

    if conv_ref or conv_sci:
        ax_l[0,0].annotate('', xy=(0.3, -.15), xytext=(0.3, -.025), 
                        xycoords='axes fraction', textcoords='axes fraction', 
                        arrowprops=dict(facecolor='r', edgecolor='k', shrink=0.05))
        fig.text(.225, .495, r'$\bigotimes$ SCI PSF', ha='center', va='center', fontsize=small_fsize)

        ax_l[0,1].annotate('', xy=(0.3, -.15), xytext=(0.3, -.025), 
                        xycoords='axes fraction', textcoords='axes fraction', 
                        arrowprops=dict(facecolor='r', edgecolor='k', shrink=0.05))
        fig.text(.415, .495, r'$\bigotimes$ REF PSF', ha='center', va='center', fontsize=small_fsize)

    ###################################################################
    #Scalebars
    
    if scalebar:     
        dx = ps/3600.
        dy = ps/3600.
        padx = im_lref.shape[0] * .05
        pady = im_lref.shape[1] * .04
                    
        bar_width = (5/3600)/dx                 # 10 arcsec (in px)
        txt = '5"'
            
        bar_height = im_lref.shape[0] / 30
        
        bar_x2 = im_lref.shape[1] - padx
        bar_x1 = bar_x2 - bar_width
        bar_y = pady

        for a in [ax_l[1,0], ax_l[1,1]]:
            a.hlines(y=bar_y, xmin=bar_x1, xmax=bar_x2, color='yellow', lw=2)
            a.vlines(x=bar_x1, ymin=bar_y - bar_height/2, ymax=bar_y + bar_height/2, color='yellow', lw=2)
            a.vlines(x=bar_x2, ymin=bar_y - bar_height/2, ymax=bar_y + bar_height/2, color='yellow', lw=2)
        
            ytext = (  bar_y + (im_lref.shape[0]/100)  )/im_lref.shape[0]
            xtext = (bar_x1 + bar_x2)/2/im_lref.shape[1] + .01
            a.text(xtext, ytext, txt, color='yellow', fontsize=13, ha='center', va='bottom', transform=a.transAxes, weight='bold')

    
    ###################################################################

    txt = 'Reference Image'
    if ref_title is not None:
        txt = ref_title
    ax_l[0,0].set_title(txt, fontsize=big_fsize)

    txt = 'Science Image'
    if sci_title is not None:
        txt = sci_title
    ax_l[0,1].set_title(txt, fontsize=big_fsize)

    ax_l[0,0].set_ylabel('Input', fontsize=small_fsize)
    ax_l[1,0].set_ylabel('Cross-Convolved', fontsize=small_fsize)


    for a in ax_l.flatten():
        a.set_yticklabels([])
        a.set_xticklabels([])
        
        a.tick_params('both', which='both', width=1)
        a.tick_params('both', which='major', length=6)
        a.tick_params('both', which='minor', length=3)
        
        a.set_xlim(0, im_lref.shape[1])
        a.set_ylim(0, im_lref.shape[0])



    ###################################################################
    ###########################  RIGHT  ###############################
    ###################################################################
    #Images

    if ~np.all(np.isnan(im_diff)):
        norm = ImageNormalize(im_diff, interval=ZScaleInterval())
        ax_r[0,0].imshow(im_diff, cmap=cmap, norm=norm, origin='lower', aspect='auto')
        
        
        if show_sat_sources:
            for cat in [catdat_sat_r, catdat_sat_s, catdat_sat_rg, catdat_sat_sg]:            
                for i in range(len(cat)):
                    ra = cat['ALPHA_J2000'][i]
                    dec = cat['DELTA_J2000'][i]
                    coord = SkyCoord(ra, dec, unit=(u.deg, u.deg))
                    px = skycoord_to_pixel(coord, wcs_ref)
                    circ = plt.Circle((px[0], px[1]), 20, color='r', fill=False, lw=1)
                    ax_r[0,0].add_patch(circ)

    if ~np.all(np.isnan(im_diff_decorr)):
        norm = ImageNormalize(im_diff_decorr, interval=ZScaleInterval())
        ax_r[0,1].imshow(im_diff_decorr, cmap=cmap, norm=norm, origin='lower', aspect='auto')
        
        if show_sat_sources:
            for cat in [catdat_sat_r, catdat_sat_s, catdat_sat_rg, catdat_sat_sg]:            
                for i in range(len(cat)):
                    ra = cat['ALPHA_J2000'][i]
                    dec = cat['DELTA_J2000'][i]
                    coord = SkyCoord(ra, dec, unit=(u.deg, u.deg))
                    px = skycoord_to_pixel(coord, wcs_ref)
                    circ = plt.Circle((px[0], px[1]), 20, color='r', fill=False, lw=1)
                    ax_r[0,1].add_patch(circ)


    if ~np.all(np.isnan(im_diff_decorr_snr)):
        norm = ImageNormalize(im_diff_decorr_snr, interval=ZScaleInterval())
        im_out_snr = ax_r[1,1].imshow(im_diff_decorr_snr, cmap=cmap_snr, origin='lower', aspect='auto', vmin=-5, vmax=5)
        p16, p50, p84 = np.nanpercentile(im_diff_decorr_snr, [16, 50, 84])
        print(p16, p50, p84)
        
        cbar = plt.colorbar(im_out_snr, ax=ax_r[1,1], orientation='horizontal', pad=0.01, fraction=0.05)
        cbar.ax.tick_params(labelsize=15)
    
    ###################################################################
    #Cutouts
    
    w = '30%'
    h = '30%'
    
    if cutout_center is not None:
        cutout_decorr = Cutout2D(im_diff_decorr, obj_coord, shape, wcs=wcs_ref)
        cutout_snr = Cutout2D(im_diff_decorr_snr, obj_coord, shape, wcs=wcs_ref)
        
        #Decorr
        norm = ImageNormalize(cutout_decorr.data, interval=ZScaleInterval())
        ax_i = inset_axes(ax_r[0,1], width=w, height=h, loc='upper right', borderpad=0)
        ax_i.imshow(cutout_decorr.data, cmap='Greys', norm=norm, origin='lower')
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        
            #Full image
        px = skycoord_to_pixel(obj_coord, wcs_ref)
        circ = plt.Circle((px[0], px[1]), 10, color='red', fill=False)
        ax_r[0,1].add_patch(circ)
        
            #Cutout
        px = skycoord_to_pixel(obj_coord, cutout_decorr.wcs)
        circ = plt.Circle((px[0], px[1]), 10, color='red', fill=False)
        ax_i.add_patch(circ)
        

        #SNR
        norm = ImageNormalize(cutout_snr.data, interval=ZScaleInterval())
        ax_i = inset_axes(ax_r[1,1], width=w, height=h, loc='upper right', borderpad=0)
        ax_i.imshow(cutout_snr.data, cmap='Greys', norm=norm, origin='lower')
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        
            #Full image
        px = skycoord_to_pixel(obj_coord, wcs_ref)
        circ = plt.Circle((px[0], px[1]), 10, color='red', fill=False)
        ax_r[1,1].add_patch(circ)
        
            #Cutout
        px = skycoord_to_pixel(obj_coord, cutout_snr.wcs)
        circ = plt.Circle((px[0], px[1]), 10, color='red', fill=False)
        ax_i.add_patch(circ)



    ###################################################################
    #Stats
    
    if show_stats:
        snr_lim = 15.
        bins = np.linspace(-snr_lim, snr_lim, 100)

        row = statsdat[0]
        ax_r[1,0].hist( decorr_bkg[np.abs(decorr_bkg) < snr_lim], bins=bins, density=True,
                        color='#12aa9c', linewidth=1.3, label='background', alpha=0.2, zorder=1)

        ax_r[1,0].plot(bins, stats.norm.pdf(bins, row['Mean'], row['STD']), ls='-', color='#12aa9c', linewidth=3,  #label=txt,
                    alpha=1, zorder=1)



        row = statsdat[1]
        ax_r[1,0].hist( decorr_sig10[np.abs(decorr_sig10) < snr_lim], bins=bins, density=True,
                        color='#FD574F', linewidth=1.5, label=r'${\bf {\rm signal} (> 10\sigma_s) }$', 
                        alpha=1, zorder=2, histtype='step')

        row = statsdat[2]
        ax_r[1,0].hist( decorr_sig100[np.abs(decorr_sig100) < snr_lim], bins=bins, density=True,
                        color='g', linewidth=1.5, label=r'${\bf {\rm signal} (> 100\sigma_s) }$',
                        alpha=1, zorder=3, histtype='step')



        ax_r[1,0].text(.65, .925, 'Skewness', fontsize=15., color='black', va='top', ha='left', transform=ax_r[1,0].transAxes)
        ax_r[1,0].text(.65, .875, '{:.2f}'.format(statsdat['Skew'][0]), fontsize=15., 
                    color='#12aa9c', va='top', ha='left', transform=ax_r[1,0].transAxes)
        ax_r[1,0].text(.65, .825, '{:.2f}'.format(statsdat['Skew'][1]), fontsize=15.,
                        color='#FD574F', va='top', ha='left', transform=ax_r[1,0].transAxes)
        ax_r[1,0].text(.65, .775, '{:.2f}'.format(statsdat['Skew'][2]), fontsize=15.,
                        color='g', va='top', ha='left', transform=ax_r[1,0].transAxes)

        ax_r[1,0].text(.65, .65, r'$\mu$', fontsize=15., color='black', va='top', ha='left', transform=ax_r[1,0].transAxes)
        ax_r[1,0].text(.65, .6, '{:.2f}'.format(statsdat['Mean'][0]), fontsize=15., color='#12aa9c', va='top', ha='left', transform=ax_r[1,0].transAxes)
        ax_r[1,0].text(.65, .55, '{:.2f}'.format(statsdat['Mean'][1]), fontsize=15., color='#FD574F', va='top', ha='left', transform=ax_r[1,0].transAxes)
        ax_r[1,0].text(.65, .5, '{:.2f}'.format(statsdat['Mean'][2]), fontsize=15., color='g', va='top', ha='left', transform=ax_r[1,0].transAxes)

        ax_r[1,0].text(.85, .65, r'$\sigma$', fontsize=15., color='black', va='top', ha='left', transform=ax_r[1,0].transAxes)
        ax_r[1,0].text(.85, .6, '{:.2f}'.format(statsdat['STD'][0]), fontsize=15., color='#12aa9c', va='top', ha='left', transform=ax_r[1,0].transAxes)
        ax_r[1,0].text(.85, .55, '{:.2f}'.format(statsdat['STD'][1]), fontsize=15., color='#FD574F', va='top', ha='left', transform=ax_r[1,0].transAxes)
        ax_r[1,0].text(.85, .5, '{:.2f}'.format(statsdat['STD'][2]), fontsize=15., color='g', va='top', ha='left', transform=ax_r[1,0].transAxes)


        ax_r[1,0].legend(loc='upper left', frameon=False, fontsize=12, borderpad=1)
        ax_r[1,0].set_xlim(-snr_lim, snr_lim)
        ax_r[1,0].set_xlabel('SNR', fontsize=small_fsize)


    ###################################################################
    #PSF

    w = '30%'
    h = '30%'

    #Matched Kernel
    
    if ~np.all(np.isnan(ker_match)):
        norm = ImageNormalize(ker_match, interval=ZScaleInterval())
        ax_i = inset_axes(ax_r[0,0], width=w, height=h, loc='lower right', borderpad=0)
        ax_i.imshow(ker_match, cmap='magma', norm=norm, origin='lower')
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        ax_i.text(.5, 1.05, 'Matching'+'\n'+'Kernel', fontsize=14, ha='center', va='bottom', ma='center', transform=ax_i.transAxes, color='yellow', fontweight='bold')

    # #Decorrelation Kernel
    # norm = ImageNormalize(ker_decorr, interval=ZScaleInterval())
    # ax_i = inset_axes(ax_r[0,1], width=w, height=h, loc='lower right', borderpad=0)
    # ax_i.imshow(ker_decorr, cmap='magma', norm=norm, origin='lower')
    # ax_i.set_xticks([])
    # ax_i.set_yticks([])
    # ax_i.text(.5, 1.05, 'Decorr'+'\n'+'Kernel', fontsize=14, ha='center', va='bottom', ma='center', transform=ax_i.transAxes, color='yellow', fontweight='bold')

    # #Decorrelated kernel
    # if forceconv == 'ref':
    #     im_psf1 = im_psf_lsci.copy()
    #     im_psf2 = im_psf_lref.copy()
    # if forceconv == 'sci':
    #     im_psf1 = im_psf_lref.copy()
    #     im_psf2 = im_psf_lsci.copy()

    # if conv_sci:
    #     new_psf = convolve_fft(im_psf1, im_psf2, boundary='fill', nan_treatment='fill', fill_value=0., normalize_kernel=True)
    # else:
    #     new_psf = im_psf1.copy()
      
    # if np.nansum(ker_decorr) < .01:
    #     ker_decorr /= np.nansum(ker_decorr)
    #     norm = False
    # else:
    #     norm = True

    # kernel = convolve_fft(new_psf, ker_decorr, boundary='fill', nan_treatment='fill', fill_value=0., normalize_kernel=norm)
    # norm = ImageNormalize(kernel, interval=ZScaleInterval())
    # ax_i = inset_axes(ax_r[0,1], width=w, height=h, loc='lower left', borderpad=0)
    # ax_i.imshow(kernel, cmap='magma', norm=norm, origin='lower')
    # ax_i.set_xticks([])
    # ax_i.set_yticks([])
    
    # if show_fwhm:
    #     #Get FWHM estimate
    #     fwhm_est_in = np.mean([fval_ref, fval_sci])
    #     x0 = int(  im_psf_lsci.shape[0] // 2  )
    #     y0 = int(  im_psf_lsci.shape[1] // 2  )
        
    #     dict_out = estimate_psf_fwhm(new_psf, x0, y0, fwhm_est_in, ps=ps)
    #     fval_decorr = (dict_out['FWHM_major'] + dict_out['FWHM_minor'])/2
        
    #     ax_i.text(.05, .95, 'FWHM = {:.2f}"'.format(fval_decorr), fontsize=10, ha='left', va='top', transform=ax_i.transAxes, color='yellow', fontweight='bold')


    ###################################################################
    #Arrows

    ax_r[0,1].annotate('', xy=(0.3, -.15), xytext=(0.3, -.025),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(facecolor='r', edgecolor='k', shrink=0.05))
    fig.text(.8, .49, 'SNR', ha='center', va='center', fontsize=small_fsize)

    if conv_ref or conv_sci:
        xytext = (1,1)
        xy = (1.2, 1.145)
        xtxt = .525
        ytxt = .48
    else:
        xytext = (1.01,1.7)
        xy = (1.2, 1.7)
        xtxt = .512
        ytxt = .68

    ax_l[1,1].annotate('', xy=xy, xytext=xytext,
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(facecolor='r', edgecolor='k', shrink=0.05))
    fig.text(xtxt, ytxt, 'SFFT', ha='center', va='center', fontsize=small_fsize)

    plt.annotate('', xy=(.625, .68), xytext=(.6, .68),
                        xycoords='figure fraction', textcoords='figure fraction',
                        arrowprops=dict(facecolor='r', edgecolor='k', shrink=0.05))

    plt.annotate('', xy=(.6, .25), xytext=(.625, .25),
                        xycoords='figure fraction', textcoords='figure fraction',
                        arrowprops=dict(facecolor='r', edgecolor='k', shrink=0.05))

    ###################################################################

    for a in ax_r.flatten()[:2]:
        a.set_yticklabels([])
        a.set_xticklabels([])
        
        a.tick_params('both', which='both', width=1)
        a.tick_params('both', which='major', length=6)
        a.tick_params('both', which='minor', length=3)
        
    for a in ax_r.flatten()[2:3]:
        a.set_yticks([])
        
        a.tick_params('x', labelsize=15)
        
        a.tick_params('both', which='both', width=1)
        a.tick_params('both', which='major', length=7)
        a.tick_params('both', which='minor', length=3)
        

    for a in ax_r.flatten()[3:]:
        a.set_yticklabels([])
        a.set_xticklabels([])
        
        a.tick_params('both', which='both', width=1)
        a.tick_params('both', which='major', length=6)
        a.tick_params('both', which='minor', length=3)
        

    ax_r[0,0].set_title('Difference', fontsize=big_fsize)
    ax_r[0,1].set_title('Decorrelated Difference', fontsize=big_fsize)

    if output_fname is not None:
        plt.savefig(output_fname, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
        
    plt.cla()
    plt.clf()
    plt.close()

    return