import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.visualization import ImageNormalize, ZScaleInterval

import os
import glob
import multiprocessing as mp

import numpy as np
from astropy.io import fits


def get_imname(f):
    list_of_str = os.path.basename(f).split('.')[0].split('_')
    return '_'.join(list_of_str[:4])



def get_preprocess_plot(maindir, imname, show=False, save=True):
    
    ##################
    #Get data

    fname0 = maindir + 'raw/' + imname + '_uncal.fits'
    fname1 = maindir + 'detector1/' + imname + '_rate.fits'
    fname2 = maindir + 'wisp/' + imname + '_wisp.fits'
    fname3 = maindir + 'stripe/' + imname + '_destripe.fits'
    fname4 = maindir + 'image2/' + imname + '_cal.fits'
    fname5 = maindir + 'skysub/' + imname + '_skysub.fits'
    
    fname_nc = maindir + 'stripe/' + imname + '_noisechisel_mask.fits'
    fname_h = maindir + 'stripe/' + imname + '_mask_horiz.fits'
    fname_v = maindir + 'stripe/' + imname + '_mask_vert.fits'
    
    im0 = fits.open(fname0)[1].data[0,0]
    im1 = fits.getdata(fname1, 'SCI')
    im2 = fits.getdata(fname2, 'SCI')
    im3 = fits.getdata(fname3, 'SCI')
    im4 = fits.getdata(fname4, 'SCI')
    im5 = fits.getdata(fname5, 'SCI')
    
    
    dq_mask = fits.open(fname1)[3].data
    hdul_nc = fits.open(fname_nc)
    hdul_h = fits.open(fname_h)
    hdul_v = fits.open(fname_v)
    
    ##################
    #Plot
    
    fig, ax = plt.subplots(2, 6, figsize=(25, 12), sharex=True, sharey=True)
    cmap = cm.Greys_r
    cmap.set_bad('firebrick', .9)

    norm = ImageNormalize(im0, interval=ZScaleInterval())
    ax[0,0].imshow(im0, origin='lower', aspect='auto', interpolation='none', norm=norm, cmap=cmap)

    norm = ImageNormalize(im1, interval=ZScaleInterval())
    ax[0,1].imshow(im1, origin='lower', aspect='auto', interpolation='none', norm=norm, cmap=cmap)

    norm = ImageNormalize(im2, interval=ZScaleInterval())
    ax[0,2].imshow(im2, origin='lower', aspect='auto', interpolation='none', norm=norm, cmap=cmap)

    norm = ImageNormalize(im3, interval=ZScaleInterval())
    ax[0,3].imshow(im3, origin='lower', aspect='auto', interpolation='none', norm=norm, cmap=cmap)

    norm4 = ImageNormalize(im4, interval=ZScaleInterval())
    ax[0,4].imshow(im4, origin='lower', aspect='auto', interpolation='none', norm=norm4, cmap=cmap)

    norm5 = ImageNormalize(im5, interval=ZScaleInterval())
    ax[0,5].imshow(im5, origin='lower', aspect='auto', interpolation='none', norm=norm5, cmap=cmap)



    #DQ Mask
    dq_plot = np.array(im0.copy(), dtype=float)
    dq_plot[dq_mask > 0] = np.nan
    norm = ImageNormalize(dq_plot, interval=ZScaleInterval())
    ax[1,0].imshow(dq_plot, origin='lower', aspect='auto', norm=norm, cmap=cmap, interpolation='none')

    #NoiseChisel mask
    im_mask = im1.copy()
    nc_mask = hdul_nc[2].data
    im_mask[nc_mask > 0] = np.nan
    norm = ImageNormalize(im_mask, interval=ZScaleInterval())
    ax[1,2].imshow(im_mask, origin='lower', aspect='auto', norm=norm, cmap=cmap, interpolation='none')

    #Stripe Mask
    smask = hdul_h[0].data + hdul_v[0].data
    smask /= np.nanmax(smask)
    norm = ImageNormalize(smask, interval=ZScaleInterval())
    ax[1,3].imshow(smask, origin='lower', aspect='auto', norm=norm, cmap=cmap, interpolation='none')


    ax[0,0].set_title('Uncal', fontsize=15)
    ax[0,1].set_title('Rate', fontsize=15)
    ax[0,2].set_title('Wisp Removed', fontsize=15)
    ax[0,3].set_title('Destriped', fontsize=15)
    ax[0,4].set_title('Cal', fontsize=15)
    ax[0,5].set_title('Sky Subtracted', fontsize=15)

    ax[1,0].set_title('DQ Mask', fontsize=15)
    ax[1,2].set_title('NoiseChisel Source Mask', fontsize=15)
    ax[1,3].set_title('Stripe Mask', fontsize=15)

    for a in ax.flatten():
        a.set_xticklabels([])
        a.set_yticklabels([])
        
    ax[1,1].axis('off')
    ax[1,4].axis('off')
    ax[1,5].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.125)
    
    output_fname = maindir + 'plots/' + imname + '_preprocess.png'
    if save:
        plt.savefig(output_fname, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close()
    
    return



def plot_all(maindir, ncpu=1):
    
    fnames = glob.glob(maindir + 'preprocess/raw/*.fits')
    imnames = [get_imname(f) for f in fnames]

    if ncpu > 1:
        pool = mp.Pool(ncpu)
        pool.starmap(get_preprocess_plot, [(maindir + 'preprocess/', imname, False, True) for imname in imnames])
        pool.close()
        pool.join()
    else:
        for imname in imnames:
            get_preprocess_plot(maindir, imname, False, True)
    
    return