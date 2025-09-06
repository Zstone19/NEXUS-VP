# %matplotlib inline
import matplotlib.pyplot as plt

from functools import partial
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from scipy.optimize import curve_fit
from photutils.utils import ImageDepth


from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.wcs import WCS


def get_dm(fdiff, fwide):
    return -2.5 * np.log10(1. + fdiff/fwide)

def get_dm_err(fdiff, fwide, ferr_diff, ferr_wide):
    return (5/2/np.log(10)/(fwide + fdiff)) * np.sqrt( (ferr_wide*fdiff/fwide)**2 + ferr_diff**2 )


def fitfunc(xvals, A, B, mstar=0, alpha=0, beta=0):
    ystar = alpha*mstar + beta
    
    #Where (y'=alpha) and (y=ystar)
    C = alpha - 3*A*mstar**2 - 2*B*mstar
    D = ystar - (A*mstar**3 + B*mstar**2 + C*mstar)
    
    yvals_fit = A*xvals**3 + B*xvals**2 + C*xvals + D

    #Need the dip (y' = 0) to be before mstar
    # fill_val = 10000
    # if alpha < (3.*A*mstar + 2.*B)*(mstar - 1.):
        
    #     if isinstance(xvals, np.ndarray):
    #         yvals_fit[xvals > mstar] = fill_val
    #     elif isinstance(xvals, list):
    #         yvals_fit = [fill_val if x > mstar else y for x, y in zip(xvals, yvals_fit)]
    #     else:
    #         if xvals > mstar:
    #             yvals_fit = fill_val

    return yvals_fit 


def fitfunc_large_poly(xvals, A, B, C, D, mstar=0, alpha=0, beta=0):
    #Set arbitrary poly degree to 5 
    #y = Ax^5 + Bx^4 + Cx^3 + Dx^2 + Ex + F
    #y' = 5Ax^4 + 4Bx^3 + 3Cx^2 + 2Dx + E
    
    ystar = alpha*mstar + beta
    #Where (y'=alpha) and (y=ystar)
    E = alpha - (5*A*mstar**4 + 4*B*mstar**3 + 3*C*mstar**2 + 2*D*mstar)
    F = ystar - (A*mstar**5 + B*mstar**4 + C*mstar**3 + D*mstar**2 + E*mstar)

    yvals_fit = A*xvals**5 + B*xvals**4 + C*xvals**3 + D*xvals**2 + E*xvals + F

    return yvals_fit    
    


def get_extra_coeffs(A, B, mstar, alpha, beta):
    ystar = alpha*mstar + beta

    C = alpha - (3*A*mstar**2 + 2*B*mstar)
    D = ystar - (A*mstar**3 + B*mstar**2 + C*mstar)
    return A, B, C, D

def get_extra_coeffs_large_poly(A, B, C, D, mstar, alpha, beta):
    ystar = alpha*mstar + beta

    E = alpha - (5*A*mstar**4 + 4*B*mstar**3 + 3*C*mstar**2 + 2*D*mstar)
    F = ystar - (A*mstar**5 + B*mstar**4 + C*mstar**3 + D*mstar**2 + E*mstar)
    return A, B, C, D, E, F



def fit_for_sigma_curve(mavg_vals_fit, binned_dm_std, mag_centers, mstar, slope_max=.005):
    A0 = 1.
    B0 = 1.
    
    
    coeffs0 = [10, 10]
    
    # mstar = 24.5
    while (coeffs0[0] > slope_max): 
        if np.sum(mavg_vals_fit <= mstar) == 0:
            coeffs0 = np.array([0., mavg_vals_fit[0]])

            func_i = partial(fitfunc, mstar=mstar, alpha=coeffs0[0], beta=coeffs0[1])
            res, _ = curve_fit(func_i, mavg_vals_fit[mavg_vals_fit >= mstar], binned_dm_std[mavg_vals_fit >= mstar], p0=[A0,B0],
                            bounds=([0, -np.inf], [np.inf, np.inf]) )
            coeffs1 = res    
            break
        
        
        
        coeffs0 = np.polyfit(mavg_vals_fit[mavg_vals_fit <= mstar], binned_dm_std[mavg_vals_fit <= mstar], 1)

        func_i = partial(fitfunc, mstar=mstar, alpha=coeffs0[0], beta=coeffs0[1])
        res, _ = curve_fit(func_i, mavg_vals_fit[mavg_vals_fit >= mstar], binned_dm_std[mavg_vals_fit >= mstar], p0=[A0,B0],
                        bounds=([0, -np.inf], [np.inf, np.inf]) )
        coeffs1 = res
        
        if coeffs0[0] < 0: 
            coeffs0 = np.polyfit(mavg_vals_fit[mavg_vals_fit <= mstar], binned_dm_std[mavg_vals_fit <= mstar], 0)
            
            coeffs0 = np.array([0., coeffs0[0]])


            func_i = partial(fitfunc, mstar=mstar, alpha=coeffs0[0], beta=coeffs0[1])
            res, _ = curve_fit(func_i, mavg_vals_fit[mavg_vals_fit >= mstar], binned_dm_std[mavg_vals_fit >= mstar], p0=[A0,B0],
                            bounds=([0, -np.inf], [np.inf, np.inf]) )
            coeffs1 = res                
            break


        if mstar-.1 < mag_centers[0]:
            coeffs0 = np.polyfit(mavg_vals_fit[mavg_vals_fit <= mstar], binned_dm_std[mavg_vals_fit <= mstar], 0)
            
            coeffs0 = np.array([0., coeffs0[0]])


            func_i = partial(fitfunc, mstar=mstar, alpha=coeffs0[0], beta=coeffs0[1])
            res, _ = curve_fit(func_i, mavg_vals_fit[mavg_vals_fit >= mstar], binned_dm_std[mavg_vals_fit >= mstar], p0=[A0,B0],
                            bounds=([0, -np.inf], [np.inf, np.inf]) )
            coeffs1 = res
            break
            
        if coeffs0[0] > slope_max:
            mstar -= 0.05
            
            
    return mstar, coeffs0, coeffs1


def fit_for_mean_curve(mavg_vals_fit, binned_dm_std, err, mag_centers, mstar=26, slope_max=.005):
    A0 = 1.
    B0 = 1.
    C0 = 1.
    D0 = 1.
    
    coeffs0 = [10., 10.]
    
    # mstar = 24.5
    while (np.abs(coeffs0[0]) > slope_max): 
        if np.sum(mavg_vals_fit <= mstar) == 0:
            coeffs0 = np.array([0., mavg_vals_fit[0]])

            func_i = partial(fitfunc_large_poly, mstar=mstar, alpha=coeffs0[0], beta=coeffs0[1])
            res, _ = curve_fit(func_i, mavg_vals_fit[mavg_vals_fit >= mstar], binned_dm_std[mavg_vals_fit >= mstar], p0=[A0,B0,C0,D0], sigma=err[mavg_vals_fit >= mstar],
                            bounds=([0., -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]) )
            coeffs1 = res    
            break
        
        
        coeffs0 = np.polyfit(mavg_vals_fit[mavg_vals_fit <= mstar], binned_dm_std[mavg_vals_fit <= mstar], 1)

        func_i = partial(fitfunc_large_poly, mstar=mstar, alpha=coeffs0[0], beta=coeffs0[1])
        res, _ = curve_fit(func_i, mavg_vals_fit[mavg_vals_fit >= mstar], binned_dm_std[mavg_vals_fit >= mstar], p0=[A0,B0,C0,D0], sigma=err[mavg_vals_fit >= mstar],
                        bounds=([0., -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]) )
        coeffs1 = res


        if mstar-.05 < mag_centers[0]:
            coeffs0 = np.polyfit(mavg_vals_fit[mavg_vals_fit <= mstar], binned_dm_std[mavg_vals_fit <= mstar], 0)
            
            coeffs0 = np.array([0., coeffs0[0]])


            func_i = partial(fitfunc_large_poly, mstar=mstar, alpha=coeffs0[0], beta=coeffs0[1])
            res, _ = curve_fit(func_i, mavg_vals_fit[mavg_vals_fit >= mstar], binned_dm_std[mavg_vals_fit >= mstar], p0=[A0,B0,C0,D0], sigma=err[mavg_vals_fit >= mstar],
                            bounds=([0., -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]) )
            coeffs1 = res
            break
            
        if np.abs(coeffs0[0]) > slope_max:
            mstar -= 0.05
            
            
    return mstar, coeffs0, coeffs1


def fit_for_mean_curve_posslope(mavg_vals_fit, binned_dm_std, err, mag_centers, mstar=26, slope_max=.005):
    A0 = 1.
    B0 = 1.
    C0 = 1.
    D0 = 1.
    
    coeffs0 = [10., 10.]
    
    # mstar = 24.5
    while (coeffs0[0] > slope_max): 
        if np.sum(mavg_vals_fit <= mstar) == 0:
            coeffs0 = np.array([0., mavg_vals_fit[0]])

            func_i = partial(fitfunc_large_poly, mstar=mstar, alpha=coeffs0[0], beta=coeffs0[1])
            res, _ = curve_fit(func_i, mavg_vals_fit[mavg_vals_fit >= mstar], binned_dm_std[mavg_vals_fit >= mstar], p0=[A0,B0,C0,D0], sigma=err[mavg_vals_fit >= mstar],
                            bounds=([0., -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]) )
            coeffs1 = res    
            break
        
        
        coeffs0 = np.polyfit(mavg_vals_fit[mavg_vals_fit <= mstar], binned_dm_std[mavg_vals_fit <= mstar], 1)

        func_i = partial(fitfunc_large_poly, mstar=mstar, alpha=coeffs0[0], beta=coeffs0[1])
        res, _ = curve_fit(func_i, mavg_vals_fit[mavg_vals_fit >= mstar], binned_dm_std[mavg_vals_fit >= mstar], p0=[A0,B0,C0,D0], sigma=err[mavg_vals_fit >= mstar],
                        bounds=([0., -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]) )
        coeffs1 = res
        
        if coeffs0[0] < 0: 
            coeffs0 = np.polyfit(mavg_vals_fit[mavg_vals_fit <= mstar], binned_dm_std[mavg_vals_fit <= mstar], 0)
            
            coeffs0 = np.array([0., coeffs0[0]])


            func_i = partial(fitfunc_large_poly, mstar=mstar, alpha=coeffs0[0], beta=coeffs0[1])
            res, _ = curve_fit(func_i, mavg_vals_fit[mavg_vals_fit >= mstar], binned_dm_std[mavg_vals_fit >= mstar], p0=[A0,B0, C0, D0], sigma=err[mavg_vals_fit >= mstar],
                            bounds=([0., -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]) )
            coeffs1 = res                
            break


        if mstar-.05 < mag_centers[0]:
            coeffs0 = np.polyfit(mavg_vals_fit[mavg_vals_fit <= mstar], binned_dm_std[mavg_vals_fit <= mstar], 0)
            
            coeffs0 = np.array([0., coeffs0[0]])


            func_i = partial(fitfunc_large_poly, mstar=mstar, alpha=coeffs0[0], beta=coeffs0[1])
            res, _ = curve_fit(func_i, mavg_vals_fit[mavg_vals_fit >= mstar], binned_dm_std[mavg_vals_fit >= mstar], p0=[A0,B0,C0,D0], sigma=err[mavg_vals_fit >= mstar],
                            bounds=([0., -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]) )
            coeffs1 = res
            break
            
        if coeffs0[0] > slope_max:
            mstar -= 0.05
            
            
    return mstar, coeffs0, coeffs1


def unbias_dm(dat_in, suffix='', nmin=50, max_slope=1e-3, dmag=0.5,
              remove_flag=True, remove_saturated=True, remove_elongated=True, max_magdiff=35,
              difftype='FINAL'):

    apers = [0.2, 0.3, 0.5]
    bands = ['F200W', 'F444W']

    if 'PHOT' in suffix:
        naper = 3
    else:
        naper = 4

    dat_all = []
    for i in range(2):        
        #Restrict to m_diff_best (0.2") < 35
        magdiff = dat_in[i]['DIFF_MAG_FINAL'][:,1].data
        mask = magdiff < max_magdiff
        dat_all.append(dat_in[i][mask].copy())


    # max_slope = .005


    rows_dm_std = []
    rows_dm_mean = []
    rows_dmerr_med = []
    rows_dm_std_unbias = []
    rows_dm_mean_unbias = []
    rows_dmerr_med_unbias = []


    suffix = ''
    for i in range(2):
        print(bands[i])
        
        bad_mask = np.zeros(len(dat_all[i]), dtype=bool)
            
        bad_mask1 = (dat_all[i]['FLAGS_REFStack'].data > 0) | (dat_all[i]['FLAGS_SCIStack'].data > 0)
        bad_mask3 = (dat_all[i]['ELONGATION_REFStack'].data > 1.5) | (dat_all[i]['ELONGATION_SCIStack'].data > 1.5)
        bad_mask4 = dat_all[i]['SATURATED']
        
        if remove_flag:
            bad_mask |= bad_mask1
        if remove_saturated:
            bad_mask |= bad_mask4
        if remove_elongated:
            bad_mask |= bad_mask3
        
        # bad_mask = np.zeros(len(dat_all[i]), dtype=bool)
            
        for j in range(naper):
            print('\t Aper {}'.format(j+1))
            
            if (i == 0) and (j == 0):
                mmax = 29.
            elif (i==1) and (j == 0):
                mmax = 29.
            elif (i == 0) and (j == 1):
                mmax = 30.
            elif (i==1) and (j == 1):
                mmax = 29.75
            elif (i == 0) and (j == 2):
                mmax = 29.5
            elif (i==1) and (j == 2):
                mmax = 29.5
            elif (i == 0) and (j == 3):
                mmax = 29.5
            elif (i==1) and (j == 3):
                mmax = 29.
            
            # mstar_init = mmax-1.
            mstar_init = 24.5
            
            mag_bins = np.arange(19, mmax, dmag)
            mag_centers = mag_bins[:-1] + (mag_bins[1] - mag_bins[0])/2.

            
            
            if (naper == 4) and (j == 0):                
                fr = dat_all[i]['FLUX_AUTO_REFStack'].data[~bad_mask]
                ferr_r = dat_all[i]['FLUXERR_AUTO_REFStack'].data[~bad_mask]
                
                fs = dat_all[i]['FLUX_AUTO_SCIStack'].data[~bad_mask]
                ferr_s = dat_all[i]['FLUXERR_AUTO_SCIStack'].data[~bad_mask]
                
                if difftype == 'FINAL':
                    dm_diff = dat_all[i]['DM{}_FINAL'.format(suffix)].data[:,j][~bad_mask]
                    dmerr_diff = dat_all[i]['DMERR{}_FINAL'.format(suffix)].data[:,j][~bad_mask]
                elif difftype == 'SUB':
                    dm_diff = dat_all[i]['DM_AUTO_DiffStackSub'].data[~bad_mask]
                    dmerr_diff = dat_all[i]['DMERR_AUTO_DiffStackSub'].data[~bad_mask]
                elif difftype == 'SFFT':
                    dm_diff = dat_all[i]['DM_AUTO_DiffStackSFFT'].data[~bad_mask]
                    dmerr_diff = dat_all[i]['DMERR_AUTO_DiffStackSFFT'].data[~bad_mask]
                elif difftype == 'CAT':
                    mag_r = -2.5 * np.log10(np.abs(fr)) + 23.9
                    mag_s = -2.5 * np.log10(np.abs(fs)) + 23.9
                    magerr_r = np.abs(ferr_r / fr / np.log(10) * 2.5)
                    magerr_s = np.abs(ferr_s / fs / np.log(10) * 2.5)
                    
                    dm_diff = mag_s - mag_r
                    dmerr_diff = np.sqrt(magerr_s**2 + magerr_r**2)
                    

                
            else:
                if (naper == 4):
                    ind1 = j-1
                    ind2 = j
                else:
                    ind1 = j
                    ind2 = j+1
                
                fr = dat_all[i]['FLUX_APER_REFStack'].data[:,ind1][~bad_mask]
                ferr_r = dat_all[i]['FLUXERR_APER_REFStack'].data[:,ind1][~bad_mask]
                
                fs = dat_all[i]['FLUX_APER_SCIStack'].data[:,ind1][~bad_mask]
                ferr_s = dat_all[i]['FLUXERR_APER_SCIStack'].data[:,ind1][~bad_mask]

                if difftype == 'FINAL':
                    dm_diff = dat_all[i]['DM{}_FINAL'.format(suffix)].data[:,ind2][~bad_mask]
                    dmerr_diff = dat_all[i]['DMERR{}_FINAL'.format(suffix)].data[:,ind2][~bad_mask]
                elif difftype == 'SUB':
                    dm_diff = dat_all[i]['DM_APER_DiffStackSub'].data[:,ind1][~bad_mask]
                    dmerr_diff = dat_all[i]['DMERR_APER_DiffStackSub'].data[:,ind1][~bad_mask]
                elif difftype == 'SFFT':
                    dm_diff = dat_all[i]['DM_APER_DiffStackSFFT'].data[:,ind1][~bad_mask]
                    dmerr_diff = dat_all[i]['DMERR_APER_DiffStackSFFT'].data[:,ind1][~bad_mask]
                elif difftype == 'CAT':
                    mag_r = -2.5 * np.log10(np.abs(fr)) + 23.9
                    mag_s = -2.5 * np.log10(np.abs(fs)) + 23.9
                    magerr_r = np.abs(ferr_r / fr / np.log(10) * 2.5)
                    magerr_s = np.abs(ferr_s / fs / np.log(10) * 2.5)

                    dm_diff = mag_s - mag_r
                    dmerr_diff = np.sqrt(magerr_s**2 + magerr_r**2)

            mag_r = -2.5 * np.log10(np.abs(fr)) + 23.9
            mag_s = -2.5 * np.log10(np.abs(fs)) + 23.9
            magerr_r = np.abs( ferr_r / fr / np.log(10) * 2.5 )
            magerr_s = np.abs( ferr_s / fs / np.log(10) * 2.5 )

            mag_avg = (mag_r + mag_s) / 2.
            
            dm_use = dm_diff.copy()
            dmerr_use = dmerr_diff.copy()
            
            #######################
            
            
            
            mavg_vals_fit = []
            binned_dm_std = []
            for k in range(len(mag_bins)-1):
                mask = (mag_avg > mag_bins[k]) & (mag_avg <= mag_bins[k+1])
                nobj = np.sum(mask)
                
                if nobj < nmin:
                    continue    
                        
                _, _, std = sigma_clipped_stats(dm_use[mask], sigma=3., maxiters=None)
                
                assert np.sum(dm_use[mask] > 3*std) < float(nobj)/2.
                
                mavg_vals_fit.append(mag_centers[k])
                binned_dm_std.append(std)

                
            mavg_vals_fit = np.array(mavg_vals_fit)
            binned_dm_std = np.array(binned_dm_std)
                
                
            #Fit using DIFF    
            # mstar, coeffs0, coeffs1 = fit_for_sigma_curve(mavg_vals_fit, binned_dm_std, mag_centers, mstar=mstar_init, slope_max=max_slope)   
            # func_i = partial(fitfunc, mstar=mstar, alpha=coeffs0[0], beta=coeffs0[1])     

            mstar, coeffs0, coeffs1 = fit_for_mean_curve_posslope(mavg_vals_fit, binned_dm_std, np.ones_like(mavg_vals_fit), mag_centers, mstar=mstar_init, slope_max=max_slope)
            func_i = partial(fitfunc_large_poly, mstar=mstar, alpha=coeffs0[0], beta=coeffs0[1])

            
            dm_std = np.zeros_like(dm_use)
            dm_std[mag_avg < mstar] = coeffs0[0]*mag_avg[mag_avg < mstar] + coeffs0[1]
            dm_std[mag_avg >= mstar] = func_i(mag_avg[mag_avg >= mstar], *coeffs1)
            dm_std = np.max([dm_std, np.zeros_like(dm_std)], axis=0)

            outlier_mask = np.abs(dm_use) > 3*dm_std
            # sort_ind = np.argsort(mag_avg)
            
            
            # plt.scatter(mag_avg[~outlier_mask], dm_use[~outlier_mask], s=1, color='k', alpha=0.1)
            # plt.scatter(mag_avg[outlier_mask], dm_use[outlier_mask], s=1, color='r', alpha=0.1)
            # plt.scatter(mavg_vals_fit, binned_dm_std, s=15, color='g')
            # plt.plot(mag_avg[sort_ind], dm_std[sort_ind], color='orange', lw=2, ls='--')            
            # plt.plot(mag_avg[sort_ind], 3*dm_std[sort_ind], color='orange', lw=2, ls='--')
            # plt.plot(mag_avg[sort_ind], -3*dm_std[sort_ind], color='orange', lw=2, ls='--')
            # plt.xlim(19, 30)
            # plt.ylim(-.3, .3)
            # plt.savefig('test_dmstd_{}_aper{}_{}.png'.format(bands[i], j+1, difftype), dpi=150)
            # plt.cla()
            # plt.clf()
            # plt.close()



            binned_dm_mean = np.full_like(mag_centers, np.nan)
            binned_dm_std = np.full_like(mag_centers, np.nan)
            binned_dmerr_med = np.full_like(mag_centers, np.nan)
            nobj = np.zeros_like(mag_centers, dtype=int)
            for k in range(len(mag_bins)-1):
                mask = (mag_avg > mag_bins[k]) & (mag_avg <= mag_bins[k+1]) & ~outlier_mask
                nobj[k] = np.sum(mask)

                if nobj[k] < nmin:
                    continue

                mean = np.nanmean(dm_use[mask])
                std = np.nanstd(dm_use[mask])

                med = np.nanmedian(dmerr_use[mask])

                binned_dm_mean[k] = mean
                binned_dm_std[k] = std
                binned_dmerr_med[k] = med

            nanmask = np.isnan(binned_dm_mean)

            #Fit dm_mean and dmerr_med and dm_std
            mstar_dmerr_med, coeffs0_dmerr_med, coeffs1_dmerr_med = fit_for_mean_curve(mag_centers[~nanmask], binned_dmerr_med[~nanmask], np.ones((~nanmask).sum()), mag_centers, mstar=26., slope_max=max_slope)
            mstar_dm_mean, coeffs0_dm_mean, coeffs1_dm_mean = fit_for_mean_curve(mag_centers[~nanmask], binned_dm_mean[~nanmask], np.ones((~nanmask).sum()), mag_centers, mstar=26., slope_max=max_slope)  
            mstar_dm_std, coeffs0_dm_std, coeffs1_dm_std = fit_for_mean_curve_posslope(mag_centers[~nanmask], binned_dm_std[~nanmask], np.ones((~nanmask).sum()), mag_centers, mstar=mstar_init, slope_max=max_slope)
            
            
            # dm_std = np.zeros_like(dm_use)
            # dm_std[mag_avg < mstar_dm_std] = coeffs0_dm_std[0]*mag_avg[mag_avg < mstar_dm_std] + coeffs0_dm_std[1]
            # func_i = partial(fitfunc_large_poly, mstar=mstar_dm_std, alpha=coeffs0_dm_std[0], beta=coeffs0_dm_std[1])
            # dm_std[mag_avg >= mstar_dm_std] = func_i(mag_avg[mag_avg >= mstar_dm_std], *coeffs1_dm_std)
            # dm_std = np.max([dm_std, np.zeros_like(dm_std)], axis=0)
            
            # outlier_mask = np.abs(dm_use) > 3*dm_std
            # sort_ind = np.argsort(mag_avg)
            
            
            # plt.scatter(mag_avg[~outlier_mask], dm_use[~outlier_mask], s=1, color='k', alpha=0.1)
            # plt.scatter(mag_avg[outlier_mask], dm_use[outlier_mask], s=1, color='r', alpha=0.1)
            # plt.scatter(mag_centers, binned_dm_std, s=15, color='g')
            # plt.plot(mag_avg[sort_ind], dm_std[sort_ind], color='orange', lw=2, ls='--')            
            # plt.plot(mag_avg[sort_ind], 3*dm_std[sort_ind], color='orange', lw=2, ls='--')
            # plt.plot(mag_avg[sort_ind], -3*dm_std[sort_ind], color='orange', lw=2, ls='--')
            # plt.xlim(19, 30)
            # plt.ylim(-.3, .3)
            # plt.savefig('test_dmstd_{}_aper{}_{}.png'.format(bands[i], j+1, difftype), dpi=150)
            # plt.cla()
            # plt.clf()
            # plt.close()
            
            
            
            
            #Add to tables
            if j == 0:
                aper_j = 'AUTO'
            else:
                aper_j = '{:.1f}'.format(apers[j-1])
                
                
            coeffs1_all = get_extra_coeffs_large_poly(coeffs1[0], coeffs1[1], coeffs1[2], coeffs1[3], mstar, coeffs0[0], coeffs0[1])
            coeffs1_dm_mean_all = get_extra_coeffs_large_poly(coeffs1_dm_mean[0], coeffs1_dm_mean[1], coeffs1_dm_mean[2], coeffs1_dm_mean[3], mstar_dm_mean, coeffs0_dm_mean[0], coeffs0_dm_mean[1])
            coeffs1_dmerr_med_all = get_extra_coeffs_large_poly(coeffs1_dmerr_med[0], coeffs1_dmerr_med[1], coeffs1_dmerr_med[2], coeffs1_dmerr_med[3], mstar_dmerr_med, coeffs0_dmerr_med[0], coeffs0_dmerr_med[1])

            rows_dm_std.append([bands[i], aper_j, mstar_dm_std, coeffs0, coeffs1_all])
            rows_dm_mean.append([bands[i], aper_j, mstar_dm_mean, coeffs0_dm_mean, coeffs1_dm_mean_all])
            rows_dmerr_med.append([bands[i], aper_j, mstar_dmerr_med, coeffs0_dmerr_med, coeffs1_dmerr_med_all])        
            
            
            
            
            ######################################################################################################################################################
            ######################################################################################################################################################
            # Unbias the dm values and do it again
            
            bias = np.zeros_like(dm_use)
            bias[mag_avg < mstar_dm_mean] = coeffs0_dm_mean[0] * dm_use[mag_avg < mstar_dm_mean] + coeffs0_dm_mean[1]
            
            func_i = partial(fitfunc_large_poly, mstar=mstar_dm_mean, alpha=coeffs1_dm_mean[0], beta=coeffs1_dm_mean[1])
            bias[mag_avg >= mstar_dm_mean] = func_i(mag_avg[mag_avg >= mstar_dm_mean], *coeffs1_dm_mean_all[:-2])
            
            dm_unbias = dm_use - bias
            

            mavg_vals_fit = []
            binned_dm_std = []
            binned_dmcat_std = []
            for k in range(len(mag_bins)-1):
                mask = (mag_avg > mag_bins[k]) & (mag_avg <= mag_bins[k+1])
                nobj = np.sum(mask)
                
                if nobj < nmin:
                    continue    
                        
                _, _, std = sigma_clipped_stats(dm_unbias[mask], sigma=3., maxiters=None)
                
                mavg_vals_fit.append(mag_centers[k])
                binned_dm_std.append(std)
            
            
            mavg_vals_fit = np.array(mavg_vals_fit)
            binned_dm_std = np.array(binned_dm_std)
            
            
            mstar, coeffs0, coeffs1 = fit_for_sigma_curve(mavg_vals_fit, binned_dm_std, mag_centers, mstar=mstar_init, slope_max=max_slope)   
            func_i = partial(fitfunc, mstar=mstar, alpha=coeffs0[0], beta=coeffs0[1])     

            
            outlier_mask = np.zeros_like(mag_avg, dtype=bool)
            for k in range(len(outlier_mask)):
                if mag_avg[k] < mstar:
                    yline = coeffs0[0]*mag_avg[k] + coeffs0[1]
                    yline = np.max([yline, 0.])
                    
                    outlier_mask[k] = np.abs(dm_unbias[k]) > 3*yline
                else:
                    yline = func_i(mag_avg[k], *coeffs1)
                    yline = np.max([yline, 0.])
                    
                    outlier_mask[k] = np.abs(dm_unbias[k]) > 3*yline

            
            binned_dm_mean = np.full_like(mag_centers, np.nan)
            binned_dm_std = np.full_like(mag_centers, np.nan)
            binned_dmerr_med = np.full_like(mag_centers, np.nan)
            nobj = np.zeros_like(mag_centers, dtype=int)
            for k in range(len(mag_bins)-1):
                mask = (mag_avg > mag_bins[k]) & (mag_avg <= mag_bins[k+1]) & ~outlier_mask
                nobj[k] = np.sum(mask)

                if nobj[k] < nmin:
                    continue

                mean = np.nanmean(dm_unbias[mask])
                std = np.nanstd(dm_unbias[mask])

                med = np.nanmedian(dmerr_use[mask])

                binned_dm_mean[k] = mean
                binned_dm_std[k] = std
                binned_dmerr_med[k] = med
                
            nanmask = np.isnan(binned_dm_mean)
            
            try:
                #Fit dm_mean and dmerr_med and dm_std
                mstar_dm_mean, coeffs0_dm_mean, coeffs1_dm_mean = fit_for_mean_curve(mag_centers[~nanmask], binned_dm_mean[~nanmask], np.ones((~nanmask).sum()), mag_centers, mstar=26., slope_max=max_slope)  
                mstar_dmerr_med, coeffs0_dmerr_med, coeffs1_dmerr_med = fit_for_mean_curve(mag_centers[~nanmask], binned_dmerr_med[~nanmask], np.ones((~nanmask).sum()), mag_centers, mstar=26., slope_max=max_slope)
                mstar_dm_std, coeffs0_dm_std, coeffs1_dm_std = fit_for_sigma_curve(mag_centers[~nanmask], binned_dm_std[~nanmask], mag_centers, mstar=26., slope_max=max_slope)
                
                #Add to tables
                if j == 0:
                    aper_j = 'AUTO'
                else:
                    aper_j = '{:.1f}'.format(apers[j-1])
                    
                    
                coeffs1_dm_std_all = get_extra_coeffs(coeffs1_dm_std[0], coeffs1_dm_std[1], mstar_dm_std, coeffs0_dm_std[0], coeffs0_dm_std[1])
                coeffs1_dm_mean_all = get_extra_coeffs_large_poly(coeffs1_dm_mean[0], coeffs1_dm_mean[1], coeffs1_dm_mean[2], coeffs1_dm_mean[3], mstar_dm_mean, coeffs0_dm_mean[0], coeffs0_dm_mean[1])
                coeffs1_dmerr_med_all = get_extra_coeffs_large_poly(coeffs1_dmerr_med[0], coeffs1_dmerr_med[1], coeffs1_dmerr_med[2], coeffs1_dmerr_med[3], mstar_dmerr_med, coeffs0_dmerr_med[0], coeffs0_dmerr_med[1])

                rows_dm_std_unbias.append([bands[i], aper_j, mstar_dm_std, coeffs0_dm_std, coeffs1_dm_std_all])
                rows_dm_mean_unbias.append([bands[i], aper_j, mstar_dm_mean, coeffs0_dm_mean, coeffs1_dm_mean_all])
                rows_dmerr_med_unbias.append([bands[i], aper_j, mstar_dmerr_med, coeffs0_dmerr_med, coeffs1_dmerr_med_all])
            
            except:
                continue
            
        
        
    colnames = ['BAND', 'APER', 'MSTAR', 'LINEAR_COEFFS', 'POLY_COEFFS']

    corrdat_dm_mean = Table(rows=rows_dm_mean, names=colnames)
    corrdat_dm_std = Table(rows=rows_dm_std, names=colnames)
    corrdat_dmerr_med = Table(rows=rows_dmerr_med, names=colnames)
    
    corrdat_dm_mean_unbias = Table(rows=rows_dm_mean_unbias, names=colnames)
    corrdat_dm_std_unbias = Table(rows=rows_dm_std_unbias, names= colnames)
    corrdat_dmerr_med_unbias = Table(rows=rows_dmerr_med_unbias, names=colnames)
    
    return corrdat_dm_mean, corrdat_dm_std, corrdat_dmerr_med, corrdat_dm_mean_unbias, corrdat_dm_std_unbias, corrdat_dmerr_med_unbias



if __name__ == '__main__':
    names = ['wide', 'deep']
    epochs = ['01', '01']
    name_prefix = '{}{}_{}{}'.format(names[0], epochs[0], names[1], epochs[1])
    
    figdir = '/data6/stone28/nexus/nexus_{}_nuclear_variability_figs/'.format(name_prefix)
    maindir = '/data6/stone28/nexus/'
    outdir = maindir + 'correction_curves_{}/'.format(name_prefix)

    dm_mean_all = []
    dm_std_all = []
    dmerr_med_all = []
    dm_mean_unbias_all = []
    dm_std_unbias_all = []
    dmerr_med_unbias_all = []


    dat_f200w_stacked = Table.read(maindir + 'nexus_{}_stacked_sources_F200W.fits'.format(name_prefix))
    dat_f444w_stacked = Table.read(maindir + 'nexus_{}_stacked_sources_F444W.fits'.format(name_prefix))

    #Combine
    dat_all = [dat_f200w_stacked, dat_f444w_stacked]

    for dt in ['FINAL', 'SUB', 'SFFT', 'CAT']:

        cdat_dm_mean, cdat_dm_std, cdat_dmerr_med, cdat_dm_mean_unbias, cdat_dm_std_unbias, cdat_dmerr_med_unbias = unbias_dm(dat_all, suffix='', nmin=50, max_slope=1e-3, dmag=0.5,
                                                                                                                              remove_flag=False, remove_saturated=False, remove_elongated=False, max_magdiff=35,
                                                                                                                              difftype=dt)

        cdat_dm_mean['DIFF_TYPE'] = dt
        cdat_dm_std['DIFF_TYPE'] = dt
        cdat_dmerr_med['DIFF_TYPE'] = dt
        cdat_dm_mean_unbias['DIFF_TYPE'] = dt
        cdat_dm_std_unbias['DIFF_TYPE'] = dt
        cdat_dmerr_med_unbias['DIFF_TYPE'] = dt
        
        dm_mean_all.append(cdat_dm_mean)
        dm_std_all.append(cdat_dm_std)
        dmerr_med_all.append(cdat_dmerr_med)
        dm_mean_unbias_all.append(cdat_dm_mean_unbias)
        dm_std_unbias_all.append(cdat_dm_std_unbias)
        dmerr_med_unbias_all.append(cdat_dmerr_med_unbias)


    cdat_dm_mean_all = vstack(dm_mean_all, join_type='exact')
    cdat_dm_std_all = vstack(dm_std_all, join_type='exact')
    cdat_dmerr_med_all = vstack(dmerr_med_all, join_type='exact')
    cdat_dm_mean_unbias_all = vstack(dm_mean_unbias_all, join_type='exact')
    cdat_dm_std_unbias_all = vstack(dm_std_unbias_all, join_type='exact')
    cdat_dmerr_med_unbias_all = vstack(dmerr_med_unbias_all, join_type='exact')
    
    cdat_dm_mean_all.write(outdir + 'corr_dm_mean.fits', overwrite=True)
    cdat_dm_std_all.write(outdir + 'corr_dm_std.fits', overwrite=True)
    cdat_dmerr_med_all.write(outdir + 'corr_dmerr_med.fits', overwrite=True)
    cdat_dm_mean_unbias_all.write(outdir + 'corr_dm_mean_unbias.fits', overwrite=True)
    cdat_dm_std_unbias_all.write(outdir + 'corr_dm_std_unbias.fits', overwrite=True)
    cdat_dmerr_med_unbias_all.write(outdir + 'corr_dmerr_med_unbias.fits', overwrite=True)