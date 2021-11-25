""" Numpy """
import numpy as np

"""Pandas"""
import pandas as pd

"""Matplotlib"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.units as munits
import matplotlib.ticker
from   cycler import cycler
import datetime
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm  
import matplotlib as mpl

"""Seaborn"""
import seaborn as sns

""" Wavelets """
import pywt

""" Scipy """
import scipy.io
from scipy.io import savemat

"""Sort files in folder"""
import natsort

""" Load files """
from   spacepy import pycdf
import pickle
import glob
import os


""" Import manual functions """
import sys

sys.path.insert(1,r'C:\Users\nikos.000\PVI\python_scripts')
import functions2 as fun

sys.path.insert(1,r'C:\Users\nikos.000\coh_struc_dist_final\python')
import struc_func as struc_f

def plot_pretty(dpi=175,fontsize=9):
    # import pyplot and set some parameters to make plots prettier
    plt.rc("savefig", dpi=dpi)
    plt.rc("figure", dpi=dpi)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in') 
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5) 
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5) 
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [0.5, 1.1])
    return
#plot_pretty(dpi=175,fontsize=9)
plt.style.use(['science','notebook','grid'])

plot_pretty(dpi=150, fontsize=12)



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#from lightkurve.lightcurve import LightCurve as LC
#from lightkurve.search import search_lightcurve
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import batman


COLOR = 'k'#'#FFFAF1'
plt.rcParams['font.size'] = 25
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR

plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.major.size']  = 6 #12
plt.rcParams['ytick.major.size']  = 6 #12

plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.minor.width'] = 2
plt.rcParams['xtick.minor.size']  = 4
plt.rcParams['ytick.minor.size']  = 4

#plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Comic Sans MS"
plt.rcParams['axes.linewidth'] = 2




def clean_SOLO_particles(environment,year,  replace, window_size, desired_min, target_path,save_path):
    os.environ['CDF_LIB'] = environment
    for kk in range(len(year)):
        target_path_final   = target_path+"\_"+str(year[kk])
        file_names  = glob.glob(target_path_final+os.sep+'*.cdf')                           # file names
        file_names  = natsort.natsorted(file_names)
        for i in range(len(file_names)):
            #""" Load magnetic field data """
            particles= pycdf.CDF(file_names[i])


            """ Clean particle data """
            dt     = particles['EPOCH'][:]
            vx     = particles['SWA_PAS_VELOCITY'][:].T[0]
            vy     = particles['SWA_PAS_VELOCITY'][:].T[1]
            vz     = particles['SWA_PAS_VELOCITY'][:].T[2]

            vx    = fun.nan_removal(vx, replace,window_size,desired_min)
            vy    = fun.nan_removal(vy, replace,window_size,desired_min)
            vz    = fun.nan_removal(vz, replace,window_size,desired_min)

            vx    = fun.hampel_filter_forloop_numba(vx, window_size, n_sigmas=3)
            vy    = fun.hampel_filter_forloop_numba(vy, window_size, n_sigmas=3)
            vz    = fun.hampel_filter_forloop_numba(vz, window_size, n_sigmas=3)

            """ Create df """

            V =pd.DataFrame({"DateTime":dt,
                            "Vr":vx[0],
                            "Vt":vy[0],
                            "Vn":vz[0]})
            V = V.set_index('DateTime')

            if i ==0:
                """Resample to a cadence of 1s"""
                V1 = V.resample('1s').nearest().interpolate(method='linear')
            else:
                """Resample to a cadence of 1s"""
                V = V.resample('1s').nearest().interpolate(method='linear')

            V1 = pd.concat([V1,V])

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        file_to_store = open(save_path+"\_"+str(year[kk])+".dat", "wb") # sto trito bale 2 opws einai twra
        pickle.dump(V1, file_to_store)
        file_to_store.close()


def clean_SOLO_magnetic_field(gap_time_threshold, resampling_time, environment, year, target_path, save_path):
    os.environ['CDF_LIB'] = environment
    for kk in range(len(year)):
        target_path_final   = target_path+"\_"+str(year[kk])
        file_names  = glob.glob(target_path_final+os.sep+'*.cdf')                           # file names
        file_names  = natsort.natsorted(file_names)
        for i in range(len(file_names)):
            #""" Load magnetic field data """
            B= pycdf.CDF(file_names[i])


            """ Clean particle data """
            df =pd.DataFrame({'DateTime': B['EPOCH'][:],
                              'Br':B['B_SRF'][:].T[0],
                              'Bt':B['B_SRF'][:].T[1],
                              'Bn':B['B_SRF'][:].T[2]})
            df =df.set_index('DateTime')
            
            ### Identify  big gaps in our timeseries ###
            f2         = df[df.Br>-1e10]
            time       = (f2.index.to_series().diff()/np.timedelta64(1, 's'))
            big_gaps   = time[time>gap_time_threshold]



            """ Create df """

            if i ==0:
                """ Resample to a cadence of 1s """
                df1 = df.resample(resampling_time).nearest().interpolate(method='linear')

                """ Now remove the gaps indentified earlier """ 
                for o in range(len(big_gaps)):
                    dt2 = big_gaps.index[o]
                    dt1 = big_gaps.index[o]-datetime.timedelta(seconds=big_gaps[o])
                    df1 = df1[(df1.index<dt1) | (df1.index>dt2) ]
            else:
                """ Resample to a cadence of 1s"""
                df = df.resample(resampling_time).nearest().interpolate(method='linear')


                """ Now remove the gaps indentified earlier """
                for o in range(len(big_gaps)):
                    dt2 = big_gaps.index[o]
                    dt1 = big_gaps.index[o]-datetime.timedelta(seconds=big_gaps[o])
                    df  = df[(df.index<dt1) | (df.index>dt2) ]
            if i%10==0:
                print(str(year[kk])+", progress "+ str(round(100*i/(len(file_names)),1))+ ' %')
            df1 = pd.concat([df1,df])

        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        file_to_store = open(save_path+"\_"+str(year[kk])+".dat", "wb") # sto trito bale 2 opws einai twra
        pickle.dump(df1, file_to_store)
        file_to_store.close()


