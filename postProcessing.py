#!/usr/bin/python3
"""
================================================================================
Run post processing on all subjects. The post processing steps include:
1. Extract time series of each ROI according to power atlas 
2. Remove first 16 time points of each run
3. Combine all the runs together
4. Performe scrubbing of time points according to the following steps:
	4.1 Calculate FD for each time point
	4.2 Remove time points with FD >= 0.2
	4.3 Save only 5 and above continues time points
	4.4 Calculate standart deviation of each time point
	4.5 Calculate MAD (median absolute deviation) of all the sd values
	4.6 Remove outliers- time points with sd > MAD + 3*MAD and sd < MAD - 3*MAD
5. Calculate covariance matrix using LedoitWolf algorithm
6. Calculate correlation matrix
================================================================================

"""

import nilearn
import os, sys
import pickle
from nilearn import datasets
from nilearn import input_data
import numpy as np
from sklearn.covariance import LedoitWolf
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
from nilearn.input_data import NiftiMapsMasker
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from scipy import stats

start = time.time()
allParticipantsDic = {}


def perform_scrubbing(motion_files, time_series):
    """Perform scrubbing according to ABCD paper
    compute FD 
	volumes with FD greater than 0.2 mm are excluded. Time
	periods with fewer than five contiguous, sub-threshold time points are also excluded.
    """
    fd_all = []
    # For each run calculate the FD and concatenate all runs to one array 
    for motion_file in motion_files:
	    fd_all.append(compute_fd(motion_file))
    fd_new = np.concatenate(fd_all)
    # print("fd_new: \n",fd_new)
    # print("fd_new_shape: \n",fd_new.shape)
    fd_thresh = 0.2
	#Left only the indexes in the FD array that the FD value is less than 0.2 
    left_indexes = np.nonzero(fd_new <= fd_thresh)[0]
    # print("Number of values less or equal than 0.2 =", fd_new[fd_new <= fd_thresh].size)
    # print("Their indices are ", left_indices)
	#Leave only five contiguous points
    count_five = 1
    prev_index = 0
    left_indexes_copy = left_indexes
	# The final indexes after censoring
    final_indexes = np.array([]).astype(int)
    for i in left_indexes_copy[1:]:
	    #check if contiguous points
        if ((i-prev_index) == 1):
            count_five = count_five + 1
        else:
            if(count_five >= 5): 
			    #save the contiguous indexes
                final_indexes = np.append(final_indexes,left_indexes[:count_five])
            #cut the indexes already passed 
            left_indexes = left_indexes[count_five:]
            #start to count from beginning			
            count_five = 1
        prev_index = i
    #insert the last 5 continues indexes
    if(count_five >=5):
        final_indexes = np.append(final_indexes,left_indexes[:count_five])
    sd_left_indexes = compute_sd(final_indexes,time_series)
    #print("sd_left_indexes : ", sd_left_indexes)
    #print("sd_left_indexes size:", sd_left_indexes.size)
    #print("final_indexes_after_fd : ", final_indexes)
    #print("final_indexes_after_fd size:", final_indexes.size)
    final_indexes = np.intersect1d(final_indexes,sd_left_indexes)
    #print("final_indexes_after_sd : ", final_indexes)
    #print("final_indexes_after_sd size:", final_indexes.size)
    return (final_indexes)
    
def postProcessing(rs_files, motion_files, subject_folder):
    try:
        print ("subject_folder: " + subject_folder) 
        print("Extract timeseries")  
        timeseries_all = []
        for rs_file in rs_files:
		    #Remove the first 16 volumes
            timeseries = spheres_masker.fit_transform(rs_file, confounds=None)
            timeseries = np.delete(timeseries, range(16), axis = 0)
		    #Combine all runs
            timeseries_all.append(timeseries)
        timeseries_new = np.concatenate(timeseries_all)
        timeseries_censored = timeseries_new[perform_scrubbing(motion_files, timeseries_new)]
        num_of_left_volumes = timeseries_censored.shape[0]
        print("timeseries_censored left volumes: ", num_of_left_volumes)	
        print("Plot timeseries after censoring")
        fig3 = plt.figure()
        plt.plot(timeseries_censored.T[2])
        plt.plot(timeseries_censored.T[0])
        plt.plot(timeseries_censored.T[5])
        plt.title('Time Series')
        plt.xlabel('Scan number')
        plt.ylabel('Normalized signal')
        plt.legend()
        plt.tight_layout()
        fig3.savefig(os.path.abspath(os.path.join(subject_folder,"timeSeries_after_censoring.png")))
        print("Extract and plot covariance matrix after censoring")
        cov_measure = ConnectivityMeasure(cov_estimator=LedoitWolf(assume_centered=False, block_size=1000, store_precision=False), kind='covariance')
        cov = cov_measure.fit_transform([timeseries_censored])
        fig4 = plt.figure()
        plotting.plot_matrix(cov[0, :, :], colorbar=True, figure=fig4, title='Power covariance matrix')
        fig4.savefig(os.path.abspath(os.path.join(subject_folder, "cov_matrix_after_censoring.png")))
        print("Extract and plot correlation matrix")
        cor = nilearn.connectome.cov_to_corr(cov[0, :, :])
        fig5 = plt.figure()
        plotting.plot_matrix(cor, colorbar=True, figure=fig5, vmin=-1., vmax=1., title='Power correlation matrix')
        fig5.savefig(os.path.abspath(os.path.join(subject_folder, "cor_matrix_after_censoring.png")))
    except:
        raise Exception( "subject_folder: %s \n" % subject_folder + str(sys.exc_info()[1])).with_traceback(sys.exc_info()[2])
    return (subject_folder, {"time_series" : timeseries_new, "covariance" : cov[0, :, :], "correlation" : cor, "num_of_volumes" : timeseries_censored.shape[0]})
	

def upload_motion_derivatives(motion_file):
    motion_array = []
    with open(motion_file, 'r') as f:
        for line in f.readlines():
            line = " ".join(line.split())
            motion_array.append([float(value) for value in line.split(' ')])
    #Remove first 16 time points
    motion_array = np.delete(motion_array, range(16),axis = 0)
    return motion_array
	
def compute_fd(motion_file):
    #delete#
    # motion_array = np.array(upload_motion_derivatives(motion_file))
    # motion_array[1:,:]=np.abs(motion_array[1:,:] - motion_array[:-1,:])
    # headradius = 50
    # motion_array[:,0:3] = np.pi*headradius*2*(motion_array[:,0:3]/360)
	#################
	#original#
    motion_array = np.abs(upload_motion_derivatives(motion_file))
    headradius = 50
    motion_array[:,0:3] = np.pi*headradius*2*(motion_array[:,0:3]/360)
	
	##################
    fd=np.sum(motion_array,1)
    return fd
	
def compute_sd(fd_keep_indexes,time_series):
    std_array = np.std(time_series, axis=1)
    #print ("std_array: ", std_array)
    #print ("std_array.shape: ", std_array.shape)
    mad = stats.median_absolute_deviation(std_array[fd_keep_indexes])
    #print(mad)
    #print(mad - (mad*3))
    #print(mad + (mad*3))
    mask = (std_array >= (mad - (mad*3))) & (std_array <= (mad + (mad*3)))
    #print("mask: ", mask)
    #print("mask shape: ", mask.shape)
    left_indexes = np.nonzero(mask)[0]
    return (left_indexes)
	    
	

def collect_results(result):
    min_num_of_volumes = 375 # 5 minutes scan
    if(result != None):
        if(result[1]["num_of_volumes"] >= min_num_of_volumes):
            allParticipantsDic[result[0]] = result[1]
        else:
            excluded_subject_file.write(result[0])
            excluded_subject_file.write("\n")
			
def collect_errors(exception):
    """ Callback for errors collecting from threads. Get the exception and write to file
    param exception: Exception. The exception that was raised
    """
    error_file.write(str(exception))
    error_file.write("\n\n\n")

#check input
if len(sys.argv) <3:
    print("Insert preprocessed folder and path to Power coordinates")
    exit()
	
preproc_folder = sys.argv[1]
power_coords = sys.argv[2]

excluded_subject_file = open(preproc_folder + "/excluded_subject.txt", "w+")
#open the error and out files
error_file = open(preproc_folder + "/error_file_postProcessing.txt", "w+")

#Extract rois
mniCoordsFile = open(power_coords,"rb")
splitedLines = [ ]
coords = []
for line in mniCoordsFile.read().splitlines():
    splitedLine = line.decode().split(' ')
    newCoord = []
    for part in splitedLine:
        if part is not '':
            newCoord.append(float(part))
    coords.append(newCoord)
mniCoordsFile.close()

#create mask according to the extracted rois
#seeds: List of coordinates of the seeds in the same space as the images (typically MNI or TAL).
#radius: Indicates, in millimeters, the radius for the sphere around the seed. Default is None (signal is extracted on a single voxel).
#smoothing_fwhm: If smoothing_fwhm is not None, it gives the full-width half maximum in millimeters of the spatial smoothing to apply to the signal.
#standardize: If standardize is True, the time-series are centered and normed: their mean is set to 0 and their variance to 1 in the time dimension
#detrend, low_pass, high_pass and t_r are passed to signal.clean function. This function improve the SNR on masked fMRI signals.
spheres_masker = input_data.NiftiSpheresMasker(
    seeds=coords, radius=10.,allow_overlap=True,
    detrend=True, standardize=True, low_pass=0.08, high_pass=0.009 , t_r=0.8)


# Init multiprocessing.Pool()
pool = mp.Pool(15)
i=0
for subject_folder in os.listdir(preproc_folder):
    rs_files=[]
    motion_files=[]
    subject_folder = os.path.join(preproc_folder,subject_folder)
    if(os.path.isdir(subject_folder)):
        for root, dirs, files in os.walk(subject_folder):
            for sub_folder in dirs:
                rs_folder = os.path.abspath(os.path.join(preproc_folder,subject_folder,sub_folder))
                motion_files.append(os.path.abspath(os.path.join(rs_folder, "motionparameters_filtered.backdif.1D")))
                rs_files.append(os.path.abspath(os.path.join(rs_folder, "fmri_rpi_aligned_nonlinear_denoising_sm.nii.gz")))
        [pool.apply_async(postProcessing, args=(rs_files,motion_files,subject_folder,),callback=collect_results, error_callback = collect_errors)]
pool.close() 
pool.join()


print("Write to pkl file")
f = open(preproc_folder + '/matAndTimeSerias.pkl', mode="wb")
pickle.dump(allParticipantsDic, f)
f.close()

excluded_subject_file.close()

#print the exucation time
end = time.time()
timeInSeconds = (end - start)
timeInMinutes = timeInSeconds / 60
timeInHours = int(timeInMinutes / 60)
print ("Total time : " + str(timeInHours) + " hours and " + str(int(timeInMinutes % 60)) + " minutes")	
