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
5. Calculate covariance matrix using LedoitWolf algorithm
6. Calculate correlation matrix
================================================================================

@Input:  
preproc_folder = path to folder containing all the preprocessed scans
power_coords = path to MNI_power.txt with all the power coordinates
use_prev = boolean. Indicates to use the previous dictionary and add only new subjects.

@Output:
error_file_postProcessing.txt file: located in the preproc_folder. This file contains all the
errors occured during post processing. 
subjects_data.pkl file: located in the preproc_folder. This file contains dictionary with all the subjects data as following:
key: subject's key . 
value: {"time_series" : matrix of time series after censoring (power_rois,time_points), "covariance" : covariance matrix of power rois (power_rois, power_rois),
 "correlation" : correlation matrix of power rois (power_rois, power_rois), "num_of_volumes" : num of volumes left after censoring}

"""

import nilearn
import os, sys, argparse
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


def perform_scrubbing(motion_files):
    """Perform scrubbing according to ABCD paper
    compute FD 
	volumes with FD greater than 0.2 mm are excluded. 
	Time periods with fewer than five contiguous, sub-threshold time points are also excluded.
	param motion_files: list. list of motion files paths. 
	return: final_indexes : ndarray
			array of left indexes after scrubbing
    """
    fd_all = []
    # For each run calculate the FD and concatenate all runs to one array 
    for motion_file in motion_files:
	    fd_all.append(compute_fd(motion_file))
    fd_new = np.concatenate(fd_all)
    fd_thresh = 0.2
	#Left only the indexes in the FD array that the FD value is less than 0.2 
    left_indexes = np.nonzero(fd_new <= fd_thresh)[0]
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
    return (final_indexes)
    
def postProcessing(rs_files, motion_files, subject_folder, spheres_masker):
    """Perform post processing
	param rs_files: list. rs files paths 
    param motion_files: list. motion files paths
    param subject_folder: string. full path of the subject's folder
	return: dictionary raw. 
		key: subject's key . 
		value: {"time_series" : matrix of time series after censoring (time_points, power_rois), "covariance" : covariance matrix of power rois (power_rois, power_rois),
			"correlation" : correlation matrix of power rois (power_rois, power_rois), "num_of_volumes" : num of volumes left after censoring}
    """
    try:
        print ("subject_folder: " + subject_folder) 
        print("Extract timeseries")  
        timeseries_all = []
        for rs_file in rs_files:
		    # Extract the time series
            timeseries = spheres_masker.fit_transform(rs_file, confounds=None)
            #Remove the first 16 volumes
            timeseries = np.delete(timeseries, range(16), axis = 0)
		    #Combine to other runs
            timeseries_all.append(timeseries)
        timeseries_new = np.concatenate(timeseries_all)
        timeseries_censored = timeseries_new[perform_scrubbing(motion_files)]
        num_of_left_volumes = timeseries_censored.shape[0]
        print("Number of left volumes after censoring: ", num_of_left_volumes)	
        print("Extract covariance matrix")
        cov_measure = ConnectivityMeasure(cov_estimator=LedoitWolf(assume_centered=False, block_size=1000, store_precision=False), kind='covariance')
        cov = cov_measure.fit_transform([timeseries_censored])
        print("Extract correlation matrix")
        cor = nilearn.connectome.cov_to_corr(cov[0, :, :])
    except:
        raise Exception( "subject_folder: %s \n" % subject_folder + str(sys.exc_info()[1])).with_traceback(sys.exc_info()[2])
    #get only the subject key from full path
    subject_key = subject_folder.split("\\")[-1]
    return (subject_key, {"time_series" : timeseries_censored, "covariance" : cov[0, :, :], "correlation" : cor, "num_of_volumes" : timeseries_censored.shape[0]})
	

def upload_motion_derivatives(motion_file):
    """Upload the motion's derivatives file, delete the first 16 volumes
     	and create an array with all the values.
	param motion_file: string. path to the motion file
	return: motion_array : ndarray. 
		two dimentional array (number of time points, 6)
    """
    motion_array = []
    with open(motion_file, 'r') as f:
        for line in f.readlines():
            line = " ".join(line.split())
            motion_array.append([float(value) for value in line.split(' ')])
    #Remove first 16 time points
    motion_array = np.delete(motion_array, range(16),axis = 0)
    return motion_array
	
def compute_fd(motion_file):
    """Compute FD of each time point
	param motion_file: string. path to the motion's derivatives file
	return: fd : ndarray. 
		one dimentional array. The length is the number of time points  
    """
    motion_array = np.abs(upload_motion_derivatives(motion_file))
    headradius = 50
    motion_array[:,0:3] = np.pi*headradius*2*(motion_array[:,0:3]/360)
    fd=np.sum(motion_array,1)
    return fd
		    
	

def collect_results(result):
    """Collect the results from postProcessing function. 
	   Insert the result to allParticipantsDic.
	param result: dictionary raw. 
		key: subject's key . 
		value: {"time_series" : matrix of time series after censoring (time_points, power_rois), "covariance" : covariance matrix of power rois (power_rois, power_rois),
			"correlation" : correlation matrix of power rois (power_rois, power_rois), "num_of_volumes" : num of volumes left after censoring}
	return: None 
    """
    if(result != None):
        allParticipantsDic[result[0]] = result[1]
			
def collect_errors(exception):
    """ Callback for errors collecting from threads. Get the exception and write to file
    param exception: Exception. The exception that was raised
    """
    error_file.write(str(exception))
    error_file.write("\n\n\n")

	
if __name__ == "__main__":
    start = time.time()
    allParticipantsDic = {}
    parser = argparse.ArgumentParser()
    parser.add_argument('--preproc_folder', required=True, type=str, help='path to folder containing all the preprocessed scans')
    parser.add_argument('--power_coords', required=True, type=str, help='path to MNI_power.txt with all the power coordinates')
    parser.add_argument('--use_prev', help='use the previous dictionary and concatenate only new subjects',action='store_true')
    args = parser.parse_args()
		
    preproc_folder = args.preproc_folder
    power_coords = args.power_coords

    #open the error and out files
    error_file = open(preproc_folder + "/error_file_postProcessing.txt", "w+")

	#if use_prev is true, open the previous pkl file and upload dictionary
    if args.use_prev == True:
        print("Read from previous pkl file")
        pkl_file = open(preproc_folder + '/subjects_data.pkl', 'rb')
        allParticipantsDic = pickle.load(pkl_file)
        pkl_file.close()
		
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
    pool = mp.Pool(5)
    i=0
    # For each subject run the postprocessing steps
    for subject_folder in os.listdir(preproc_folder):
        rs_files=[]
        motion_files=[]
        if (subject_folder not in allParticipantsDic):
	        subject_folder = os.path.join(preproc_folder,subject_folder)
	        if(os.path.isdir(subject_folder)):
		        for root, dirs, files in os.walk(subject_folder):
			        for sub_folder in dirs:
				        rs_folder = os.path.abspath(os.path.join(preproc_folder,subject_folder,sub_folder))
				        motion_files.append(os.path.abspath(os.path.join(rs_folder, "motionparameters_filtered.backdif.1D")))
				        rs_files.append(os.path.abspath(os.path.join(rs_folder, "fmri_rpi_aligned_nonlinear_denoising_sm.nii.gz")))
		        [pool.apply_async(postProcessing, args=(rs_files,motion_files,subject_folder,spheres_masker,),callback=collect_results, error_callback = collect_errors)]
    pool.close() 
    pool.join()

    #write the allParticipantsDic to pkl file
    print("Write to pkl file")
    f = open(preproc_folder + '/subjects_data.pkl', mode="wb")
    pickle.dump(allParticipantsDic, f)
    f.close()

	#close the files
    error_file.close() 
	
	#print the exucation time
    end = time.time()
    timeInSeconds = (end - start)
    timeInMinutes = timeInSeconds / 60
    timeInHours = int(timeInMinutes / 60)
    print ("Total time : " + str(timeInHours) + " hours and " + str(int(timeInMinutes % 60)) + " minutes")	
