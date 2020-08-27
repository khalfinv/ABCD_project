#!/usr/bin/python3
"""
================================================================================
Run post processing on all subjects. The post processing steps include:
1. Extract time series of each ROI according to atlas 
2. Calculate covariance matrix using LedoitWolf algorithm
3. Calculate correlation matrix
================================================================================

@Input:  
preproc_folder = path to folder containing all the preprocessed scans
coords = path to text file with all the atlas coordinates

@Output:
error_file_postProcessing.txt file: located in the preproc_folder. This file contains all the
errors occured during post processing. 
subjects_data.pkl file: located in the preproc_folder. This file contains dictionary with all the subjects data as following:
key: subject's key . 
value: {"time_series" : matrix of time series  (time_points, rois), "covariance" : covariance matrix of atlas rois (rois, rois),
 "correlation" : correlation matrix of atlas rois (rois, rois)}

"""

import nilearn
import os, sys, argparse
import pickle
from nilearn import input_data
import numpy as np
from sklearn.covariance import LedoitWolf
from nilearn.input_data import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
import multiprocessing as mp
import traceback



   
def postProcessing(nifti_file, subject_key, spheres_masker):
    """Perform post processing
	param nifti_file: string. path to the nifty file
    param subject_key: string. subject's key
	return: dictionary raw. 
		key: subject's key . 
		value: {"time_series" : matrix of time series (time_points,rois), "covariance" : covariance matrix of atlas rois (rois, rois),
			"correlation" : correlation matrix of atlas rois (rois, rois)}
    """
    try:
        print ("subject_key: " + subject_key) 
        print("Extract timeseries")  
		# Extract the time series
        print(nifti_file)
        timeseries = spheres_masker.fit_transform(nifti_file, confounds=None)
        print("Extract covariance matrix")
        cov_measure = ConnectivityMeasure(cov_estimator=LedoitWolf(assume_centered=False, block_size=1000, store_precision=False), kind='covariance')
        cov = []
        cor = []
        cov = cov_measure.fit_transform([timeseries])[0, :, :]
        print("Extract correlation matrix")
        cor = nilearn.connectome.cov_to_corr(cov)
    except:
		
        raise Exception( "subject_key: %s \n" % subject_key + traceback.format_exc())
    return (subject_key, {"time_series" : timeseries, "covariance" : cov, "correlation" : cor})
	
	
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
    allParticipantsDic = {}
    parser = argparse.ArgumentParser()
    parser.add_argument('--preproc_folder', required=True, type=str, help='path to folder containing all the preprocessed scans')
    parser.add_argument('--atlas_coords', required=True, type=str, help='path to text file  with all the atlas coordinates')
    parser.add_argument('--out_folder', required=True, type=str, help='path to output folder')
    args = parser.parse_args()
        
    preproc_folder = args.preproc_folder
    atlas_coords = args.atlas_coords
    out_folder = args.out_folder

    #open the error and out files
    error_file = open(os.path.join(out_folder, "error_file_postProcessing.txt"), mode="w")

        
    #Extract rois
    mniCoordsFile = open(atlas_coords,"rb")
    splitedLines = [ ]
    coords = []
    for line in mniCoordsFile.read().splitlines():
        splitedLine = line.decode().split()
        newCoord = []
        for part in splitedLine:
            if part != '':
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
    num_of_cpu = mp.cpu_count()
    if (num_of_cpu > 4):
        num_of_cpu = 4
    pool = mp.Pool(num_of_cpu)
    # For each subject run the postprocessing steps
    subject_key = 0
    for file in os.listdir(preproc_folder):
        subject_key = subject_key + 1
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            full_file = os.path.abspath(os.path.join(preproc_folder, file))
            [pool.apply_async(postProcessing, args=(full_file,str(subject_key),spheres_masker,),callback=collect_results, error_callback = collect_errors)]
    pool.close() 
    pool.join()

    #write the allParticipantsDic to pkl file
    print("Write to pkl file")

    f = open(os.path.join(out_folder, "subjects_data.pkl"), mode="wb")
    pickle.dump(allParticipantsDic, f)
    f.close()

    #close the files
    error_file.close() 


