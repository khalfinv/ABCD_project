import nibabel as nib
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
from nitime.analysis import FilterAnalyzer
from nitime.timeseries import TimeSeries

start = time.time()
allParticipantsDic = {}
def createCovMat(rs_files, subject_folder):
    print ("subject_folder: " + subject_folder) 
    print("Extract timeseries")  
    timeseries_all = []
    for rs_file in rs_files:
        timeseries_all.append(spheres_masker.fit_transform(rs_files[0], confounds=None))
    timeseries_new = np.concatenate(timeseries_all)
	#Prior to the use of motion estimates for regression and censoring, estimated motion time
	#courses are temporally filtered using an infinite impulse response (IIR) notch filter, to attenuate
	#signals in the range of 0.31 - 0.43 Hz. This frequency range corresponds to empirically
	#observed oscillatory signals in the motion estimates that are linked to respiration and the
	#dynamic changes in magnetic susceptibility due to movement of the lungs in the range of 18.6 -
	#25.7 respirations / minute.
    # T = TimeSeries(timeseries_new, sampling_interval=0.8)
    # print ("T before FilterAnalyzer: ", T)
    # F = FilterAnalyzer(T, ub=0.43, lb=0.31)
    # print ("T after FilterAnalyzer: ", T)
    # filtered_timeseries = F.iir.data
    print("Plot timeseries")
    fig1 = plt.figure()
    plt.plot(timeseries_new.T[2])
    plt.plot(timeseries_new.T[0])
    plt.plot(timeseries_new.T[5])
    plt.title('Time Series')
    plt.xlabel('Scan number')
    plt.ylabel('Normalized signal')
    plt.legend()
    plt.tight_layout()
    fig1.savefig(os.path.abspath(os.path.join(subject_folder,"timeSeries.png")))
    print("Extract and plot covariance matrix")
    cov_measure = ConnectivityMeasure(cov_estimator=LedoitWolf(assume_centered=False, block_size=1000, store_precision=False), kind='covariance')
    cov = cov_measure.fit_transform([timeseries_new])
    fig2 = plt.figure()
    plotting.plot_matrix(cov[0, :, :], colorbar=True, figure=fig2, title='Power covariance matrix')
    fig2.savefig(os.path.abspath(os.path.join(subject_folder, "cov_matrix.png")))
    print("Extract and plot correlation matrix")
    cor = nilearn.connectome.cov_to_corr(cov[0, :, :])
    fig3 = plt.figure()
    plotting.plot_matrix(cor, colorbar=True, figure=fig3, vmin=-1., vmax=1., title='Power correlation matrix')
    fig3.savefig(os.path.abspath(os.path.join(subject_folder, "cor_matrix.png")))
    return (subject_folder, {"time_series" : timeseries_new, "covariance" : cov, "correlation" : cor})
	


def collect_results(result):
    allParticipantsDic[result[0]] = result[1]

	
#check input
if len(sys.argv) <3:
    print("Insert preprocessed folder and path to Power coordinates")
    exit()
	
preproc_folder = sys.argv[1]
power_coords = sys.argv[2]

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
    subject_folder = os.path.join(preproc_folder,subject_folder)
    for root, dirs, files in os.walk(subject_folder):
        for sub_folder in dirs:
            rs_folder = os.path.abspath(os.path.join(preproc_folder,subject_folder,sub_folder))
            rs_files.append(os.path.abspath(os.path.join(rs_folder, "fmri_rpi_aligned_nonlinear_mc_denoising_sm.nii.gz")))
    res = pool.apply_async(createCovMat, args=(rs_files,subject_folder,),callback=collect_results)
    res.get()
    #createCovMat(rs_files,subject_folder) 

pool.close() 
pool.join()


print("Write to pkl file")
f = open('covMatAndTimeSerias.pkl', mode="wb")
pickle.dump(allParticipantsDic, f)
f.close()

#print the exucation time
end = time.time()
timeInSeconds = (end - start)
timeInMinutes = timeInSeconds / 60
timeInHours = int(timeInMinutes / 60)
print ("Total time : " + str(timeInHours) + " hours and " + str(int(timeInMinutes % 60)) + " minutes")	
