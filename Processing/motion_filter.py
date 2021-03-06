#!/usr/bin/python3

"""
====================================================================
Run IIR filter on motion file
====================================================================
input: 
--in_file : full path to motion text file in the format like the output from the MotionCorrection Broccoli function.
--out_folder : full path to the output folder. Not obligatory

output: 
motionparameters_filtered.1D text file 
motionparameters_filtered.squared.1D text file with all the motion squared parameteres
motionparameters_filtered.backdif.1D text file with all the motion derivatives (backwards) parameteres
"""

import os, sys, argparse
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def upload_motion_file(motion_file):
    """Upload motion file to two dimensional array
	param motion_file: string. Full path to motion text file in the format like the output from the MotionCorrection Broccoli function.
	return: motion_array : ndarray
			two dimensional array (num of scans x 6) with all the motion regressors
	"""
    motion_array = []
    with open(motion_file, 'r') as f:
        for line in f.readlines():
            line = " ".join(line.split())
            motion_array.append([float(value) for value in line.split(' ')])
    return np.array(motion_array)
	
def create_filter(band_stop_min, band_stop_max, tr):
    """Create IIR notch filter 
	param band_stop_min: float. minimum band stop, in Hz 
	pram band_stop_max: float. maximum band stop, in Hz 
	param tr: float. TR, in seconds
	return: b,a : ndarray, ndarray
			Numerator (b) and denominator (a) polynomials of the IIR filter
	"""
    rr = np.array([band_stop_min , band_stop_max ])
    fs = 1 / tr;
    fNy = fs / 2;
    rr_new = (rr+fNy) / fs
    rr_new = np.array([np.floor(x) for x in rr_new])
    fa = abs(rr - (rr_new * fs))
    W_notch = fa / fNy;
    w0 = np.mean(W_notch)
    bw = np.diff(W_notch)
    Q = w0 / bw 
    b, a = signal.iirnotch(w0, Q );
    return b,a
	
	
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', required=True, type=str, help='path to motion text file')
    parser.add_argument('--out_folder', required=False, default = ".", type=str, help='path to output folder')
    args = parser.parse_args()	
	
    motion_array = upload_motion_file(args.in_file)
    band_stop_min = 0.31
    band_stop_max = 0.43
    tr = 0.8
    order = 2
    b, a = create_filter(band_stop_min, band_stop_max, tr)
    num_f_apply = int(np.floor(order / 2)); # if order<4 apply filter 1x, if order=4 2x, if order=6 3x
    filtered_array = signal.filtfilt(b,a,motion_array,axis=0)
    for i in range(0,num_f_apply-1):
        filtered_array = signal.filtfilt(b_filt,a_filt,filtered_array, axis=0)
    filtered_array = np.round(filtered_array,6)
	
	#write the output motion file
    motion_output = os.path.join(args.out_folder, "motionparameters_filtered.1D")
    filtered_motion_file = open(motion_output, 'w')
    squared_motion_output = os.path.join(args.out_folder, "motionparameters_filtered.squared.1D")
    filtered_motion_squred_file = open(squared_motion_output, 'w')
    backdif_motion_output = os.path.join(args.out_folder, "motionparameters_filtered.backdif.1D")
    backdif_motion_output_file = open(backdif_motion_output, 'w')
    for row in filtered_array:
        for val in row:
            filtered_motion_file.write(str(val)+" ")
            filtered_motion_squred_file.write(str(round(val*val,6))+" ")          
        filtered_motion_file.write("\n")	
        filtered_motion_squred_file.write("\n")
    backdif_motion_output_file.write("0 0 0 0 0 0\n")
    backdif_array=filtered_array[1:,:] - filtered_array[:-1,:]
    for row in backdif_array:
        for val in row:
            backdif_motion_output_file.write(str(round(val,6))+" ")
        backdif_motion_output_file.write("\n")	
