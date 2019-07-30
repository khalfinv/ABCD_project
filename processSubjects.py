#!/usr/bin/python3
"""
====================================================
Run preprocessing for all the fmri files
====================================================
input: 
--t1_folder : full path to folder containing all the T1 files
--fmri_folder : full path to folder containing all the fmri files
--out_folder' : path to output folder
--dict_pkl' : full path to t1 to fmri dictionary pkl file


output: 
	preprocessed folder for each subject 
"""

import subprocess, os, sys, pickle, argparse
import multiprocessing as mp
import time


def preProcessSubject(subject_id, value, device_num):
    """ Call the preProcessing script
    param subject_id: string. The subject's id
    param value: dictionary. t1_to_fmri dictionary value corresponding to subject's id. {T1: the T1 file, fmri: list of all fMRI folders}
    param device_num: integer. The gpu device number (0 or 1)
    return: None
    """
    try:
        if (value['T1'] is not "") and (value['fmri'] is not []):
            T1_subject_folder = T1_folder + "/" + value['T1']
            for folder,sub_folders,files in os.walk(T1_subject_folder):
                for file in files:
                    if file.endswith('.nii'):
                        T1_file = os.path.abspath(os.path.join(folder, file))
            for fmri_subject_folder in value['fmri']:
                fmri_subject_folder_full_path = fmri_folder + "/" + fmri_subject_folder
			    #search for the fMRI file and motion file
                fmri_file = None
                motion_file = None
                for folder,sub_folders,files in os.walk(fmri_subject_folder_full_path):
                    for file in files:
                        if file.endswith('.nii'):
                            fmri_file = os.path.abspath(os.path.join(folder, file))
                            print (fmri_file)
                        if file.endswith('.1D'):
                            motion_file = os.path.abspath(os.path.join(folder, file))
                            print (motion_file)
                if motion_file != None and fmri_file != None:
                    newSubjectDir = os.path.abspath(os.path.join(out_folder, subject_id))
                    if not os.path.exists(newSubjectDir):
                        os.makedirs(newSubjectDir)
                    #start preprocessing
                    run_num = fmri_file.split("rest_run-")[1].split("_bold")[0]
                    run_command = "preProcessingScript " + "--root " + newSubjectDir + " --fmri " + fmri_file + " --t1 " + T1_file + " --out run_" + run_num + " --device " + str(device_num) + " --motion " + motion_file
                    print (run_command)
                    ret = subprocess.run(run_command, shell=True, check=True)
                    #end preprocessing
                else:
                    raise Exception("Not found fmri file or motion file in one of the folders")
        else:
            raise Exception("Missing T1 or fmri value in dictionary") 
    except:
        raise Exception( "subject_id: %s \n" % subject_id + str(sys.exc_info()[1])).with_traceback(sys.exc_info()[2])
		
		
def collect_errors(exception):
    """ Callback for errors collecting from threads. Get the exception and write to file
    param exception: Exception. The exception that was raised
    """
    error_file.write(str(exception))
    error_file.write("\n\n\n")
    error_file.flush()
	
if __name__ == "__main__":
	#start timer
    start = time.time()	
	
	#parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--t1_folder', required=True, type=str, help='path to folder containing of the T1 files')
    parser.add_argument('--fmri_folder', required=True, type=str, help='path to folder containing of the fMRI files')
    parser.add_argument('--out_folder', required=True, type=str, help='path to output folder')
    parser.add_argument('--dict_pkl', required=True, type=str, help='full path to t1 to fmri dictionary pkl file')
    args = parser.parse_args()
	
    T1_folder = args.t1_folder
    fmri_folder = args.fmri_folder
    out_folder = args.out_folder
    dict_pkl = args.dict_pkl
						
    #read dict from pkl file
    pkl_file = open(dict_pkl, 'rb')
    dict = pickle.load(pkl_file)
    pkl_file.close()
						
    #open the error and out files
    error_file = open(out_folder + "/error_file.txt", "w+")
    # Init multiprocessing.Pool()
    pool = mp.Pool(4)

    device_num = 0 #gpu device number (0 or 1)
    general_index = 0 #If we want only part of the subjects
    for key, value in dict.items():
        #general_index = general_index + 1
        #print("general_index: " + str(general_index))
        device_num = 1 - device_num
        [pool.apply_async(preProcessSubject, args=(key,value,device_num,), error_callback = collect_errors )]
        #if general_index >= 2:
	    #    break;
			
	#Close the multiprocessing.Pool() and wait for all threads
    pool.close() 
    pool.join()
	
    #stop timer and print the elapsed time
    end = time.time()
    timeInSeconds = (end - start)
    timeInMinutes = timeInSeconds / 60
    timeInHours = int(timeInMinutes / 60)
    print ("Total time : " + str(timeInHours) + " hours and " + str(int(timeInMinutes % 60)) + " minutes")

	


