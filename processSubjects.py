import subprocess, os, sys, pickle
import multiprocessing as mp
import time
start = time.time()
finishedFiles = open("FinishedFiles.txt","wb")
device_num = 0
def preProcessSubject(key, value, device_num):
    if (value['T1'] is not "") and (value['fmri'] is not []):
        T1_subject_file = T1_folder + "/" + value['T1']
        for folder,sub_folders,files in os.walk(T1_subject_file):
            for file in files:
                if file.endswith('.nii'):
                    T1_file = os.path.abspath(os.path.join(folder, file))
                    #print (T1_file)
        for rs_subject in value['fmri']:
            rs_subject_file = fmri_folder + "/" + rs_subject
            for folder,sub_folders,files in os.walk(rs_subject_file):
                for file in files:
                    if file.endswith('.nii'):
                        rs_file = os.path.abspath(os.path.join(folder, file))
                        #print (rs_file)
                        newSubjectDir = "/home/neuro/Desktop/neuro-group/Users/Vicki/ABCD/rs_fmri/BroccoliPreProcessed/" + key
                        if not os.path.exists(newSubjectDir):
                            os.makedirs(newSubjectDir)
                        print ("start preprocessing")
                        run_num = file.split("rest_run-")[1].split("_bold")[0]
                        run_command = "preProcessingScript " + "--root " + newSubjectDir + " --fmri " + rs_file + " --t1 " + T1_file + " --out run_" + run_num + " --device " + str(device_num)
                        print (run_command)
                        subprocess.call(run_command, shell=True)
                        print ("end preprocessing")
        finishedFiles.write(key + "\n")
#check input
if len(sys.argv) <3:
    print("Insert the T1 files folder and fmri files folder in this order!!!")
    exit()

T1_folder = sys.argv[1]
fmri_folder = sys.argv[2]
	
#read python dict back from the file
pkl_file = open('t1_to_fmri_dic.pkl', 'rb')
dict = pickle.load(pkl_file)
pkl_file.close()

# Init multiprocessing.Pool()
pool = mp.Pool(4)
print ("Number of cpus = " + str(mp.cpu_count()))

general_index = 0
for key, value in dict.items():
    #If all the neccesary files exist
    general_index = general_index + 1
    print("general_index: " + str(general_index))
    device_num = 1 - device_num
    [pool.apply_async(preProcessSubject, args=(key,value,device_num,))]
    if general_index >= 50:
	    break;
	
pool.close() 
pool.join()
finishedFiles.close()

end = time.time()
timeInSeconds = (end - start)
timeInMinutes = timeInSeconds / 60
timeInHours = int(timeInMinutes / 60)

print ("Total time : " + str(timeInHours) + " hours and " + str(int(timeInMinutes % 60)) + " minutes")

	


