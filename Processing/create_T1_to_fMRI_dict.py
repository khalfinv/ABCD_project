"""
====================================================================
Create dictionary of T1 to fMRI files of each subject
====================================================================
key = SUBJECT_KEY
value = {'rs' : [list of paths to fmri files], 'T1' : ""}
"""

import os, sys, re, pickle, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--subject_list', required=True, type=str, help='list of SUBJECT_KEY')
parser.add_argument('--t1_folder', required=True, type=str, help='path to T1 folder')
parser.add_argument('--fmri_folder', required=True, type=str, help='path to fmri folder')
args = parser.parse_args()
	

listFile = open(args.subject_list)

notRsExistingFile = open("notRsExistingFile.txt", "w")
notT1ExistingFile = open("notT1ExistingFile.txt", "w")
dic = {}
for subject_key in listFile:
    rsFound = False
    T1Found = False
    subject_key = re.sub(r"[^a-zA-Z0-9]","",subject_key) 	#remove '_' and whitespaces in subject_key  
    rsListDir = os.listdir(args.fmri_folder)
    rsListDir.sort()
    for rsFolderName in rsListDir:
        if subject_key in rsFolderName:
            if subject_key not in dic:
                dic[subject_key] = {'fmri' : [], 'T1' : ""} 
            dic[subject_key]["fmri"].append(rsFolderName) 
            rsFound = True
    t1ListDir = os.listdir(args.t1_folder)
    t1ListDir.sort()
    for T1FolderName in t1ListDir: 
        if subject_key in T1FolderName:
            if subject_key not in dic:
                dic[subject_key] = {'fmri' : [], 'T1' : ""}
            dic[subject_key]['T1'] = T1FolderName 
            T1Found = True
    if rsFound == False:
        notRsExistingFile.write(subject_key + "\n")
    if T1Found == False:
	    notT1ExistingFile.write(subject_key + "\n")

# write python dict to a file
output = open('t1_to_fmri_dic.pkl', 'wb')
pickle.dump(dic, output)
output.close()
	
notRsExistingFile.close()
notT1ExistingFile.close()