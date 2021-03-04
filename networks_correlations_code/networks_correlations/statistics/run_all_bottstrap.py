import subprocess
import time


def checkSubprocessRet(return_code):
	if (return_code != 0):
		print("Command failed!")
		exit(1)
	else:
		print("Command succeeded!")
if __name__ == "__main__":
	start = time.time()
	sample_size = 1
	R = 100
	bootstrap_commands = ["bootstrap.py --out_folder Y:\Vicki\Original_data\Class1\R100 --subjects_dic Y:\Vicki\Original_data\Class1\subjects_data_class_1.pkl --R " + str(R) + " --sample_size " + str(sample_size)
	,"bootstrap.py --out_folder Y:\Vicki\Original_data\Class2\R100 --subjects_dic Y:\Vicki\Original_data\Class2\subjects_data_class_2.pkl --R " + str(R) + " --sample_size " + str(sample_size)
	,"bootstrap.py --out_folder Y:\Vicki\Original_data\Class3\R100 --subjects_dic Y:\Vicki\Original_data\Class3\subjects_data_class_3.pkl --R " + str(R) + " --sample_size " + str(sample_size)
	,"bootstrap.py --out_folder Y:\Vicki\Original_data\Class4\R100 --subjects_dic Y:\Vicki\Original_data\Class4\subjects_data_class_4.pkl --R " + str(R) + " --sample_size " + str(sample_size)
	]
	
	for command in bootstrap_commands:
		print(command)
		return_code = subprocess.call(command, shell=True)
		checkSubprocessRet(return_code)
	#print the exucation time
	end = time.time()
	timeInSeconds = (end - start)
	timeInMinutes = timeInSeconds / 60
	timeInHours = int(timeInMinutes / 60)
	print ("Total time : " + str(timeInHours) + " hours and " + str(int(timeInMinutes % 60)) + " minutes")	