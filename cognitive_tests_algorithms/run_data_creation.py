import subprocess, argparse, os


if __name__ == "__main__":
	X_files = [r"C:\Users\ויקי\Desktop\googleDriveSync\secondDegree\ABCD_important_data\BehavioralData\abcd_tbss01.xlsx",
		r"C:\Users\ויקי\Desktop\googleDriveSync\secondDegree\ABCD_important_data\BehavioralData\abcd_stq01_screenTime.xlsx",
		r"C:\Users\ויקי\Desktop\googleDriveSync\secondDegree\ABCD_important_data\BehavioralData\pmq01_parental_monitoring.xlsx",
		r"C:\Users\ויקי\Desktop\googleDriveSync\secondDegree\ABCD_important_data\BehavioralData\abcd_mhy02.xlsx",
		r"C:\Users\ויקי\Desktop\googleDriveSync\secondDegree\ABCD_important_data\BehavioralData\lmtp201_little_man.xlsx",
		r"C:\Users\ויקי\Desktop\googleDriveSync\secondDegree\ABCD_important_data\BehavioralData\cc01_cash_choice.xlsx",
		r"C:\Users\ויקי\Desktop\googleDriveSync\secondDegree\ABCD_important_data\BehavioralData\abcd_ps01_pearson_scores.xlsx"]
	Y_file =  r"C:\Users\ויקי\Desktop\googleDriveSync\secondDegree\ABCD_important_data\BehavioralData\abcd_tbss01.xlsx"
	X_columns = ["NIHTBX_PICVOCAB_AGECORRECTED",
		"NIHTBX_PICVOCAB_FC",
		"NIHTBX_LIST_AGECORRECTED",
		"NIHTBX_LIST_FC",
		"NIHTBX_CARDSORT_AGECORRECTED",
		"NIHTBX_CARDSORT_FC",
		"NIHTBX_PATTERN_AGECORRECTED",
		"NIHTBX_PATTERN_FC",
		"NIHTBX_PICTURE_AGECORRECTED",
		"NIHTBX_PICTURE_FC",
		"NIHTBX_READING_AGECORRECTED",
		"NIHTBX_READING_FC",
		"SCREEN1_WKDY_Y",
		"SCREEN2_WKDY_Y",
		"SCREEN3_WKDY_Y",
		"SCREEN4_WKDY_Y",
		"SCREEN5_WKDY_Y",
		"SCREEN_WKDY_Y",
		"SCREEN7_WKND_Y",
		"SCREEN8_WKND_Y",
		"SCREEN9_WKND_Y",
		"SCREEN10_WKND_Y",
		"SCREEN11_WKND_Y",
		"SCREEN12_WKND_Y",
		"SCREEN13_Y",
		"SCREEN14_Y",
		"PARENT_MONITOR_Q1_Y",
		"PARENT_MONITOR_Q2_Y",
		"PARENT_MONITOR_Q3_Y",
		"PARENT_MONITOR_Q4_Y",
		"PARENT_MONITOR_Q5_Y",
		"UPPS_Y_SS_NEGATIVE_URGENCY",
		"UPPS_Y_SS_LACK_OF_PLANNING",
		"UPPS_Y_SS_SENSATION_SEEKING",
		"UPPS_Y_SS_POSITIVE_URGENCY",
		"UPPS_Y_SS_LACK_OF_PERSEVERANCE",
		"BIS_Y_SS_BIS_SUM",
		"BIS_Y_SS_BAS_RR",
		"BIS_Y_SS_BAS_DRIVE",
		"BIS_Y_SS_BAS_FS",
		"LMT_SCR_PERC_CORRECT",
		"LMT_SCR_PERC_WRONG",
		"LMT_SCR_NUM_CORRECT",
		"LMT_SCR_NUM_WRONG",
		"LMT_SCR_NUM_TIMED_OUT",
		"LMT_SCR_AVG_RT",
		"LMT_SCR_RT_CORRECT",
		"LMT_SCR_RT_WRONG",
		"LMT_SCR_EFFICIENCY",
		"CASH_CHOICE_TASK",
		"PEA_WISCV_TRS",
		"PEA_WISCV_TSS"
		]
	Y_Column = "NIHTBX_FLANKER_AGECORRECTED"
	X_files_str = ' '.join([file for file in X_files]) 
	X_columns_str = ' '.join([col for col in X_columns]) 
	command = "create_data_file.py --X_files " + X_files_str +  " --Y_file " + Y_file + " --X_columns " + X_columns_str + " --Y_column " + Y_Column
	print(command)
	subprocess.call(command, shell=True)