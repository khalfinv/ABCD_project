#########################################
# Run LPA.
# Output the chosen model to excel file.
#############################################


library(readxl)
library(writexl)
library(tidyLPA)
library(dplyr)
library(mclust)
library(ggplot2)
library(tidyr)
library(reshape)


# read the data
input_file = "C:/Users/ειχι/Desktop/googleDriveSync/secondDegree/ABCD_important_data/final_dataset.xlsx"
data <- read_excel(input_file)
#delete empty rows
data <- na.omit(data)
# Redirect output to file
LPA_consule_output = "C:/Users/ειχι/Desktop/ABCD_project/LPA_algo/LPA_Output.txt"
sink(LPA_consule_output, append=FALSE, split=FALSE)
options(dplyr.width = Inf)
columns = c('CBCL_SCR_DSM5_ADHD_T',
             'BIS_Y_SS_BIS_SUM',
             'BIS_Y_SS_BAS_RR',
             'BIS_Y_SS_BAS_DRIVE',
             'BIS_Y_SS_BAS_FS',
             'TFMRI_SST_ALL_BEH_INCRGO_RT',
             'TFMRI_SST_ALL_BEH_INCRS_RT',
             'TFMRI_SST_ALL_BEH_INCRS_MRT')
#Run LPA with n models
lpa_models <- data %>%
  select(columns) %>%
  single_imputation() %>%
  scale() %>%
  estimate_profiles(1:6)
#Redirect back to console
#sink()
#Get the output for all the models
get_fit(lpa_models)
#Get the chosen model and save to excel file
df <- get_data(lpa_models[[4]])

#Calculate mean value for each variable (column) per class
means <- aggregate(select(df, -c(model_number, classes_number, CPROB1, CPROB2,CPROB3,CPROB4, Class)),
                 by=list(df$Class), mean) %>%
                t() %>%
                melt(id.vars = "Group.1", measure.vars = c("V1", "V2", "V3","V4")) %>%
                subset(X1!="Group.1")
#Plot the means
means %>%
  ggplot(aes(X1,value, group = X2, color = as.factor(X2))) +
  geom_point(size = 2.25) +
  geom_line(size = 1.25) +
  theme_bw(base_size = 14) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "top")+
  labs(x = NULL, y = "Standardized mean", color = "classes") 

#Add originals columns ans save the data frame
for (col in columns){
  col_new = paste(col,"_not_scaled",sep="")
  df[col_new] = paste(data[[col]])
  print(col_new)
}

#Add subject key column
df$SUBJECTKEY <- paste(data$SUBJECTKEY)
profiles_output_data = "C:/Users/ειχι/Desktop/ABCD_project/LPA_algo/LPA_output_model_4_final.xlsx"
write_xlsx(df,profiles_output_data)
