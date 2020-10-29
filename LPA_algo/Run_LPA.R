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
data <- read_excel("C:/Users/ויקי/Desktop/googleDriveSync/secondDegree/ABCD_important_data/profile_data.xlsx")
#delete empty rows
data <- na.omit(data)
#write_xlsx(data,"C:/Users/ויקי/Desktop/googleDriveSync/secondDegree/ABCD_important_data/profile_data.xlsx")
# Redicrect output to file
sink("LPA_Output.txt", append=FALSE, split=FALSE)
options(dplyr.width = Inf)
#Run LPA with n models
lpa_models <- data %>%
  select(RSFMRI_C_NGD_DT_NGD_DT,
         RSFMRI_C_NGD_DLA_NGD_DLA,
         RSFMRI_C_NGD_FO_NGD_FO, 
         RSFMRI_C_NGD_VTA_NGD_VTA,
         RSFMRI_C_NGD_CGC_NGD_CGC,
         RSFMRI_C_NGD_SA_NGD_SA,
         RSFMRI_C_NGD_CGC_NGD_DT,
         RSFMRI_C_NGD_CGC_NGD_DLA,
         RSFMRI_C_NGD_CGC_NGD_FO,
         RSFMRI_C_NGD_CGC_NGD_SA,
         RSFMRI_C_NGD_CGC_NGD_VTA,
         RSFMRI_C_NGD_DT_NGD_DLA,
         RSFMRI_C_NGD_DT_NGD_FO,
         RSFMRI_C_NGD_DT_NGD_SA,
         RSFMRI_C_NGD_DT_NGD_VTA,
         RSFMRI_C_NGD_DLA_NGD_FO,
         RSFMRI_C_NGD_DLA_NGD_SA,
         RSFMRI_C_NGD_DLA_NGD_VTA,
         RSFMRI_C_NGD_FO_NGD_SA,
         RSFMRI_C_NGD_FO_NGD_VTA,
         RSFMRI_C_NGD_SA_NGD_VTA
         ) %>%
  single_imputation() %>%
#  scale() %>%
  estimate_profiles(1:8)
#Redirect back to consule
#sink()
#Get the output for all the models
get_fit(lpa_models)
#Get the chosen model and save to excel file
df <- get_data(lpa_models[[6]])
#Add subject key column - not tested
#df$SUBJECTKEY <- paste(data$SUBJECTKEY)
write_xlsx(df,"LPA_output_model_brain_6.xlsx")
#Calculate mean value for each variable (column) per class
means <- aggregate(select(df, -c(model_number, classes_number, CPROB1, CPROB2,CPROB3,CPROB4,CPROB5,CPROB6, Class)),
                 by=list(df$Class), mean) %>%
                t() %>%
                melt(id.vars = "Group.1", measure.vars = c("V1", "V2", "V3","V4","V5","V6")) %>%
                subset(X1!="Group.1")
#Plot the means
means %>%
  subset(means$X2 == 3 | means$X2 == 1)%>%
  subset(means$X1 == 'RSFMRI_C_NGD_CGC_NGD_DT' |
         means$X1 == 'RSFMRI_C_NGD_CGC_NGD_FO' |
         means$X1 == 'RSFMRI_C_NGD_CGC_NGD_SA' |
         means$X1 == 'RSFMRI_C_NGD_CGC_NGD_VTA' |
         means$X1 == 'RSFMRI_C_NGD_DT_NGD_DLA' |
         means$X1 == 'RSFMRI_C_NGD_DT_NGD_SA' |
         means$X1 == 'RSFMRI_C_NGD_DT_NGD_VTA' |
         means$X1 == 'RSFMRI_C_NGD_SA_NGD_SA' |
         means$X1 == 'RSFMRI_C_NGD_SA_NGD_VTA' |
         means$X1 == 'RSFMRI_C_NGD_VTA_NGD_VTA') %>%
  ggplot(aes(X1,value, group = X2, color = as.factor(X2))) +
  geom_point(size = 2.25) +
  geom_line(size = 1.25) +
  theme_bw(base_size = 14) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "top")+
  labs(x = NULL, y = "Standardized mean", color = "classes") 

