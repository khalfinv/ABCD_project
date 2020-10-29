#########################################
# Run Manova and Anova
#############################################

library(lsr)
library(car)
library(rstatix)
library(readxl)
library(ggpubr)
library(dplyr)
library(GGally)
library(statsExpressions)
library(writexl)
library(ggplot2)
library(tidyr)
library(reshape)
library(corrplot)
library(FSA)
library(DescTools)

create_means_plot <- function(df, columns) {
  #Calculate mean value for each variable (column) per class
  means <- aggregate(select(df, columns),
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
}

find_outliers <- function(df, column ){
  df_outliers <- df %>%
    group_by(Class) %>%
    identify_outliers(column) %>%
    subset(select = c(is.extreme, SUBJECTKEY))
  
  df_outliers <- subset(df_outliers, select = -c(is.extreme), df_outliers$is.extreme == TRUE)
  df_outliers <- merge(df_outliers,df,by="SUBJECTKEY")
  return (df_outliers)
}

outliers_exam <- function(df, outlier_col, all_columns){
  print(paste("Exam ",outlier_col))
  df_outliers = find_outliers(df, outlier_col)
  df_no_outliers <- setdiff(df,df_outliers)
  create_means_plot(df_no_outliers,all_columns)
  
  excel_name = paste("df_outliers_",outlier_col,".xlsx")
  write_xlsx(df_outliers,excel_name)
  
  df_no_outliers %>%
    group_by(Class) %>%
    get_summary_stats(all_columns, type = "mean_sd")
  return (df_no_outliers)
}

#Redirect output to file
sink("stats_output.txt", append=FALSE, split=FALSE)
options(tibble.print_min = Inf)
options(max.print = 400)
#Read the data
df <- read_excel("LPA_output_model_brain_6.xlsx")
all_col_scaled <- c('RSFMRI_C_NGD_DT_NGD_DT',
                     'RSFMRI_C_NGD_DLA_NGD_DLA',
                     'RSFMRI_C_NGD_FO_NGD_FO', 
                     'RSFMRI_C_NGD_VTA_NGD_VTA',
                     'RSFMRI_C_NGD_CGC_NGD_CGC',
                     'RSFMRI_C_NGD_SA_NGD_SA',
                     'RSFMRI_C_NGD_CGC_NGD_DT',
                     'RSFMRI_C_NGD_CGC_NGD_DLA',
                     'RSFMRI_C_NGD_CGC_NGD_FO',
                     'RSFMRI_C_NGD_CGC_NGD_SA',
                     'RSFMRI_C_NGD_CGC_NGD_VTA',
                     'RSFMRI_C_NGD_DT_NGD_DLA',
                     'RSFMRI_C_NGD_DT_NGD_FO',
                     'RSFMRI_C_NGD_DT_NGD_SA',
                     'RSFMRI_C_NGD_DT_NGD_VTA',
                     'RSFMRI_C_NGD_DLA_NGD_FO',
                     'RSFMRI_C_NGD_DLA_NGD_SA',
                     'RSFMRI_C_NGD_DLA_NGD_VTA',
                     'RSFMRI_C_NGD_FO_NGD_SA',
                     'RSFMRI_C_NGD_FO_NGD_VTA',
                     'RSFMRI_C_NGD_SA_NGD_VTA')
all_col_not_scaled <- c('CBCL_DSM5_ADHD_not_scaled', 
                    'BIS_SUM_not_scaled', 
                    'BAS_RR_not_scaled', 
                    'BAS_DRIVE_not_scaled', 
                    'BAS_FS_not_scaled',
                    'INCRGO_RATE_not_scaled',
                    'INCRGO_MRT_not_scaled',
                    'INCRS_RATE_not_scaled',
                    'INCRS_MRT_not_scaled')

#Data visualization
ggboxplot(
  df, x = "Class", y = all_col_scaled, 
  merge = TRUE, palette = "jco"
)

#Group statistics scaled
df %>%
  group_by(Class) %>%
  get_summary_stats(all_col_scaled, type = "mean_sd")

#Group statistics not scaled
df %>%
  group_by(Class) %>%
  get_summary_stats(all_col_not_scaled, type = "mean_sd")

#Check outliers
#univariate outliers
#INCRGO_RATE
df_no_outliers_INCRGO_RATE = outliers_exam(df,"INCRGO_RATE",all_col_scaled)

#INCRGO_MRT
df_no_outliers_INCRGO_MRT = outliers_exam(df,"INCRGO_MRT",all_col_scaled)
df_no_outliers = merge(df_no_outliers_INCRGO_MRT,df_no_outliers_INCRGO_RATE)

#INCRS_RATE
df_no_outliers_INCRS_RATE = outliers_exam(df,"INCRS_RATE",all_col_scaled)
df_no_outliers = merge(df_no_outliers,df_no_outliers_INCRS_RATE)

#INCRS_MRT
df_no_outliers_INCRS_MRT = outliers_exam(df,"INCRS_MRT",all_col_scaled)
df_no_outliers = merge(df_no_outliers,df_no_outliers_INCRS_MRT)

#multivariate outliers
df_mul_outliers <- df_no_outliers %>%
  group_by(Class) %>%
  subset(select = all_col_scaled) %>%
  mahalanobis_distance() %>%
  as.data.frame()

subset(df_mul_outliers, df_mul_outliers$is.outlier==TRUE) %>%
merge(df) %>%
write_xlsx("multivariate_outliers.xlsx")

df_no_outliers <- subset(df_no_outliers, !df_mul_outliers$is.outlier)
create_means_plot(df_no_outliers,all_col_scaled)

# QQ plot of 
lapply(all_col_scaled, function(s) {
  jpeg(paste(s,"_qqplot.jpeg"))
  plot <- ggqqplot(df, s, facet.by = "Class",
           ylab = s, ggtheme = theme_bw()) 
  print(plot)
  dev.off()
  
})


#Check multicollinearity
jpeg("corr_plot.jpg")
dff = subset(df, select = all_col_scaled) %>% cor()
corrplot(dff, method="number")
dev.off()

#Check linearity assumption

results <- df_no_outliers %>%
  select(all_col_scaled, Class) %>%
  group_by(Class) %>%
  doo(~ggpairs(.) + theme_bw(), result = "plots")
jpeg("linearity_group1.jpg")
results$plots[1]
dev.off()
jpeg("linearity_group2.jpg")
results$plots[2]
dev.off()
jpeg("linearity_group3.jpg")
results$plots[3]
dev.off()
jpeg("linearity_group4.jpg")
results$plots[4]
dev.off()


#Check the homogeneity of covariances assumption
box_m(df_no_outliers[, all_col_scaled], df_no_outliers$Class)

#Check the homogneity of variance assumption
df_no_outliers %>% 
  gather(key = "variable", value = "value", all_col_scaled) %>%
  group_by(variable) %>%
  levene_test(value ~ as.factor(Class))

res.man <- manova(cbind(RSFMRI_C_NGD_DT_NGD_DT,
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
                         RSFMRI_C_NGD_SA_NGD_VTA) ~ Class, data = df)
summary(res.man)

# Group the data by variable
grouped.data <- df %>%
  gather(key = "variable", value = "value", all_col_scaled) %>%
  group_by(variable)

# Do welch one way anova test
grouped.data %>% welch_anova_test(value ~ Class) 

# or do Kruskal-Wallis test
res = grouped.data %>% kruskal_test(value ~ Class)
grouped.data %>% kruskal_effsize(value ~ Class)
print(res)


#Run for every variable (not worked with lapply)
print("RSFMRI_C_NGD_DT_NGD_DT")
DunnTest(data = df, RSFMRI_C_NGD_DT_NGD_DT ~ Class, method="bonferroni")
print("RSFMRI_C_NGD_DLA_NGD_DLA")
DunnTest(data = df, RSFMRI_C_NGD_DLA_NGD_DLA ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_FO_NGD_FO")
DunnTest(data = df, RSFMRI_C_NGD_FO_NGD_FO ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_VTA_NGD_VTA")
DunnTest(data = df, RSFMRI_C_NGD_VTA_NGD_VTA ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_CGC_NGD_CGC")
DunnTest(data = df, RSFMRI_C_NGD_CGC_NGD_CGC ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_SA_NGD_SA")
DunnTest(data = df, RSFMRI_C_NGD_SA_NGD_SA ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_CGC_NGD_DT")
DunnTest(data = df, RSFMRI_C_NGD_CGC_NGD_DT ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_CGC_NGD_DLA")
DunnTest(data = df, RSFMRI_C_NGD_CGC_NGD_DLA ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_CGC_NGD_FO")
DunnTest(data = df, RSFMRI_C_NGD_CGC_NGD_FO ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_CGC_NGD_SA")
DunnTest(data = df, RSFMRI_C_NGD_CGC_NGD_SA ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_CGC_NGD_VTA")
DunnTest(data = df, RSFMRI_C_NGD_CGC_NGD_VTA ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_DT_NGD_DLA")
DunnTest(data = df, RSFMRI_C_NGD_DT_NGD_DLA ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_DT_NGD_FO")
DunnTest(data = df, RSFMRI_C_NGD_DT_NGD_FO ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_DT_NGD_SA")
DunnTest(data = df, RSFMRI_C_NGD_DT_NGD_SA ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_DT_NGD_VTA")
DunnTest(data = df, RSFMRI_C_NGD_DT_NGD_VTA ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_DLA_NGD_FO")
DunnTest(data = df, RSFMRI_C_NGD_DLA_NGD_FO ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_DLA_NGD_SA")
DunnTest(data = df, RSFMRI_C_NGD_DLA_NGD_SA ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_DLA_NGD_VTA")
DunnTest(data = df, RSFMRI_C_NGD_DLA_NGD_VTA ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_FO_NGD_SA")
DunnTest(data = df, RSFMRI_C_NGD_FO_NGD_SA ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_FO_NGD_VTA")
DunnTest(data = df, RSFMRI_C_NGD_FO_NGD_VTA ~ Class, method="bonferroni") 
print("RSFMRI_C_NGD_SA_NGD_VTA")
DunnTest(data = df, RSFMRI_C_NGD_SA_NGD_VTA ~ Class, method="bonferroni") 



