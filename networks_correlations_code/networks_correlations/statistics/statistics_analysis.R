library(readxl)
library(psych)

Mat1_excel = "C:/Users/ειχι/Desktop/ABCD_project/networks_correlations_code/networks_correlations/out/Class2/correlation_matrix_['Default'].xlsx"
Mat2_excel = "C:/Users/ειχι/Desktop/ABCD_project/networks_correlations_code/networks_correlations/out/All/correlation_matrix_['Default'].xlsx"

Mat1 <- read_excel(Mat1_excel)
Mat1 <- Mat1[-c(1)]
A <- as.matrix(Mat1)
Mat2 <- read_excel(Mat2_excel)
Mat2 <- Mat2[-c(1)]
B <- as.matrix(Mat2)
cortest.jennrich(A,B,n1=2661,n2=5055) # The Jennrich test
