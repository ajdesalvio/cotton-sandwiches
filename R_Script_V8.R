library(fdapace)
library(dplyr)
library(nlme)
library(sjstats)
library(lme4)
library(RColorBrewer)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(ks)
library(readxl)
library(merTools)


#### Combined analysis - FPCA using Manual Senescence Scores ####
## IMPORTANT - PLEASE READ ##
# This FPCA uses pooled data from OGR (E1) AND HB (E2) - the reasoning is as follows:
# FPCA produces values that are unitless. Therefore, if E1 and E2 were assessed
# with FPCA separately (which E1 is later in the script), they may end up with different
# scales of values for FPC1. This could complicate CNN-based regression of FPC1
# values because an FPC1 score of 10 in E1 may not match up with an FPC1 score of 10 in E2 
# (they may confer different temporal phenotypes). It is appropriate however to use FPCA 
# separately for E1 when using FPC1 as a response variable for ANOVA, but for CNN 
# regression it makes more sense to perform a combined FPCA and use FPC1 scores 
# from the combined analysis for regression.

df <- read.csv('df1.csv')
df_clean <- df[!is.na(df$Score),] # Use the "Score" column to drop NAs. This is
# done with confidence because while visually scoring senescence, single plants
# that perished were identified and noted. We can check to make sure that 7 plants
# were omitted with the following math: 6720-6622=98; 98/14=7 plants omitted
# Divided by 14 because the data frame was a stacked version of the data
# with 14 temporal observations per plant.

# Here we're making sure that the days after transplanting (DAT) match up with
# their respective flight dates.
day_DAP <- df_clean %>% distinct(DAT, Flight_Date)

head(df_clean)

# Preparing data into "wide" format for use with FPCA
data_wide <- reshape2::dcast(df_clean, Pltg_ID ~ DAT, value.var="Score")
data_wide <- na.omit(data_wide)

L3 <- MakeFPCAInputs(IDs = rep(data_wide$Pltg_ID, each=length(unique(df_clean$DAT))),
                     tVec=rep(unique(df_clean$DAT),length(data_wide$Pltg_ID)),
                     t(data_wide[,-1]))


FPCAsparse <- FPCA(L3$Ly, L3$Lt,list(dataType='Sparse',plot=TRUE,
                                     methodMuCovEst='smooth',
                                     userBwCov=2,
                                     methodBwCov = "GCV",
                                     methodBwMu = "GCV"))
plot(FPCAsparse)
CreatePathPlot(FPCAsparse)

CreateOutliersPlot(FPCAsparse,optns=list(K=2,variant='KDE'))
CreatePathPlot(FPCAsparse,subset=c(3,5,135),K=3,main="K=3",pch=4)

fpca <- as.data.frame(FPCAsparse$xiEst)
fpca <- fpca[,c(1,2)]
fpca$Pltg_ID <- data_wide$Pltg_ID

# These two lines reveal the percentage of temporal variation explained by the
# first two eigenfunctions (FPC1 and FPC2, respectively).
fpca1 <- FPCAsparse$cumFVE[1]
fpca2 <- FPCAsparse$cumFVE[2]- FPCAsparse$cumFVE[1]

colnames(fpca)[1:2] <- c('FPC1_Combined', 'FPC2_Combined')

head(fpca)

# Merge manual senescence scores data frame with FPC1 and FPC2 for inspection later
# if desired.
fpca_and_rawscores_Combined <- fpca %>% left_join(data_wide, by = 'Pltg_ID')


#### Combined analysis - FPCA using RCC ####
df <- read.csv('df1.csv')
df_clean <- df[!is.na(df$Score),]
# Check: 6720-6622=98; 98/14=7 plants omitted

day_DAP <- df_clean %>% distinct(DAT, Flight_Date)

head(df_clean)

data_wide <- reshape2::dcast(df_clean, Pltg_ID ~ DAT, value.var="RCC")
data_wide <- na.omit(data_wide)

L3 <- MakeFPCAInputs(IDs = rep(data_wide$Pltg_ID, each=length(unique(df_clean$DAT)) ),
                     tVec=rep(unique(df_clean$DAT),length(data_wide$Pltg_ID)),
                     t(data_wide[,-1]))


FPCAsparse <- FPCA(L3$Ly, L3$Lt,list(dataType='Sparse',plot=TRUE,
                                     methodMuCovEst='smooth',
                                     userBwCov=2,
                                     methodBwCov = "GCV",
                                     methodBwMu = "GCV"))
plot(FPCAsparse)
CreatePathPlot(FPCAsparse)

CreateOutliersPlot(FPCAsparse,optns=list(K=2,variant='KDE'))
CreatePathPlot(FPCAsparse,subset=c(3,5,135),K=3,main="K=3",pch=4)
CreatePathPlot(FPCAsparse,subset=c(3,5,135),K=4,main="K=4",pch=4)

fpca <- as.data.frame(FPCAsparse$xiEst)
fpca <- fpca[,c(1,2)]
fpca$Pltg_ID <- data_wide$Pltg_ID
fpca1 <- FPCAsparse$cumFVE[1]
fpca2 <- FPCAsparse$cumFVE[2]- FPCAsparse$cumFVE[1]

colnames(fpca)[1:2] <- c('FPC1_RCC', 'FPC2_RCC')

head(fpca)

# Merge RCC index value data frame with FPC1 and FPC2 for inspection later
# if desired.
fpca_and_rawscores_RCC <- fpca %>% left_join(data_wide, by = 'Pltg_ID')


#### Combined analysis - FPCA using TNDGR ####
df <- read.csv('df1.csv')
df_clean <- df[!is.na(df$Score),]
# Check: 6720-6622=98; 98/14=7 plants omitted

day_DAP <- df_clean %>% distinct(DAT, Flight_Date)

head(df_clean)

data_wide <- reshape2::dcast(df_clean, Pltg_ID ~ DAT, value.var="TNDGR")
data_wide <- na.omit(data_wide)

L3 <- MakeFPCAInputs(IDs = rep(data_wide$Pltg_ID, each=length(unique(df_clean$DAT)) ),
                     tVec=rep(unique(df_clean$DAT),length(data_wide$Pltg_ID)),
                     t(data_wide[,-1]))


FPCAsparse <- FPCA(L3$Ly, L3$Lt,list(dataType='Sparse',plot=TRUE,
                                     methodMuCovEst='smooth',
                                     userBwCov=2,
                                     methodBwCov = "GCV",
                                     methodBwMu = "GCV"))
plot(FPCAsparse)
CreatePathPlot(FPCAsparse)

CreateOutliersPlot(FPCAsparse,optns=list(K=2,variant='KDE'))
CreatePathPlot(FPCAsparse,subset=c(3,5,135),K=3,main="K=3",pch=4)
CreatePathPlot(FPCAsparse,subset=c(3,5,135),K=4,main="K=4",pch=4)

fpca <- as.data.frame(FPCAsparse$xiEst)
fpca <- fpca[,c(1,2)]
fpca$Pltg_ID <- data_wide$Pltg_ID
fpca1 <- FPCAsparse$cumFVE[1]
fpca2 <- FPCAsparse$cumFVE[2]- FPCAsparse$cumFVE[1]

colnames(fpca)[1:2] <- c('FPC1_TNDGR', 'FPC2_TNDGR')

head(fpca)

# Merge TNDGR index value data frame with FPC1 and FPC2 for inspection later
# if desired.
fpca_and_rawscores_TNDGR <- fpca %>% left_join(data_wide, by = 'Pltg_ID')


#### FPCA of only OGR (E1) using manual scores - this is for ANOVA variance components figure ####
df <- read.csv('df1.csv')
df_clean <- df[!is.na(df$Score),]
# Check: 6720-6622=98; 98/14=7 plants omitted

day_DAP <- df_clean %>% distinct(DAT, Flight_Date)

head(df_clean)

# Subset df_clean to only have OGR (E1)
df_clean <- df_clean[df_clean$Test == 'OGR',]

data_wide <- reshape2::dcast(df_clean, Pltg_ID_Key ~ DAT, value.var="Score")
data_wide <- na.omit(data_wide)

L3 <- MakeFPCAInputs(IDs = rep(data_wide$Pltg_ID_Key, each=length(unique(df_clean$DAT)) ),
                     tVec=rep(unique(df_clean$DAT),length(data_wide$Pltg_ID_Key)),
                     t(data_wide[,-1]))


FPCAsparse <- FPCA(L3$Ly, L3$Lt,list(dataType='Sparse',plot=TRUE,
                                     methodMuCovEst='smooth',
                                     userBwCov=2,
                                     methodBwCov = "GCV",
                                     methodBwMu = "GCV"))
plot(FPCAsparse)
CreatePathPlot(FPCAsparse)

CreateOutliersPlot(FPCAsparse,optns=list(K=2,variant='KDE'))
CreatePathPlot(FPCAsparse,subset=c(3,5,135),K=3,main="K=3",pch=4)
CreatePathPlot(FPCAsparse,subset=c(3,5,135),K=4,main="K=4",pch=4)

fpca <- as.data.frame(FPCAsparse$xiEst)
fpca <- fpca[,c(1,2)]
fpca$Pltg_ID_Key <- data_wide$Pltg_ID_Key
fpca1 <- FPCAsparse$cumFVE[1]
fpca2 <- FPCAsparse$cumFVE[2]- FPCAsparse$cumFVE[1]

colnames(fpca)[1:2] <- c('FPC1_Visual_E1', 'FPC2_Visual_E1')

head(fpca)


#### FPCA of only OGR (E1) using RCC - this is for ANOVA variance components figure ####
df <- read.csv('df1.csv')
df_clean <- df[!is.na(df$Score),]
# Check: 6720-6622=98; 98/14=7 plants omitted

day_DAP <- df_clean %>% distinct(DAT, Flight_Date)

head(df_clean)

# Subset df_clean to only have OGR
df_clean <- df_clean[df_clean$Test == 'OGR',]

data_wide <- reshape2::dcast(df_clean, Pltg_ID_Key ~ DAT, value.var="RCC")
data_wide <- na.omit(data_wide)

L3 <- MakeFPCAInputs(IDs = rep(data_wide$Pltg_ID_Key, each=length(unique(df_clean$DAT)) ),
                     tVec=rep(unique(df_clean$DAT),length(data_wide$Pltg_ID_Key)),
                     t(data_wide[,-1]))


FPCAsparse <- FPCA(L3$Ly, L3$Lt,list(dataType='Sparse',plot=TRUE,
                                     methodMuCovEst='smooth',
                                     userBwCov=2,
                                     methodBwCov = "GCV",
                                     methodBwMu = "GCV"))
plot(FPCAsparse)
CreatePathPlot(FPCAsparse)

CreateOutliersPlot(FPCAsparse,optns=list(K=2,variant='KDE'))
CreatePathPlot(FPCAsparse,subset=c(3,5,135),K=3,main="K=3",pch=4)
CreatePathPlot(FPCAsparse,subset=c(3,5,135),K=4,main="K=4",pch=4)


fpca <- as.data.frame(FPCAsparse$xiEst)
fpca <- fpca[,c(1,2)]
fpca$Pltg_ID_Key <- data_wide$Pltg_ID_Key
fpca1 <- FPCAsparse$cumFVE[1]
fpca2 <- FPCAsparse$cumFVE[2]- FPCAsparse$cumFVE[1]

colnames(fpca)[1:2] <- c('FPC1_RCC_E1', 'FPC2_RCC_E1')

head(fpca)


#### FPCA of only OGR (E1) using TNDGR - this is for ANOVA variance components figure ####
df <- read.csv('df1.csv')
df_clean <- df[!is.na(df$Score),]
# Check: 6720-6622=98; 98/14=7 plants omitted

day_DAP <- df_clean %>% distinct(DAT, Flight_Date)

head(df_clean)

# Subset df_clean to only have OGR (E1)
df_clean <- df_clean[df_clean$Test == 'OGR',]

data_wide <- reshape2::dcast(df_clean, Pltg_ID_Key ~ DAT, value.var="TNDGR")
data_wide <- na.omit(data_wide)

L3 <- MakeFPCAInputs(IDs = rep(data_wide$Pltg_ID_Key, each=length(unique(df_clean$DAT)) ),
                     tVec=rep(unique(df_clean$DAT),length(data_wide$Pltg_ID_Key)),
                     t(data_wide[,-1]))


FPCAsparse <- FPCA(L3$Ly, L3$Lt,list(dataType='Sparse',plot=TRUE,
                                     methodMuCovEst='smooth',
                                     userBwCov=2,
                                     methodBwCov = "GCV",
                                     methodBwMu = "GCV"))
plot(FPCAsparse)
CreatePathPlot(FPCAsparse)

CreateOutliersPlot(FPCAsparse,optns=list(K=2,variant='KDE'))
CreatePathPlot(FPCAsparse,subset=c(3,5,135),K=3,main="K=3",pch=4)
CreatePathPlot(FPCAsparse,subset=c(3,5,135),K=4,main="K=4",pch=4)

fpca <- as.data.frame(FPCAsparse$xiEst)
fpca <- fpca[,c(1,2)]
fpca$Pltg_ID_Key <- data_wide$Pltg_ID_Key
fpca1 <- FPCAsparse$cumFVE[1]
fpca2 <- FPCAsparse$cumFVE[2]- FPCAsparse$cumFVE[1]

colnames(fpca)[1:2] <- c('FPC1_TNDGR_E1', 'FPC2_TNDGR_E1')

head(fpca)


#### ANOVA of FPC1 scores for visual senescence ratings, RCC, and TNDGR for E1 data ####
# If analyzing according to Range/Row design, there should be 40 ranges and 6 rows
# Intersection of a Range/Row is a plant. Thus, 40 x 6 = 240 plants total
# Therefore, use columns: Range and Row2 since Row2 repeats between 112 to 117
# You can verify this with table(b$Range) and table(b$Row2)

# Data frame that contains FPC1 values of visual senescence scores for E1
df <- read.csv('df2.csv')
df <- df[!is.na(df$Score),]
df <- df[df$DAT == 97,]
df <- df[order(df$Pltg_ID),]

# Data frame that contains FPC1 values of RCC and TNDGR for E1
a <- read.csv('df3.csv')
a <- a[order(a$Pltg_ID),]

# Merge to put FPC1 scores of visual, RCC, and TNDGR senescence scores
# into the same data frame (which we're calling b).
b <- df %>% left_join(a, by = 'Pltg_ID')

setequal(df$Pltg_ID, a$Pltg_ID)

# Make sure all categorical variables are set as factors.
b$Range <- factor(b$Range)
b$Range2 <- factor(b$Range2)
b$Row <- factor(b$Row)
b$Row2 <- factor(b$Row2)
b$Rep <- factor(b$Rep)
b$Pedigree <- factor(b$Pedigree)
b$DAT <- factor(b$DAT)

# Make sure all response variables are set as numerics.
b$Score <- as.numeric(b$Score)
b$FPC1_RCC_OGR <- as.numeric(b$FPC1_RCC_OGR)
b$FPC1_TNDGR_OGR <- as.numeric(b$FPC1_TNDGR_OGR)


#### Analyze FPC1 as response variable for Visual Score ####
model <- lme4::lmer(b$FPC1 ~ (1|Pedigree) + (1|Range) + (1|Row2) + (1|Rep), b)

rmse <- RMSE.merMod(model)

R<- MuMIn::r.squaredGLMM(model)[,2]
R

VC<-as.data.frame(print(VarCorr(model ), comp=c("Variance")))
VC$Percent<-VC$vcov/sum(VC$vcov)
VC

repeatability <- VC[1,6] / (VC[1,6] + (VC[5,6]/5)) # Note that Pedigree is in VC[2,6] and residual is in VC[6,6]
round(repeatability,3)

VC[,7] <-rmse
VC[,8] <-repeatability
VC[,9] <-R
VC[,10] <- 'FPC1 Visual Score'
names(VC)[7:10] <- c("RMSE","Repeatability","RSquared","Trait")

# Variance components for ANOVA of visual score FPC1
VC


#### Analyze FPC1 as response variable for RCC ####
model <- lme4::lmer(b$FPC1_RCC_OGR ~ (1|Pedigree) + (1|Range) + (1|Row2) + (1|Rep), b)

rmse <- RMSE.merMod(model)

R<- MuMIn::r.squaredGLMM(model)[,2]
R

VC<-as.data.frame(print(VarCorr(model ), comp=c("Variance")))
VC$Percent<-VC$vcov/sum(VC$vcov)
VC

repeatability <- VC[1,6] / (VC[1,6] + (VC[5,6]/5)) # Note that Pedigree is in VC[2,6] and residual is in VC[6,6]
round(repeatability,3)

VC[,7] <-rmse
VC[,8] <-repeatability
VC[,9] <-R
VC[,10] <- 'FPC1 RCC'
names(VC)[7:10] <- c("RMSE","Repeatability","RSquared","Trait")

# Variance components for ANOVA of RCC FPC1
VC


#### Analyze FPC1 as response variable for TNDGR ####
# Add rescaled FPC1 for ANOVA
model <- lme4::lmer(b$FPC1_TNDGR_OGR ~ (1|Pedigree) + (1|Range) + (1|Row2) + (1|Rep), b)

rmse <- RMSE.merMod(model)

R<- MuMIn::r.squaredGLMM(model)[,2]
R

VC<-as.data.frame(print(VarCorr(model ), comp=c("Variance")))
VC$Percent<-VC$vcov/sum(VC$vcov)
VC

repeatability <- VC[1,6] / (VC[1,6] + (VC[5,6]/5)) # Note that Pedigree is in VC[2,6] and residual is in VC[6,6]
round(repeatability,3)

VC[,7] <-rmse
VC[,8] <-repeatability
VC[,9] <-R
VC[,10] <- 'FPC1 TNDGR'
names(VC)[7:10] <- c("RMSE","Repeatability","RSquared","Trait")

# Variance components for ANOVA of TNDGR FPC1
VC


#### ANOVA of FPC1 scores for visual senescence ratings, RCC, and TNDGR for E1 data from deep learning regression output values ####
# Note: these FPC1 scores were obtained from CNN regression output from M2, M4, and M6. These were the top-performing 
# models within each senescence metric (visual scores, RCC, and TNDGR). CNN regression was performed in this case
# by using a 50/50 split within E1 using the Optuna-derived optimal hyperparameters for each model. FPC1 scores
# from CNN regression were averaged from 25 replications of random 50/50 train/test splits.

# If analyzing according to Range/Row design, there should be 40 ranges and 6 rows
# Intersection of a Range/Row is a plant. Thus, 40 x 6 = 240 plants total
# Therefore, use columns: Range and Row2 since Row2 repeats between 112 to 117
# You can verify this with table(b$Range) and table(b$Row2)

# Data frame that contains range, row, and rep values for E1
df <- read.csv('df2.csv')
df <- df[!is.na(df$Score),]
df <- df[df$DAT == 97,]
df <- df[order(df$Pltg_ID),]
df$Pltg_ID_Key_JPG <- gsub('.tif', '.jpg', df$Pltg_ID_Key)

# Data frame that contains FPC1 values of visual senescence scores for E1 from deep learning output
# Visual Scores
v <- read.csv('M4_Visual_Means_E1.csv')
v <- v[order(v$Pltg_ID_Key_JPG),]
# RCC
r <- read.csv('M2_RCC_ReLU_Means_E1.csv')
r <- r[order(r$Pltg_ID_Key_JPG),]
# TNDGR
t <- read.csv('M6_TNDGR_Means_E1.csv')
t <- t[order(t$Pltg_ID_Key_JPG),]

vr <- v %>% left_join(r, by = 'Pltg_ID_Key_JPG')

vrt <- vr %>% left_join(t, by = 'Pltg_ID_Key_JPG')

# Merge to put FPC1 scores of visual, RCC, and TNDGR senescence scores
# into the same data frame (which we're calling b).
b <- df %>% left_join(vrt, by = 'Pltg_ID_Key_JPG')

setequal(df$Pltg_ID_Key_JPG, vrt$Pltg_ID_Key_JPG)

# Make sure all categorical variables are set as factors.
b$Range <- factor(b$Range)
b$Range2 <- factor(b$Range2)
b$Row <- factor(b$Row)
b$Row2 <- factor(b$Row2)
b$Rep <- factor(b$Rep)
b$Pedigree <- factor(b$Pedigree)
b$DAT <- factor(b$DAT)

# Make sure all response variables are set as numerics.
b$Predicted_Mean_Visual <- as.numeric(b$Predicted_Mean_Visual)
b$Predicted_Mean_RCC <- as.numeric(b$Predicted_Mean_RCC)
b$Predicted_Mean_TNDGR <- as.numeric(b$Predicted_Mean_TNDGR)

#### Analyze FPC1 as response variable for Visual Score from CNN regression output ####
model <- lme4::lmer(b$Predicted_Mean_Visual ~ (1|Pedigree) + (1|Range) + (1|Row2) + (1|Rep), b)

rmse <- RMSE.merMod(model)

R<- MuMIn::r.squaredGLMM(model)[,2]
R

VC<-as.data.frame(print(VarCorr(model ), comp=c("Variance")))
VC$Percent<-VC$vcov/sum(VC$vcov)
VC

repeatability <- VC[1,6] / (VC[1,6] + (VC[5,6]/5)) # Note that Pedigree is in VC[2,6] and residual is in VC[6,6]
round(repeatability,3)

VC[,7] <-rmse
VC[,8] <-repeatability
VC[,9] <-R
VC[,10] <- 'FPC1 Visual (CNN Output)'
names(VC)[7:10] <- c("RMSE","Repeatability","RSquared","Trait")

# Variance components for ANOVA of visual score FPC1
VC


#### Analyze FPC1 as response variable for RCC from CNN regression output ####
model <- lme4::lmer(b$Predicted_Mean_RCC ~ (1|Pedigree) + (1|Range) + (1|Row2) + (1|Rep), b)

rmse <- RMSE.merMod(model)

R<- MuMIn::r.squaredGLMM(model)[,2]
R

VC<-as.data.frame(print(VarCorr(model ), comp=c("Variance")))
VC$Percent<-VC$vcov/sum(VC$vcov)
VC

repeatability <- VC[1,6] / (VC[1,6] + (VC[5,6]/5)) # Note that Pedigree is in VC[2,6] and residual is in VC[6,6]
round(repeatability,3)

VC[,7] <-rmse
VC[,8] <-repeatability
VC[,9] <-R
VC[,10] <- 'FPC1 RCC (CNN Output)'
names(VC)[7:10] <- c("RMSE","Repeatability","RSquared","Trait")

# Variance components for ANOVA of visual score FPC1
VC


#### Analyze FPC1 as response variable for TNDGR from CNN regression output ####
model <- lme4::lmer(b$Predicted_Mean_TNDGR ~ (1|Pedigree) + (1|Range) + (1|Row2) + (1|Rep), b)

rmse <- RMSE.merMod(model)

R<- MuMIn::r.squaredGLMM(model)[,2]
R

VC<-as.data.frame(print(VarCorr(model ), comp=c("Variance")))
VC$Percent<-VC$vcov/sum(VC$vcov)
VC

repeatability <- VC[1,6] / (VC[1,6] + (VC[5,6]/5)) # Note that Pedigree is in VC[2,6] and residual is in VC[6,6]
round(repeatability,3)

VC[,7] <-rmse
VC[,8] <-repeatability
VC[,9] <-R
VC[,10] <- 'FPC1 TNDGR (CNN Output)'
names(VC)[7:10] <- c("RMSE","Repeatability","RSquared","Trait")

# Variance components for ANOVA of visual score FPC1
VC