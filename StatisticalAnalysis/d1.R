# Libraries ---------------------------------------------------------------
# Load libraries
library(readxl)
library(BBmisc)
library(lme4)
library(lmerTest)
library(devtools)
library(car)
library(fitdistrplus)
require("lattice")
library(nortest)
library(vcd)
require(car)
require(MASS)
require(broom)
library(tidyverse)
library(ggpubr)
library(rstatix)
library(plyr)
library(ez)
library(ggplot2)
library(multcomp) # for glht
library(emmeans) # for emm

# Prepare power and phase dataset ---------------------------------------------------------

# Load powers
df_power = read.csv("../164-d1-powers.csv")
df_phase = read.csv("../164-d1-phases.csv")

# Fix column ordering
df_power <- transform(df_power, Band=factor(Band,levels=c("Theta","Mu","Beta","Gamma")))
df_power <- transform(df_power, Method=factor(Method,levels=c("FFT","Welch","Burg")))
df_power <- transform(df_power, EEG=factor(EEG,levels=c("Raw","Hjorth","Averaged")))

# Transform some columns
df_power$sub = factor(df_power$sub)
df_power$trial_abs = factor(df_power$trial_abs)
df_power$Method = factor(df_power$Method)
df_power$Band = factor(df_power$Band)
df_power$Filter = factor(df_power$Filter)
df_power$Time = factor(df_power$Time)
df_power$EEG = factor(df_power$EEG)
df_power$Resampled = factor(df_power$Resampled)
df_power$ArtifactRemoved = factor(df_power$ArtifactRemoved)

# For post-hoc pairwise comparison
# contrasts(df_power$sub) <- "contr.sum"
contrasts(df_power$trial_abs) <- "contr.sum"
contrasts(df_power$Method) <- "contr.sum"
contrasts(df_power$Band) <- "contr.sum"
contrasts(df_power$Filter) <- "contr.sum"
contrasts(df_power$Time) <- "contr.sum"
contrasts(df_power$EEG) <- "contr.sum"
contrasts(df_power$Resampled) <- "contr.sum"
contrasts(df_power$ArtifactRemoved) <- "contr.sum"

# Convert to log
df_power$mep_size_log = log(df_power$mep_size)
df_power$mep_latency_log = log(df_power$mep_latency)
df_power$mep_duration_log = log(df_power$mep_duration)
df_power$mep_area_log = log(df_power$mep_area)

# Load phases
# Fix column ordering
df_phase <- transform(df_phase, Band=factor(Band,levels=c("Theta","Mu","Beta","Gamma")))
df_phase <- transform(df_phase, EEG=factor(EEG,levels=c("Raw","Hjorth","Averaged")))

# Transform some columns
df_phase$sub = factor(df_phase$sub)
df_phase$trial_abs = factor(df_phase$trial_abs)
df_phase$Band = factor(df_phase$Band)
df_phase$Filter = factor(df_phase$Filter)
df_phase$EEG = factor(df_phase$EEG)
df_phase$ArtifactRemoved = factor(df_phase$ArtifactRemoved)

# For post-hoc pairwise comparison
# contrasts(df_phase$sub) <- "contr.sum"
contrasts(df_phase$trial_abs) <- "contr.sum"
contrasts(df_phase$Band) <- "contr.sum"
contrasts(df_phase$Filter) <- "contr.sum"
contrasts(df_phase$EEG) <- "contr.sum"
contrasts(df_phase$ArtifactRemoved) <- "contr.sum"


# LMM: Power ~ factors ---------------------------------------------------------

# Default values
power_def_variables = c('ArtifactRemoved', 'EEG', 'Resampled', 'Filter', 'Time', 'Method')
power_def_values = c('True', 'Raw', 'False', 'Butterworth', -750, 'Welch')

# Power ~ Factor + (1|sub) + (1|band)
result <- data.frame(
  modelNo=rep("",3), 
  obsCount=rep("", 3),
  factor=rep("", 3),
  p=rep("", 3), 
  f=rep("", 3), 
  dfn=rep("", 3), 
  dfd=rep("", 3), 
  stringsAsFactors=FALSE)
modelNo <- 1
for (var in power_def_variables) {
  df2 = df_power
  j = 1
  for (var2 in power_def_variables) {
    if (var != var2) {
      df2 <- subset(df2, get(var2)==power_def_values[j])
      print(paste(var2, '=', power_def_values[j], ', nrow =', nrow(df2)))
    }
    j = j + 1
  }
  m = lmer(Power ~ get(var)  + (1|sub/Band), data=df2)
  m = anova(m)
  result[modelNo, ] <- list(modelNo, nrow(df2), var, round(m[1, 6], 4), round(m[1, 5], 2), round(m[1, 3], 2), round(m[1, 4], 2))
  modelNo <- modelNo + 1
}


# LMM: Power ~ Interaction -----------------------------------------------------

result <- data.frame(
  modelNo=rep("",3), 
  obsCount=rep("", 3),
  factor=rep("", 3),
  p=rep("", 3), 
  f=rep("", 3), 
  dfn=rep("", 3), 
  dfd=rep("", 3), 
  stringsAsFactors=FALSE)
modelNo <- 1
combs <- combn(power_def_variables, 2)
for (var in 1:ncol(combs)) {
  var = combs[,var]
  df2 = df_power
  j = 1
  for (var2 in power_def_variables) {
    if (var[1] != var2 && var[2] != var2) {
      df2 <- subset(df2, get(var2)==power_def_values[j])
      print(paste(var2, '=', power_def_values[j], ', nrow =', nrow(df2)))
    }
    j = j + 1
  }
  m = lmer(Power ~ (get(var[1]) * get(var[2])) + (1|sub/Band), data=df2)
  m = anova(m)
  result[modelNo, ] <- list(modelNo, nrow(df2), paste(var[1], var[2], sep=":"), round(m[3, 6], 4), round(m[1, 5], 2), round(m[1, 3], 2), round(m[1, 4], 2))
  modelNo <- modelNo + 1
}


# LMM: Phase ~ Factors ----------------------------------------------------

# Default values
phase_def_variables = c('ArtifactRemoved', 'EEG', 'Filter')
phase_def_values = c('True', 'Raw', 'Butterworth')

# Phase ~ Factor + (1|sub) + (1|band)
result <- data.frame(
  modelNo=rep("",3), 
  phase=rep("", 3),
  obsCount=rep("", 3),
  factor=rep("", 3),
  p=rep("", 3), 
  f=rep("", 3), 
  dfn=rep("", 3), 
  dfd=rep("", 3), 
  stringsAsFactors=FALSE)
modelNo <- 1
for (phase_type in c('peak', 'trough')) {
  if (phase_type == 'peak') {
    df2 = subset(df_phase, Phase <= 180)
  }
  else {
    df2 = subset(df_phase, Phase > 180)
  }
  for (var in phase_def_variables) {
    df3 = df2
    j = 1
    for (var2 in phase_def_variables) {
      if (var != var2) {
        df3 <- subset(df3, get(var2)==phase_def_values[j])
        print(paste(var2, '=', phase_def_values[j], ', nrow =', nrow(df3)))
      }
      j = j + 1
    }
    m = lmer(Phase ~ get(var)  + (1|sub/Band), data=df3)
    m = anova(m)
    result[modelNo, ] <- list(modelNo, phase_type, nrow(df3), var, round(m[1, 6], 4), round(m[1, 5], 2), round(m[1, 3], 2), round(m[1, 4], 2))
    modelNo <- modelNo + 1
  }
}


# LMM: Phase ~ Interaction ------------------------------------------------

# Phase ~ Factor + (1|sub) + (1|band)
result <- data.frame(
  modelNo=rep("",3), 
  phase=rep("",3),
  obsCount=rep("", 3),
  factor=rep("", 3),
  p=rep("", 3), 
  f=rep("", 3), 
  dfn=rep("", 3), 
  dfd=rep("", 3), 
  stringsAsFactors=FALSE)
modelNo <- 1
combs <- combn(phase_def_variables, 2)
for (phase_type in c('peak', 'trough')) {
  if (phase_type == 'peak') {
    df2 = subset(df_phase, Phase <= 180)
  }
  else {
    df2 = subset(df_phase, Phase > 180)
  }
  for (var in 1:ncol(combs)) {
    var = combs[,var]
    df3 = df2
    j = 1
    for (var2 in phase_def_variables) {
      if (var[1] != var2 && var[2] != var2) {
        df3 <- subset(df3, get(var2)==phase_def_values[j])
        print(paste(var2, '=', phase_def_values[j], ', nrow=', nrow(df3)))
      }
      j = j + 1
    }
    m = lmer(Phase ~ (get(var[1]) * get(var[2]))  + (1|sub/Band), data=df3)
    m = anova(m)
    result[modelNo, ] <- list(modelNo, phase_type, nrow(df3), paste(var[1], var[2], sep=":"), round(m[3, 6], 4), round(m[1, 5], 2), round(m[1, 3], 2), round(m[1, 4], 2))
    modelNo <- modelNo + 1
  }
}

# Combine powers and phases for MEP LMM -----------------------------------------------

df_mep = subset(df_power, Resampled=='True')
df_mep = subset(df_mep, ArtifactRemoved=='True')
df_mep = subset(df_mep, Method=='Welch')
df_mep = subset(df_mep, Filter=='Butterworth')
df_mep = subset(df_mep, EEG=='Averaged')
df_mep = subset(df_mep, Time=='-150')

df_mep$Phase <- 0
pb = txtProgressBar(min = 0, max = nrow(df_mep), initial = 0, style=3)
for (i in 1:nrow(df_mep)) {
  df_test = subset(df_phase, ArtifactRemoved==df_mep$ArtifactRemoved[i])
  df_test = subset(df_test, Filter==df_mep$Filter[i])
  df_test = subset(df_test, EEG==df_mep$EEG[i])
  df_test = subset(df_test, Band==df_mep$Band[i])
  df_test = subset(df_test, sub==df_mep$sub[i])
  df_test = subset(df_test, trial_abs==df_mep$trial_abs[i])
  df_mep$Phase[i] <- df_test$Phase
  if (nrow(df_test) != 1) {
    print(paste('Error in ', i))
    break
  }
  setTxtProgressBar(pb, i)
}
close(pb)

# LMM: MEP ~ Power * Phase  ---------------------------------

result <- data.frame(target=rep("", 5), band=rep("", 5), input=rep("", 5), p=rep(0, 5), b=rep(0, 5), no_of_obs=rep(0, 5), stringsAsFactors=FALSE)
i <- 1
for (variable in c("mep_size_log", "mep_latency_log", "mep_duration_log", 'mep_area_log')) {
  for (band in c('Theta', 'Mu', 'Beta', 'Gamma')) {
    df_mep2 = subset(df_mep, Band==band)
    df_mep2 <- subset(df_mep2, (Phase > 45 & Phase < 135) | (Phase > 225 & Phase < 315))
    df_mep2$Phase_bin <- factor(cut(df_mep2$Phase, breaks=c(0,180,360), labels=c(1, 0)))

    formula <- paste(variable, ' ~ (Power * Phase_bin) + (1|sub)', sep='')
    m = lmer(as.formula(formula), data=df_mep2)
    m = summary(m)

    result[i, ] <- list(variable, band, 'power', round(m$coefficients[2, 5], 4), round(m$coefficients[2, 1], 4), nrow(df_power2))
    result[i+1, ] <- list(variable, band, 'phase', round(m$coefficients[3, 5], 4), round(m$coefficients[3, 1], 4), nrow(df_power2))
    result[i+2, ] <- list(variable, band, 'interaction', round(m$coefficients[4, 5], 4), round(m$coefficients[4, 1], 4), nrow(df_power2))
    i=i+3
  }
}