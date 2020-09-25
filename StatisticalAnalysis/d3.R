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


# Prepare dataset ---------------------------------------------------------

# Load file 
df_power = read.csv("../166-d3-powers.csv")
df_power = df_power[is.finite(df_power$Power),]
df_phase = read.csv("../166-d3-phases.csv")

# Fix column ordering
df_power <- transform(df_power, Band=factor(Band,levels=c("Theta","Mu","Beta","Gamma")))
df_power <- transform(df_power, EEG=factor(EEG,levels=c("Raw","Hjorth","Average")))
df_power <- transform(df_power, Method=factor(Method,levels=c("FFT","Welch","Burg")))
df_power <- transform(df_power, ArtifactRemoved=factor(ArtifactRemoved,levels=c("True","False")))
df_phase <- transform(df_phase, Band=factor(Band,levels=c("Theta","Mu","Beta","Gamma")))
df_phase <- transform(df_phase, EEG=factor(EEG,levels=c("Raw","Hjorth","Average")))
df_phase <- transform(df_phase, ArtifactRemoved=factor(ArtifactRemoved,levels=c("True","False")))

# Transform some columns
df_power$sub = factor(df_power$sub)
df_power$trial_abs = factor(df_power$trial)
df_power$Method = factor(df_power$Method)
df_power$Band = factor(df_power$Band)
df_power$Filter = factor(df_power$Filter)
df_power$Time = factor(df_power$Time)
df_power$EEG = factor(df_power$EEG)
df_power$ArtifactRemoved = factor(df_power$ArtifactRemoved)

df_phase$sub = factor(df_phase$sub)
df_phase$trial_abs = factor(df_phase$trial)
df_phase$Band = factor(df_phase$Band)
df_phase$Filter = factor(df_phase$Filter)
df_phase$EEG = factor(df_phase$EEG)
df_phase$ArtifactRemoved = factor(df_phase$ArtifactRemoved)

# For post-hoc pairwise comparison
contrasts(df_power$sub) <- "contr.sum"
contrasts(df_power$trial_abs) <- "contr.sum"
contrasts(df_power$Method) <- "contr.sum"
contrasts(df_power$Band) <- "contr.sum"
contrasts(df_power$Filter) <- "contr.sum"
contrasts(df_power$Time) <- "contr.sum"
contrasts(df_power$EEG) <- "contr.sum"
contrasts(df_power$ArtifactRemoved) <- "contr.sum"

contrasts(df_phase$sub) <- "contr.sum"
contrasts(df_phase$trial_abs) <- "contr.sum"
contrasts(df_phase$Band) <- "contr.sum"
contrasts(df_phase$Filter) <- "contr.sum"
contrasts(df_phase$EEG) <- "contr.sum"
contrasts(df_phase$ArtifactRemoved) <- "contr.sum"

# LMM: Power ~ factors ---------------------------------------------------------

# Default values
power_def_variables = c('EEG', 'Filter', 'Time', 'Method', 'ArtifactRemoved')
power_def_values = c('Raw', 'Butterworth', -750, 'Welch', 'True')

# Power ~ Factor + (1|sub) + (1|band)
result <- data.frame(
  band=rep("",3),
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
  for (band in c("Theta", "Mu", "Beta", "Gamma")) {
    df2 <- subset(df_power, Band==band)
    j = 1
    for (var2 in power_def_variables) {
      if (var != var2) {
        df2 <- subset(df2, get(var2)==power_def_values[j])
        print(paste(var2, '=', power_def_values[j], ', nrow =', nrow(df2)))
      }
      j = j + 1
    }
    m = lmer(Power ~ get(var)  + (1|sub), data=df2)
    m = anova(m)
    result[modelNo, ] <- list(band, modelNo, nrow(df2), var, round(m[1, 6], 4), round(m[1, 5], 2), round(m[1, 3], 2), round(m[1, 4], 2))
    modelNo <- modelNo + 1
  }
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
  band=rep("", 3),
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
    
    for (band in c("Theta", "Mu", "Beta", "Gamma")) {
      df3 <- subset(df2, Band==band)
      j = 1
      for (var2 in phase_def_variables) {
        if (var != var2) {
          df3 <- subset(df3, get(var2)==phase_def_values[j])
          print(paste(var2, '=', phase_def_values[j], ', nrow =', nrow(df3)))
        }
        j = j + 1
      }
      m = lmer(Phase ~ get(var)  + (1|sub), data=df3)
      m = anova(m)
      result[modelNo, ] <- list(band, modelNo, phase_type, nrow(df3), var, round(m[1, 6], 4), round(m[1, 5], 2), round(m[1, 3], 2), round(m[1, 4], 2))
      modelNo <- modelNo + 1
    }
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
