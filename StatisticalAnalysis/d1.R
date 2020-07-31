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

# Prepare ggplot theme ----------------------------------------------------------
theme_ms <- function(base_size=12, base_family="Helvetica") {
  library(grid)
  (theme_bw(base_size = base_size, base_family = base_family)+
      theme(text=element_text(color="black"),
            axis.title=element_text(face="bold", size = rel(1.3)),
            axis.text=element_text(size = rel(1), color = "black"),
            legend.title=element_text(face="bold"),
            legend.text=element_text(face="bold"),
            legend.background=element_rect(fill="transparent"),
            legend.key.size = unit(0.8, 'lines'),
            panel.border=element_rect(color="black",size=1),
            panel.grid=element_blank()
      ))
}


# Prepare power dataset ---------------------------------------------------------

# Load powers
df_power = read_excel("153-d1-power-long.xlsx")
df_power$Resampled = TRUE
df_power$ArtifactRemoved = TRUE
df = read_excel("158-d1-power-long.xlsx")
df$Resampled = FALSE
df$ArtifactRemoved = TRUE
df_power = bind_rows(df_power, df)
df = read_excel("159-d1-power-long.xlsx")
df$Resampled = TRUE
df$ArtifactRemoved = FALSE
df_power = bind_rows(df_power, df)
df = read_excel("161-d1-power-long.xlsx")
df$Resampled = FALSE
df$ArtifactRemoved = FALSE
df_power = bind_rows(df_power, df)
df_power$Filter[df_power$Filter == "Blackman-Harris"] <- "Blackmann-Harris"

# Fix column ordering
df_power <- transform(df_power, Band=factor(Band,levels=c("Theta","Mu","Beta","Gamma")))
df_power <- transform(df_power, Method=factor(Method,levels=c("FFT","Welch","Burg")))
df_power <- transform(df_power, EEG=factor(EEG,levels=c("Raw","Hjorth","Average")))

# Transform some columns
df_power$sub = factor(df_power$sub)
df_power$trial = factor(df_power$trial_abs)
df_power$Method = factor(df_power$Method)
df_power$Band = factor(df_power$Band)
df_power$Filter = factor(df_power$Filter)
df_power$Time = factor(df_power$Time)
df_power$EEG = factor(df_power$EEG)
df_power$Resampled = factor(df_power$Resampled)
df_power$ArtifactRemoved = factor(df_power$ArtifactRemoved)

# For post-hoc pairwise comparison
contrasts(df_power$sub) <- "contr.sum"
contrasts(df_power$trial) <- "contr.sum"
contrasts(df_power$Method) <- "contr.sum"
contrasts(df_power$Band) <- "contr.sum"
contrasts(df_power$Filter) <- "contr.sum"
contrasts(df_power$Time) <- "contr.sum"
contrasts(df_power$EEG) <- "contr.sum"
contrasts(df_power$Resampled) <- "contr.sum"
contrasts(df_power$ArtifactRemoved) <- "contr.sum"


# TMS-EEG - LMM - Power ~ variables ---------------------------------

# Default values
def_variables = c('ArtifactRemoved', 'EEG', 'Resampled', 'Filter', 'Time', 'Method')
def_values = c(TRUE, 'Raw', FALSE, 'Butterworth', -750, 'Welch')

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
for (var in def_variables) {
  df2 = df_power
  j = 1
  for (var2 in def_variables) {
    if (var != var2) {
      df2 <- subset(df2, get(var2)==def_values[j])
    }
    j = j + 1
  }
  m = lmer(value ~ get(var)  + (1|sub/Band), data=df2)
  m = anova(m)
  result[modelNo, ] <- list(modelNo, nrow(df2), var, round(m[1, 6], 4), round(m[1, 5], 2), round(m[1, 3], 2), round(m[1, 4], 2))
  modelNo <- modelNo + 1
}

# TMS-EEG - LMM - Power ~ interaction ---------------------------------

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
combs <- combn(def_variables, 2)
for (var in 1:ncol(combs)) {
  var = combs[,var]
  df2 = df_power
  j = 1
  for (var2 in def_variables) {
    if (var[1] != var2 && var[2] != var2) {
      df2 <- subset(df2, get(var2)==def_values[j])
    }
    j = j + 1
  }
  m = lmer(value ~ (get(var[1]) * get(var[2])) + (1|sub/Band), data=df2)
  m = anova(m)
  result[modelNo, ] <- list(modelNo, nrow(df2), paste(var[1], var[2], sep=":"), round(m[3, 6], 4), round(m[1, 5], 2), round(m[1, 3], 2), round(m[1, 4], 2))
  modelNo <- modelNo + 1
}

# Prepare phase dataset ---------------------------------------------------

# Load phases
df_phase = read_excel("153-d1-phase-long.xlsx")
df_phase$Resampled = TRUE
df_phase$ArtifactRemoved = TRUE
df = read_excel("158-d1-phase-long.xlsx")
df$Resampled = FALSE
df$ArtifactRemoved = TRUE
df_phase = bind_rows(df_phase, df)
df = read_excel("159-d1-phase-long.xlsx")
df$Resampled = TRUE
df$ArtifactRemoved = FALSE
df_phase = bind_rows(df_phase, df)
df = read_excel("161-d1-phase-long.xlsx")
df$Resampled = FALSE
df$ArtifactRemoved = FALSE
df_phase = bind_rows(df_phase, df)
df_phase$Filter[df_phase$Filter == "Blackman-Harris"] <- "Blackmann-Harris"

# Fix column ordering
df_phase <- transform(df_phase, Band=factor(Band,levels=c("Theta","Mu","Beta","Gamma")))
df_phase <- transform(df_phase, EEG=factor(EEG,levels=c("Raw","Hjorth","Average")))

# Transform some columns
df_phase$sub = factor(df_phase$sub)
df_phase$trial = factor(df_phase$trial_abs)
df_phase$Band = factor(df_phase$Band)
df_phase$Filter = factor(df_phase$Filter)
df_phase$EEG = factor(df_phase$EEG)
df_phase$ArtifactRemoved = factor(df_phase$ArtifactRemoved)

# For post-hoc pairwise comparison
contrasts(df_phase$sub) <- "contr.sum"
contrasts(df_phase$trial) <- "contr.sum"
contrasts(df_phase$Band) <- "contr.sum"
contrasts(df_phase$Filter) <- "contr.sum"
contrasts(df_phase$EEG) <- "contr.sum"
contrasts(df_phase$ArtifactRemoved) <- "contr.sum"



# TMS-EEG - LMM - Phase - Variables ---------------------------------

# Default values
def_variables = c('ArtifactRemoved', 'EEG', 'Filter')
def_values = c(TRUE, 'Raw', 'Butterworth')

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
    df2 = subset(df_phase, value <= 180)
  }
  else {
    df2 = subset(df_phase, value > 180)
  }
  for (var in def_variables) {
    df3 = df2
    j = 1
    for (var2 in def_variables) {
      if (var != var2) {
        df3 <- subset(df3, get(var2)==def_values[j])
      }
      j = j + 1
    }
    m = lmer(value ~ get(var)  + (1|sub/Band), data=df3)
    m = anova(m)
    result[modelNo, ] <- list(modelNo, phase_type, nrow(df3), var, round(m[1, 6], 4), round(m[1, 5], 2), round(m[1, 3], 2), round(m[1, 4], 2))
    modelNo <- modelNo + 1
  }
}

# Linear mixed effect model for interactions ---------------------------------

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
combs <- combn(def_variables, 2)
for (phase_type in c('peak', 'trough')) {
  if (phase_type == 'peak') {
    df2 = subset(df_phase, value <= 180)
  }
  else {
    df2 = subset(df_phase, value > 180)
  }
  for (var in 1:ncol(combs)) {
    var = combs[,var]
    df3 = df2
    j = 1
    for (var2 in def_variables) {
      if (var[1] != var2 && var[2] != var2) {
        print(paste(var2, '=', def_values[j], 'for constant', var[1], 'and', var[2]))
        df3 <- subset(df3, get(var2)==def_values[j])
      }
      j = j + 1
    }
    m = lmer(value ~ (get(var[1]) * get(var[2]))  + (1|sub/Band), data=df3)
    m = anova(m)
    result[modelNo, ] <- list(modelNo, phase_type, nrow(df3), paste(var[1], var[2], sep=":"), round(m[3, 6], 4), round(m[1, 5], 2), round(m[1, 3], 2), round(m[1, 4], 2))
    modelNo <- modelNo + 1
  }
}


# Load alcohol powers and phases ------------------------------------------

# Load powers
df_power = read_excel("157-alc-power-long.xlsx")
df_phase = read_excel("157-alc-phase-long.xlsx")
df_power = subset(df_power, alcholic==FALSE)
df_phase = subset(df_phase, alcholic==FALSE)

# Column ordering
df_power <- transform(df_power, Band=factor(Band,levels=c("Theta","Mu","Beta","Gamma")))
df_power <- transform(df_power, Method=factor(Method,levels=c("FFT","Welch","Burg")))
df_power <- transform(df_power, EEG=factor(EEG,levels=c("Raw","Hjorth","Average")))
df_phase <- transform(df_phase, Band=factor(Band,levels=c("Theta","Mu","Beta","Gamma")))
df_phase <- transform(df_phase, EEG=factor(EEG,levels=c("Raw","Hjorth","Average")))


# Transform some columns
df_power$sub = factor(df_power$sub)
df_power$trial = factor(df_power$trial)
df_power$Method = factor(df_power$Method)
df_power$Band = factor(df_power$Band)
df_power$Filter = factor(df_power$Filter)
df_power$Time = factor(df_power$Time)
df_power$EEG = factor(df_power$EEG)

df_phase$sub = factor(df_phase$sub)
df_phase$trial = factor(df_phase$trial)
df_phase$Band = factor(df_phase$Band)
df_phase$Filter = factor(df_phase$Filter)
df_phase$EEG = factor(df_phase$EEG)

# For post-hoc pairwise comparison
contrasts(df_power$sub) <- "contr.sum"
contrasts(df_power$trial) <- "contr.sum"
contrasts(df_power$Method) <- "contr.sum"
contrasts(df_power$Band) <- "contr.sum"
contrasts(df_power$Filter) <- "contr.sum"
contrasts(df_power$Time) <- "contr.sum"
contrasts(df_power$EEG) <- "contr.sum"

contrasts(df_phase$sub) <- "contr.sum"
contrasts(df_phase$trial) <- "contr.sum"
contrasts(df_phase$Band) <- "contr.sum"
contrasts(df_phase$Filter) <- "contr.sum"
contrasts(df_phase$EEG) <- "contr.sum"

# UCI - Power - Linear mixed model ------------------------------------------------

# Default values
def_variables = c('EEG', 'Filter', 'Time', 'Method')
def_values = c('Raw', 'Butterworth', -750, 'Welch')

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
for (var in def_variables) {
  df2 = df_power
  j = 1
  for (var2 in def_variables) {
    if (var != var2) {
      df2 <- subset(df2, get(var2)==def_values[j])
    }
    j = j + 1
  }
  m = lmer(value ~ get(var)  + (1|sub) + (1|Band), data=df2)
  m = anova(m)
  result[modelNo, ] <- list(modelNo, nrow(df2), var, round(m[1, 6], 4), round(m[1, 5], 2), round(m[1, 3], 2), round(m[1, 4], 2))
  modelNo <- modelNo + 1
}


# UCI LMM - Power - Interaction ---------------------------------

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
combs <- combn(def_variables, 2)
for (var in 1:ncol(combs)) {
  var = combs[,var]
  df2 = df_power
  j = 1
  for (var2 in def_variables) {
    if (var[1] != var2 && var[2] != var2) {
      df2 <- subset(df2, get(var2)==def_values[j])
    }
    j = j + 1
  }
  m = lmer(value ~ (get(var[1]) * get(var[2])) + (1|sub) + (1|Band), data=df2)
  m = anova(m)
  result[modelNo, ] <- list(modelNo, nrow(df2), paste(var[1], var[2], sep=":"), round(m[3, 6], 4), round(m[1, 5], 2), round(m[1, 3], 2), round(m[1, 4], 2))
  modelNo <- modelNo + 1
}



# UCI LMM - Phase - Variables ---------------------------------

# Default values
def_variables = c('EEG', 'Filter')
def_values = c('Raw', 'Butterworth')

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
    df2 = subset(df_phase, value <= 180)
  }
  else {
    df2 = subset(df_phase, value > 180)
  }
  for (var in def_variables) {
    df3 = df2
    j = 1
    for (var2 in def_variables) {
      if (var != var2) {
        df3 <- subset(df3, get(var2)==def_values[j])
      }
      j = j + 1
    }
    m = lmer(value ~ get(var)  + (1|sub) + (1|Band), data=df3)
    m = anova(m)
    result[modelNo, ] <- list(modelNo, phase_type, nrow(df3), var, round(m[1, 6], 4), round(m[1, 5], 2), round(m[1, 3], 2), round(m[1, 4], 2))
    modelNo <- modelNo + 1
  }
}

# UCI LMM - Phase - Interaction ---------------------------------

# Phase ~ (Factor1 * Factor2) + (1|sub) + (1|band)
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
combs <- combn(def_variables, 2)
for (phase_type in c('peak', 'trough')) {
  if (phase_type == 'peak') {
    df2 = subset(df_phase, value <= 180)
  }
  else {
    df2 = subset(df_phase, value > 180)
  }
  for (var in 1:ncol(combs)) {
    var = combs[,var]
    df3 = df2
    j = 1
    for (var2 in def_variables) {
      if (var[1] != var2 && var[2] != var2) {
        print(paste(var2, '=', def_values[j], 'for constant', var[1], 'and', var[2]))
        df3 <- subset(df3, get(var2)==def_values[j])
      }
      j = j + 1
    }
    m = lmer(value ~ (get(var[1]) * get(var[2]))  + (1|sub) + (1|Band), data=df3)
    m = anova(m)
    result[modelNo, ] <- list(modelNo, phase_type, nrow(df3), paste(var[1], var[2], sep=":"), round(m[3, 6], 4), round(m[1, 5], 2), round(m[1, 3], 2), round(m[1, 4], 2))
    modelNo <- modelNo + 1
  }
}



# Test --------------------------------------------------------------------

# var = 'Filter'
# sub = 'sub22'
# df2 = df_power
# j = 1
# for (var2 in def_variables) {
#   if (var != var2) {
#     df2 <- subset(df2, get(var2)==def_values[j])
#   }
#   j = j + 1
# }
unique(df_power$sub)
for (sub2 in unique(df_power$sub)) {
  df2 <- subset(df_power, sub=='sub21')
  df2 <- subset(df2, ArtifactRemoved==FALSE)
  df2 <- subset(df2, EEG=='Raw')
  df2 <- subset(df2, Resampled==FALSE)
  df2 <- subset(df2, Time==-750)
  df2 <- subset(df2, Method=='Welch')
  df2 <- subset(df2, Band=='Mu')
  labels <- seq(0, 150, by=10)
  g = ggplot(df2, aes(x=trial, y=value, group=Filter, color=Filter)) +
    geom_line() +
    scale_x_discrete(breaks=labels, labels=as.character(labels)) +
    ylab('Log power') +
    xlab('Trial') +
    ggtitle('Effect of filters on power trials') +
    theme_ms()
  print(g)
}


ggplot(df2, aes(x=Band, y=value, fill=Filter)) + 
  geom_boxplot() + 
  scale_color_grey() + 
  theme_ms() + 
  theme(strip.text.x = element_blank()) +
  ylab("Power (dB)")

