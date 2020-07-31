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


# Prepare power and phase dataset ---------------------------------------------------------

# Load powers
df_power = read_excel("withmep/153-d1-power-long.xlsx")
df_power$Resampled = TRUE
df_power$ArtifactRemoved = TRUE
df = read_excel("withmep/158-d1-power-long.xlsx")
df$Resampled = FALSE
df$ArtifactRemoved = TRUE
df_power = bind_rows(df_power, df)
df = read_excel("withmep/159-d1-power-long.xlsx")
df$Resampled = TRUE
df$ArtifactRemoved = FALSE
df_power = bind_rows(df_power, df)
df = read_excel("withmep/161-d1-power-long.xlsx")
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
df_phase = read_excel("153-d1-phase-long.xlsx")
df_phase$ArtifactRemoved = TRUE
df = read_excel("159-d1-phase-long.xlsx")
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
# contrasts(df_phase$sub) <- "contr.sum"
contrasts(df_phase$trial) <- "contr.sum"
contrasts(df_phase$Band) <- "contr.sum"
contrasts(df_phase$Filter) <- "contr.sum"
contrasts(df_phase$EEG) <- "contr.sum"
contrasts(df_phase$ArtifactRemoved) <- "contr.sum"


# Combine powers and phases -----------------------------------------------

df_power = subset(df_power, Resampled==FALSE)
df_power = subset(df_power, ArtifactRemoved==TRUE)
df_power = subset(df_power, Method=='Welch')
df_power = subset(df_power, Filter=='Butterworth')
df_power = subset(df_power, EEG=='Average')
df_power = subset(df_power, Time=='-750')

df_power$phase <- 0
pb = txtProgressBar(min = 0, max = nrow(df_power), initial = 0, style=3) 
for (i in 1:nrow(df_power)) {
  df_test = subset(df_phase, ArtifactRemoved==df_power$ArtifactRemoved[i])
  df_test = subset(df_test, Filter==df_power$Filter[i])
  df_test = subset(df_test, EEG==df_power$EEG[i])
  df_test = subset(df_test, Band==df_power$Band[i])
  df_test = subset(df_test, sub==df_power$sub[i])
  df_test = subset(df_test, trial_abs==df_power$trial_abs[i])
  df_power$phase[i] <- df_test$value
  if (nrow(df_test) != 1) {
    print(paste('Error in ', i))
    break
  }
  setTxtProgressBar(pb, i)
}
close(pb)

# LMM: MEP ~ Power * Phase  ---------------------------------

RESULT <- data.frame(target=rep("", 5), band=rep("", 5), input=rep("", 5), p=rep(0, 5), b=rep(0, 5), no_of_obs=rep(0, 5), stringsAsFactors=FALSE)
i <- 1
for (variable in c("mep_size_log", "mep_latency_log", "mep_duration_log", 'mep_area_log')) {
  for (band in c('Theta', 'Mu', 'Beta', 'Gamma')) {
    
    df_power2 = subset(df_power, Band==band)
    df_power2 <- subset(df_power2, (phase > 45 & phase < 135) | (phase > 225 & phase < 315))
    df_power2$phase_bin <- factor(cut(df_power2$phase, breaks=c(0,180,360), labels=c(1, 0)))
    
    formula <- paste(variable, ' ~ (value * phase_bin) + (1|sub)', sep='')
    m = lmer(as.formula(formula), data=df_power2)
    m = summary(m)
    
    RESULT[i, ] <- list(variable, band, 'power', round(m$coefficients[2, 5], 4), round(m$coefficients[2, 1], 4), nrow(df_power2))
    RESULT[i+1, ] <- list(variable, band, 'phase', round(m$coefficients[3, 5], 4), round(m$coefficients[3, 1], 4), nrow(df_power2))
    RESULT[i+2, ] <- list(variable, band, 'interaction', round(m$coefficients[4, 5], 4), round(m$coefficients[4, 1], 4), nrow(df_power2))
    i=i+3
  }
}


