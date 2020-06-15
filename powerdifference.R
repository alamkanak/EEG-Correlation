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


# Prepare dataset ---------------------------------------------------------

# Load file 
df = read_excel("153-d1-power-long.xlsx")
df_phase = read_excel("153-d1-phase-long.xlsx")

# Fix column ordering
df <- transform(df, Band=factor(Band,levels=c("Theta","Mu","Beta","Gamma")))
df <- transform(df, Method=factor(Method,levels=c("FFT","Welch","Burg")))
df <- transform(df, EEG=factor(EEG,levels=c("Raw","Hjorth","Average")))
df_phase <- transform(df_phase, Band=factor(Band,levels=c("Theta","Mu","Beta","Gamma")))
df_phase <- transform(df_phase, EEG=factor(EEG,levels=c("Raw","Hjorth","Average")))

# Transform some columns
df$sub = factor(df$sub)
df$trial = factor(df$trial_abs)
df$Method = factor(df$Method)
df$Band = factor(df$Band)
df$Filter = factor(df$Filter)
df$Time = factor(df$Time)
df$EEG = factor(df$EEG)

df_phase$sub = factor(df_phase$sub)
df_phase$trial = factor(df_phase$trial_abs)
df_phase$Band = factor(df_phase$Band)
df_phase$Filter = factor(df_phase$Filter)
df_phase$EEG = factor(df_phase$EEG)

# For post-hoc pairwise comparison
contrasts(df$sub) <- "contr.sum"
contrasts(df$trial) <- "contr.sum"
contrasts(df$Method) <- "contr.sum"
contrasts(df$Band) <- "contr.sum"
contrasts(df$Filter) <- "contr.sum"
contrasts(df$Time) <- "contr.sum"
contrasts(df$EEG) <- "contr.sum"

contrasts(df_phase$sub) <- "contr.sum"
contrasts(df_phase$trial) <- "contr.sum"
contrasts(df_phase$Band) <- "contr.sum"
contrasts(df_phase$Filter) <- "contr.sum"
contrasts(df_phase$EEG) <- "contr.sum"

# Linear mixed models -----------------------------------------------------

# Power ~ Method * Filter
result <- data.frame(
  modelNo=rep("",3), 
  eeg=rep("", 3), 
  band=rep("", 3), 
  method=rep("", 3),
  filter=rep("", 3),
  time=rep("", 3),
  inputs=rep("", 3), 
  factor=rep("", 3), 
  response=rep("", 3),
  obsCount=rep("", 3),
  p=rep("", 3), 
  f=rep("", 3), 
  dfn=rep("", 3), 
  dfd=rep("", 3), 
  stringsAsFactors=FALSE)
eeg <- 'Raw'
time <- '-150'
df2 <- subset(df, EEG==eeg)
df2 <- subset(df2, Time==time)
i = 1
modelNo <- 1
for (band in c("Theta", "Mu", "Beta", "Gamma")) {
  df3 <- subset(df2, Band==band)
  m = lmer(value ~ (Method * Filter)  + (1|sub), data=df3)
  m = anova(m)
  result[i, ] <- list(modelNo, 'Hjorth', band, '*', '*', time, 'Method*Filter', rownames(m)[1], 'Power', nrow(df3), m[1, 6], m[1, 5], m[1, 3], m[1, 4])
  i = i+1
  result[i, ] <- list(modelNo, 'Hjorth', band, '*', '*', time, 'Method*Filter', rownames(m)[2], 'Power', nrow(df3), m[2, 6], m[2, 5], m[2, 3], m[2, 4])
  i = i+1
  result[i, ] <- list(modelNo, 'Hjorth', band, '*', '*', time, 'Method*Filter', rownames(m)[3], 'Power', nrow(df3), m[3, 6], m[3, 5], m[3, 3], m[3, 4])
  i = i+1
  modelNo <- modelNo + 1
}


# Power ~ EEG * Time
method <- 'Welch'
filter <- 'Butterworth'
df2 <- subset(df, Method==method)
df2 <- subset(df2, Filter==filter)
for (band in c("Theta", "Mu", "Beta", "Gamma")) {
  df3 <- subset(df2, Band==band)
  m = lmer(value ~ (EEG * Time)  + (1|sub), data=df3)
  m = anova(m)
  result[i, ] <- list(modelNo, '*', band, method, filter, '*', 'EEG*Time', rownames(m)[1], 'Power', nrow(df3), m[1, 6], m[1, 5], m[1, 3], m[1, 4])
  i = i+1
  result[i, ] <- list(modelNo, '*', band, method, filter, '*', 'EEG*Time', rownames(m)[2], 'Power', nrow(df3), m[2, 6], m[2, 5], m[2, 3], m[2, 4])
  i = i+1
  result[i, ] <- list(modelNo, '*', band, method, filter, '*', 'EEG*Time', rownames(m)[3], 'Power', nrow(df3), m[3, 6], m[3, 5], m[3, 3], m[3, 4])
  i = i+1
  modelNo <- modelNo + 1
}

# Power ~ Filter * EEG
time <- '-750'
df2 <- subset(df_phase, value < 180)
for (band in c("Theta", "Mu", "Beta", "Gamma")) {
  df3 <- subset(df2, Band==band)
  m = lmer(value ~ (Filter * EEG)  + (1|sub), data=df3)
  m = anova(m)
  result[i, ] <- list(modelNo, '*', band, 'NA', '*', time, 'EEG*Filter', rownames(m)[1], 'Phase-peak', nrow(df3), m[1, 6], m[1, 5], m[1, 3], m[1, 4])
  i = i+1
  result[i, ] <- list(modelNo, '*', band, 'NA', '*', time, 'EEG*Filter', rownames(m)[2], 'Phase-peak', nrow(df3), m[2, 6], m[2, 5], m[2, 3], m[2, 4])
  i = i+1
  result[i, ] <- list(modelNo, '*', band, 'NA', '*', time, 'EEG*Filter', rownames(m)[3], 'Phase-peak', nrow(df3), m[3, 6], m[3, 5], m[3, 3], m[3, 4])
  i = i+1
  modelNo <- modelNo + 1
}

# Power ~ Filter * EEG (-750ms)
time <- '-750'
df2 <- subset(df_phase, value >= 180)
for (band in c("Theta", "Mu", "Beta", "Gamma")) {
  df3 <- subset(df2, Band==band)
  m = lmer(value ~ (Filter * EEG)  + (1|sub), data=df3)
  m = anova(m)
  result[i, ] <- list(modelNo, '*', band, 'NA', '*', time, 'EEG*Filter', rownames(m)[1], 'Phase-trough', nrow(df3), m[1, 6], m[1, 5], m[1, 3], m[1, 4])
  i = i+1
  result[i, ] <- list(modelNo, '*', band, 'NA', '*', time, 'EEG*Filter', rownames(m)[2], 'Phase-trough', nrow(df3), m[2, 6], m[2, 5], m[2, 3], m[2, 4])
  i = i+1
  result[i, ] <- list(modelNo, '*', band, 'NA', '*', time, 'EEG*Filter', rownames(m)[3], 'Phase-trough', nrow(df3), m[3, 6], m[3, 5], m[3, 3], m[3, 4])
  i = i+1
  modelNo <- modelNo + 1
}

write.csv(result,"d1-result.csv", row.names = FALSE)



# Plot power - histograms -------------------------------------------------

# Power hist ~ Filter + Band
df2 = df
df2 <- subset(df2, EEG=='Hjorth')
df2 <- subset(df2, Method=='Welch')
df2 <- subset(df2, Time=='-150')
ggplot(df2, aes(x=value, fill=Filter)) +
  geom_histogram( color="#e9ecef", alpha=0.6, position = 'identity') +
  facet_grid(. ~ Band) +
  theme_ms() +
  xlab("Power (log)") +
  ylab("Count")

# Power hist ~ Method + Band
df2 = df
df2 <- subset(df2, EEG=='Hjorth')
df2 <- subset(df2, Filter=='Butterworth')
df2 <- subset(df2, Time=='-150')
ggplot(df2, aes(x=value, fill=Method)) +
  geom_histogram(color="#e9ecef", alpha=0.6, position = 'identity') +
  facet_grid(. ~ Band) +
  theme_ms() +
  xlab("Power (log)") +
  ylab("Count")

# Power hist ~ EEG + Band
df2 = df
df2 <- subset(df2, Time=='-150')
df2 <- subset(df2, Method=='Welch')
df2 <- subset(df2, Filter=='Butterworth')
ggplot(df2, aes(x=value, fill=EEG)) +
  geom_histogram( color="#e9ecef", alpha=0.6, position = 'identity') +
  facet_grid(. ~ Band) +
  theme_ms() +
  xlab("Power (log)") +
  ylab("Count")

# Power hist ~ Time + Band
df2 = df
df2 <- subset(df2, EEG=='Hjorth')
df2 <- subset(df2, Filter=='Butterworth')
df2 <- subset(df2, Method=='Welch')
ggplot(df2, aes(x=value, fill=Time)) +
  geom_histogram( color="#e9ecef", alpha=0.6, position = 'identity') +
  scale_fill_manual(values=c("#69b3a2", "#404080")) +
  facet_grid(. ~ Band) +
  theme_ms() +
  xlab("Power (log)") +
  ylab("Count")



# Plot power - trials -----------------------------------------------------

# Power trials ~ Filter

df2 <- subset(df, EEG=='Hjorth')
df2 <- subset(df2, Time=='-150')
df2 <- subset(df2, sub=='co2c0000340')
df2 <- subset(df2, Band=='Beta')
df2 <- subset(df2, Method=='Welch')
labels <- seq(0, 150, by=10)
ggplot(df2, aes(x=trial, y=value, group=Filter, color=Filter)) + 
  geom_line() + 
  scale_x_discrete(breaks=labels, labels=as.character(labels)) +
  ylab('Log power') +
  xlab('Trial') +
  ggtitle('Effect of PSD estimation method on the beta power of Hjorth transformed C3 signal') +
  theme_ms()

# Power trials ~ Time
df2 <- subset(df, EEG=='Hjorth')
df2 <- subset(df2, Filter=='Butterworth')
df2 <- subset(df2, sub=='co2c0000340')
df2 <- subset(df2, Band=='Beta')
df2 <- subset(df2, Method=='Welch')
labels <- seq(0, 150, by=10)
ggplot(df2, aes(x=trial, y=value, group=Time, color=Time)) + 
  geom_line() + 
  scale_x_discrete(breaks=labels, labels=as.character(labels)) +theme(legend.title = element_blank()) +
  theme(legend.position = "none") +
  ylab('Log power') +
  xlab('Trial') +
  ggtitle('Effect of PSD estimation method on the beta power of Hjorth transformed C3 signal') +
  theme_ms()


# Power trials ~ PSD estimation method
df2 <- subset(df, EEG=='Raw')
df2 <- subset(df2, Filter=='Butterworth')
df2 <- subset(df2, sub=='sub22')
df2 <- subset(df2, Band=='Beta')
df2 <- subset(df2, Time=='-750')
labels <- seq(0, 150, by=10)
# tiff("test.tiff", units="in", width=10, height=2.5, res=300)
g1 <- ggplot(df2, aes(x=trial, y=value, group=Method, color=Method)) + 
  geom_line() + 
  scale_color_manual(values=c("#f44336", "#2196f3", '#795548')) +
  scale_x_discrete(breaks=labels, labels=as.character(labels)) +
  ylab('Log power') +
  xlab('Trial') +
  # guides(color = FALSE) +
  theme_ms()
# dev.off()

# Power trials ~ Filters
df2 <- subset(df, EEG=='Raw')
df2 <- subset(df2, Method=='Welch')
df2 <- subset(df2, sub=='sub22')
df2 <- subset(df2, Band=='Beta')
df2 <- subset(df2, Time=='-750')
labels <- seq(0, 150, by=10)
# tiff("test.tiff", units="in", width=10, height=2.5, res=300)
g2 <- ggplot(df2, aes(x=trial, y=value, group=Filter, color=Filter)) + 
  geom_line() + 
  scale_color_manual(values=c("#f44336", "#2196f3")) +
  scale_x_discrete(breaks=labels, labels=as.character(labels)) +
  ylab('Log power') +
  xlab('Trial') +
  # guides(color = FALSE) +
  theme_ms()
# dev.off()

# Power trials ~ EEG
df2 <- subset(df, Filter=='Butterworth')
df2 <- subset(df2, Method=='Welch')
df2 <- subset(df2, sub=='sub22')
df2 <- subset(df2, Band=='Beta')
df2 <- subset(df2, Time=='-750')
labels <- seq(0, 150, by=10)
# tiff("test.tiff", units="in", width=10, height=2.5, res=300)
g3 <- ggplot(df2, aes(x=trial, y=value, group=EEG, color=EEG)) + 
  geom_line() + 
  scale_color_manual(values=c("#f44336", "#2196f3", '#795548')) +
  scale_x_discrete(breaks=labels, labels=as.character(labels)) +
  ylab('Log power') +
  xlab('Trial') +
  # guides(color = FALSE) +
  # scale_y_continuous(limits = c(-8, 3)) +
  theme_ms()
# dev.off()

x=subset(df2, EEG='Raw')$value
plot(x, type="l")


# Power trials ~ Time
df2 <- subset(df, Filter=='Butterworth')
df2 <- subset(df2, Method=='Welch')
df2 <- subset(df2, sub=='sub22')
df2 <- subset(df2, Band=='Beta')
df2 <- subset(df2, EEG=='Raw')
labels <- seq(0, 150, by=10)
# tiff("test.tiff", units="in", width=10, height=2.5, res=300)
g4 <- ggplot(df2, aes(x=trial, y=value, group=Time, color=Time)) + 
  geom_line() + 
  scale_color_manual(values=c("#f44336", "#2196f3")) +
  scale_x_discrete(breaks=labels, labels=as.character(labels)) +
  ylab('Log power') +
  xlab('Trial') +
  # guides(color = FALSE) +
  theme_ms()
# dev.off()

tiff("test.tiff", units="in", width=10, height=3.8, res=300)
ggarrange(g1, g2, g3, g4, ncol = 2, nrow = 2)
dev.off()


df2 <- subset(df, Filter=='Butterworth')
df2 <- subset(df2, Method=='Welch')
df2 <- subset(df2, sub=='sub21')
df2 <- subset(df2, Band=='Beta')
df2 <- subset(df2, Time=='-750')
plot(df2)


# Plot phase - histograms --------------------------------------------------------

# Phase hist ~ Filter
df2 = df_phase
df2 <- subset(df2, EEG=='Raw')
tiff("test.tiff", units="in", width=10, height=2.5, res=300)
ggplot(df2, aes(x=value, fill=Filter)) +
  geom_histogram( color="#e9ecef", alpha=0.6, position = 'identity') +
  scale_fill_manual(values=c("#69b3a2", "#404080")) +
  facet_grid(. ~ Band) +
  theme_ms() +
  # theme(strip.text.x = element_blank()) +
  xlab("Phase") +
  ylab("Count")
dev.off()

# Phase hist ~ EEG type
df2 = df_phase
df2 <- subset(df2, Filter=='Butterworth')
# tiff("test.tiff", units="in", width=10, height=3, res=300)
ggplot(df2, aes(x=value, fill=EEG)) +
  geom_histogram( color="#e9ecef", alpha=0.6, position = 'identity') +
  # scale_fill_manual(values=c("#69b3a2", "#404080")) +
  facet_grid(. ~ Band) +
  theme_ms() +
  xlab("Phase - Hjorth") +
  ylab("Count")
# dev.off()


# Plot power - boxplot ----------------------------------------------------

# Power ~ Method (Filter)
df2 <- subset(df, EEG=='Raw')
df2 <- subset(df2, Time=='-750')
df2 <- subset(df2, sub=='sub22')
tiff("test.tiff", units="in", width=10, height=3, res=300)
ggplot(df2, aes(x=Band, y=value, fill=Filter)) + 
  geom_boxplot() + 
  facet_grid(. ~ Method) + 
  scale_color_grey() + 
  theme_ms() + 
  # theme(strip.text.x = element_blank()) +
  ylab("Log power")
dev.off()


# Power ~ Time window (EEG)
df2 <- subset(df, Method=='Welch')
df2 <- subset(df2, Filter=='Butterworth')
df2 <- subset(df2, sub=='sub22')
# df2 <- subset(df2, Method=='Welch')
# df2 <- subset(df2, Band=='Beta')
tiff("test.tiff", units="in", width=10, height=3, res=300)
ggplot(df2, aes(x=Band, y=value, fill=Time)) + 
  geom_boxplot() + 
  facet_grid(. ~ EEG) + 
  scale_color_grey() + 
  theme_ms() + 
  # theme(strip.text.x = element_blank()) +
  ylab("Log power")
dev.off()

# Power ~ Time window (Filter)
df2 <- subset(df, EEG=='Hjorth')
df2 <- subset(df2, Method=='Welch')
df2 <- subset(df2, sub=='co2c0000378')
# df2 <- subset(df2, Method=='Welch')
# df2 <- subset(df2, Band=='Beta')
ggplot(df2, aes(x=Band, y=value, fill=Time)) + 
  geom_boxplot() + 
  facet_grid(. ~ Filter) + 
  scale_color_grey() + 
  theme_ms() + 
  ylab("Log power")


# Plot phase - boxplot ----------------------------------------------------

# Phase ~ Filter + EEG type
df2 <- df_phase
# df2 <- subset(df2, EEG=='Hjorth')
# df2 <- subset(df2, sub=='sub10')
# df2 <- subset(df2, Method=='Welch')
# df2 <- subset(df2, Band=='Beta')
ggplot(df2, aes(x=Band, y=value, fill=Filter)) + 
  geom_boxplot() + 
  facet_grid(. ~ EEG) + 
  scale_color_grey() + 
  theme_ms() + 
  ylab("Log power")

# Plot interaction --------------------------------------------------------

# Power ~ time + eeg type
df2 <- df
df2 <- subset(df2, Method=='Welch')
df2 <- subset(df2, Filter=='Butterworth')
tiff("test.tiff", units="in", width=10, height=3, res=300)
ggplot(df2) +
  aes(x = EEG, color = Time, group = Time, y = value) +
  stat_summary(fun.y = mean, geom = "point") +
  stat_summary(fun.y = mean, geom = "line") +
  facet_grid(. ~ Band) + 
  theme_ms()
dev.off()


# Power ~ method + filter + band
df2 <- df
df2 <- subset(df2, EEG=='Raw')
df2 <- subset(df2, Time=='-750')
# tiff("test.tiff", units="in", width=10, height=3, res=300)
ggplot(df2) +
  aes(x = Method, color = Filter, group = Filter, y = value) +
  stat_summary(fun.y = mean, geom = "point") +
  stat_summary(fun.y = mean, geom = "line") +
  facet_grid(. ~ Band) + 
  theme_ms()
# dev.off()

# Power ~ method + filter + time
df2 <- df
df2 <- subset(df2, EEG=='Hjorth')
df2 <- subset(df2, Band=='Beta')
ggplot(df2) +
  aes(x = Method, color = Filter, group = Filter, y = value) +
  stat_summary(fun.y = mean, geom = "point") +
  stat_summary(fun.y = mean, geom = "line") +
  facet_grid(. ~ Time) + 
  theme_ms()
