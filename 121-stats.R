# install.packages("tidyverse")
# install.packages("readxl")

# Load excel file.
library(readxl)

# Libraries for LMM.
library(lme4)
library(lmerTest)
library(devtools)
library(car) # for Anova

df = read_excel("118-phase-powers-v4.xlsx")
df$subexp <- paste(df$sub, df$exp, sep='.')
# Remove multiple sessions and keep only one session for each subject.
removearr <- c(
               'sub03.exp01', 
               'sub03.exp02',
               # 'sub03.exp03', 
               # 'sub04.exp01', 
               # 'sub05.exp01', 
               'sub06.exp01', 
               # 'sub06.exp02', 
               # 'sub07.exp01', 
               # 'sub08.exp01', 
               'sub08.exp02', 
               # 'sub10.exp01', 
               'sub10.exp02', 
               'sub11.exp01',
               'sub11.exp02'
               # 'sub12.exp02', 
               # 'sub15.exp01', 
               # 'sub16.exp01',
               # 'sub18.exp01'
               )
df <- subset(df, !(subexp %in% removearr))

# Convert to nominal.
df$sub = factor(df$sub)
df$exp = factor(df$exp)
df$trial_num = factor(df$trial_num)
df$mep_by_cmap_log <- log10(df$mep_by_cmap)
df$mep_size_log <- log10(df$mep_size)
df$LTM1_beta_power_abs <- abs(df$LTM1_beta_power)
df$LTM1_mu_power_abs <- abs(df$LTM1_mu_power)
df$C3_mu_power_abs <- abs(df$C3_mu_power)
df$C3_beta_power_abs <- abs(df$C3_beta_power)
df$C4_mu_power_abs <- abs(df$C4_mu_power)
df$C4_beta_power_abs <- abs(df$C4_beta_power)
df$subexp = factor(df$subexp)

df$C3_mu_phase_bin <- cut(df$C3_mu_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
df$C4_mu_phase_bin <- cut(df$C4_mu_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
df$LTM1_mu_phase_bin <- cut(df$LTM1_mu_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
df$C3_beta_phase_bin <- cut(df$C3_beta_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
df$C4_beta_phase_bin <- cut(df$C4_beta_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
df$LTM1_beta_phase_bin <- cut(df$LTM1_beta_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))

df$C3_mu_phase_bin = factor(df$C3_mu_phase_bin)
df$C4_mu_phase_bin = factor(df$C4_mu_phase_bin)
df$LTM1_mu_phase_bin = factor(df$LTM1_mu_phase_bin)
df$C3_beta_phase_bin = factor(df$C3_beta_phase_bin)
df$C4_beta_phase_bin = factor(df$C4_beta_phase_bin)
df$LTM1_beta_phase_bin = factor(df$LTM1_beta_phase_bin)

print_mm <- function(channel, band, response, removeOutlier=TRUE) {
  power = paste(c(channel, band, 'power'), collapse='_')
  phase = paste(c(channel, band, 'phase'), collapse='_')
  phase_raw = paste(c(channel, band, 'phase'), collapse='_')
  df2 = df
  
  if (removeOutlier) {
    df2 <- subset(df2, (get(phase_raw) > 45 & get(phase_raw) < 135) | (get(phase_raw) > 225 & get(phase_raw) < 315))
  }
  
  # formula <- paste(response, ' ~ (', power, ' * ', phase, ')', nestedEffect,' + (1|sub)')
  formula <- paste(response, ' ~ (', power, ' * ', phase, ') + (1|sub)', sep='')
  print(formula)
  m = lmer(as.formula(formula), data=df2)
  # Anova(m, type=3, test.statistic="F")
  # print('----------------------')
  summary(m)
}

plot_interaction <- function(channel, band, response, removeOutlier=True) {
  power = paste(c(channel, band, 'power'), collapse='_')
  phase = paste(c(channel, band, 'phase'), collapse='_')
  df2 = df
  if (removeOutlier) {
    df2 <- subset(df2, get(phase) > 45 & get(phase) < 135)
  } else {
    df2 <- subset(df2, get(phase) < 180)
  }
  
  df3 = df
  if (removeOutlier) {
    df3 <- subset(df3, get(phase) > 225 & get(phase) < 315)
  } else {
    df3 <- subset(df3, get(phase) >= 180)
  }
  
  plot(df2[[power]], df2[[response]], pch=16, col='blue', xlab=power, ylab=response, ylim=range(df[[response]]), xlim=range(df[[power]]))
  abline(lm(df2[[response]] ~ df2[[power]]), col='blue', xlab='', ylab='', ylim=range(df[[response]]), xlim=range(df[[power]]))
  par(new=TRUE)
  plot(df3[[power]], df3[[response]], pch=16, col='red', xlab='', ylab='', ylim=range(df[[response]]), xlim=range(df[[power]]))
  abline(lm(df3[[response]] ~ df3[[power]]), col='red', xlab='', ylab='', ylim=range(df[[response]]), xlim=range(df[[power]]))
}

plot_and_mm <- function(channel, band, response, removeOutlier=TRUE) {
  plot_interaction(channel, band, response, removeOutlier)
  print_mm(channel, band, response, removeOutlier)
}

print('----------------------')
plot_and_mm('C3', 'beta', 'mep_by_cmap_log', removeOutlier=FALSE)

plot_bars <- function(df, column, title) {
  df = df[order(df$sub),]
  par(mar=c(8,5.5,4,4)+.1)
  boxplot(get(column) ~ sub, data=df, las=2, ylab=column)
  title(title)
}

plot_bars(df, 'C3_mu_power', 'Power vs session')
plot_bars(df, 'mep_by_cmap_log', 'MEP/CMAP vs session')
df2 = subset(df, C3_mu_phase > 45 & C3_mu_phase < 135)
plot_bars(df2, 'C3_mu_phase', 'Peaks phases vs session')

df2 = subset(df, C3_mu_phase > 225 & C3_mu_phase < 315)
plot_bars(df2, 'C3_mu_phase', 'Trough phases vs session')
