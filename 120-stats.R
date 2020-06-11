# install.packages("tidyverse")
# install.packages("readxl")

# Load excel file.
library(readxl)
df = read_excel("118-phase-powers-v4.xlsx")

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

# Libraries for LMM.
library(lme4)
library(lmerTest)
library(devtools)
library(car) # for Anova


print_mm <- function(channel, band, response, removeOutlier=TRUE, includeNestedEffect=FALSE) {
  power = paste(c(channel, band, 'power'), collapse='_')
  phase = paste(c(channel, band, 'phase'), collapse='_')
  df2 = df
  if (removeOutlier) {
    df2 <- subset(df2, (get(phase) > 45 & get(phase) < 135) | (get(phase) > 225 & get(phase) < 315))
  }
  nestedEffect = ''
  if (includeNestedEffect) {
    nestedEffect <- '/(1|exp)'
  }
  # formula <- paste(response, ' ~ (', power, ' * ', phase, ')', nestedEffect,' + (1|sub)')
  formula <- paste(response, ' ~ (', power, ' * ', phase, ')', nestedEffect,' + (1|sub)', sep='')
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

plot_and_mm <- function(channel, band, response, removeOutlier=TRUE, includeNestedEffect=FALSE) {
  plot_interaction(channel, band, response, removeOutlier)
  print_mm(channel, band, response, removeOutlier, includeNestedEffect)
}

hist(df$C3_mu_power)

print('----------------------')
plot_and_mm('LTM1', 'mu', 'mep_size_log', removeOutlier=TRUE, includeNestedEffect=TRUE)

plot_bars <- function(df, column, title) {
  df$subexp <- paste(df$sub, df$exp, sep='.')
  df = df[order(df$subexp),]
  par(mar=c(8,5.5,4,4)+.1)
  boxplot(get(column) ~ subexp, data=df, las=2, ylab=column)
  title(title)
}
plot_bars(df, 'C3_mu_power', 'Power vs session')
plot_bars(df, 'mep_by_cmap_log', 'MEP/CMAP vs session')
df2 = subset(df, C3_mu_phase > 45 & C3_mu_phase < 135)
plot_bars(df2, 'C3_mu_phase', 'Peaks phases vs session')

df2 = subset(df, C3_mu_phase > 225 & C3_mu_phase < 315)
plot_bars(df2, 'C3_mu_phase', 'Trough phases vs session')
