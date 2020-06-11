# Libraries ----
# install.packages("tidyverse")
# install.packages("readxl")

# Load excel file.
library(readxl)
library(BBmisc)
# Libraries for LMM.
library(lme4)
library(lmerTest)
library(devtools)
library(car) # for Anova
library(fitdistrplus)
require("lattice")
library(nortest)
library(vcd)
require(car)
require(MASS)
require(broom)

# Load file ----
# df = read_excel("136-clean-v1.xlsx")
# df = read_excel("138-all-v1.xlsx")
df = read_excel("140-ld-v3.xlsx")
df$subexp <- paste(df$sub, df$exp, sep='.')

# Transform some columns -----
# Convert to nominal.
df$sub = factor(df$sub)
df$exp = factor(df$exp)
df$trial_num = factor(df$trial_num)
# df$mep_by_cmap_log <- log10(df$mep_by_cmap)
# df$mep_size_log <- log10(df$mep_size)
df$subexp = factor(df$subexp)


df$mu_phase_bin <- cut(df$mu_phase, breaks=c(0,180,360), labels=c(1, 0))
df$mu_phase_bin <- cut(df$mu_phase, breaks=c(0,180,360), labels=c(1, 0))
# df$LTM1_mu_phase_bin <- cut(df$LTM1_mu_phase, breaks=c(0,180,360), labels=c(1, 0))
df$beta_phase_bin <- cut(df$beta_phase, breaks=c(0,180,360), labels=c(1, 0))
df$beta_phase_bin <- cut(df$beta_phase, breaks=c(0,180,360), labels=c(1, 0))
# df$LTM1_beta_phase_bin <- cut(df$LTM1_beta_phase, breaks=c(0,180,360), labels=c(1, 0))
df$gamma_phase_bin <- cut(df$gamma_phase, breaks=c(0,180,360), labels=c(1, 0))
df$gamma_phase_bin <- cut(df$gamma_phase, breaks=c(0,180,360), labels=c(1, 0))
# df$LTM1_gamma_phase_bin <- cut(df$LTM1_gamma_phase, breaks=c(0,180,360), labels=c(1, 0))
df$gamma_phase_bin <- cut(df$gamma_phase, breaks=c(0,180,360), labels=c(1, 0))
df$gamma_phase_bin <- cut(df$gamma_phase, breaks=c(0,180,360), labels=c(1, 0))
# df$LTM1_gamma_phase_bin <- cut(df$LTM1_gamma_phase, breaks=c(0,180,360), labels=c(1, 0))
df$high_gamma_phase_bin <- cut(df$high_gamma_phase, breaks=c(0,180,360), labels=c(1, 0))
df$high_gamma_phase_bin <- cut(df$high_gamma_phase, breaks=c(0,180,360), labels=c(1, 0))
# df$LTM1_high_gamma_phase_bin <- cut(df$LTM1_high_gamma_phase, breaks=c(0,180,360), labels=c(1, 0))
df$low_gamma_phase_bin <- cut(df$low_gamma_phase, breaks=c(0,180,360), labels=c(1, 0))
df$low_gamma_phase_bin <- cut(df$low_gamma_phase, breaks=c(0,180,360), labels=c(1, 0))
# df$LTM1_low_gamma_phase_bin <- cut(df$LTM1_low_gamma_phase, breaks=c(0,180,360), labels=c(1, 0))
df$theta_phase_bin <- cut(df$theta_phase, breaks=c(0,180,360), labels=c(1, 0))
df$theta_phase_bin <- cut(df$theta_phase, breaks=c(0,180,360), labels=c(1, 0))
# df$LTM1_theta_phase_bin <- cut(df$LTM1_theta_phase, breaks=c(0,180,360), labels=c(1, 0))



df$mu_phase_bin <- factor(df$mu_phase_bin)
df$mu_phase_bin <- factor(df$mu_phase_bin)
# df$LTM1_mu_phase_bin = factor(df$LTM1_mu_phase_bin)
df$beta_phase_bin <- factor(df$beta_phase_bin)
df$beta_phase_bin = factor(df$beta_phase_bin)
# df$LTM1_beta_phase_bin = factor(df$LTM1_beta_phase_bin)
df$gamma_phase_bin <- factor(df$gamma_phase_bin)
df$gamma_phase_bin <- factor(df$gamma_phase_bin)
# df$LTM1_gamma_phase_bin = factor(df$LTM1_gamma_phase_bin)
df$high_gamma_phase_bin <- factor(df$high_gamma_phase_bin)
df$high_gamma_phase_bin <- factor(df$high_gamma_phase_bin)
# df$LTM1_high_gamma_phase_bin = factor(df$LTM1_high_gamma_phase_bin)
df$low_gamma_phase_bin <- factor(df$low_gamma_phase_bin)
df$low_gramma_phase_bin <- factor(df$low_gamma_phase_bin)
# df$LTM1_low_gamma_phase_bin = factor(df$LTM1_low_gamma_phase_bin)
df$theta_phase_bin <- factor(df$theta_phase_bin)
df$theta_phase_bin <- factor(df$theta_phase_bin)
# df$LTM1_theta_phase_bin = factor(df$LTM1_theta_phase_bin)

df$mu_power <- log(df$mu_power)
df$beta_power <- log(df$beta_power)
df$gamma_power <- log(df$gamma_power)
df$low_gamma_power <- log(df$low_gamma_power)
df$high_gamma_power <- log(df$high_gamma_power)
df$theta_power <- log(df$theta_power)

# df$LTM1_mu_power = log(df$LTM1_mu_power)+1
# df$LTM1_beta_power = log(df$LTM1_beta_power)+1
# df$LTM1_theta_power = log(df$LTM1_theta_power)+1

# df$LTM1_beta_power_abs <- abs(df$LTM1_beta_power)
# df$LTM1_mu_power_abs <- abs(df$LTM1_mu_power)
# df$LTM1_theta_power_abs <- abs(df$LTM1_theta_power)
df$theta_power_abs <- abs(df$theta_power)
df$mu_power_abs <- abs(df$mu_power)
df$beta_power_abs <- abs(df$beta_power)
df$low_gamma_power_abs <- abs(df$low_gamma_power)
df$high_gamma_power_abs <- abs(df$high_gamma_power)
df$gamma_power_abs <- abs(df$gamma_power)

# df$mep_by_cmap_log <- log(df$mep_by_cmap)
df$mep_size <- abs(df$mep_size)
df$mep_size_log <- log(df$mep_size)
df$mep_latency_log <- log(df$mep_latency)
df$mep_duration_log <- log(df$mep_duration)
df$mep_area_log <- log(df$mep_area)

print_mm <- function(band, response, removeOutlier=TRUE, abs='') {
  power = paste(c(band, 'power'), collapse='_')
  power = paste(c(power, abs), collapse='')
  phase = paste(c(band, 'phase'), collapse='_')
  phase_raw = paste(c(band, 'phase'), collapse='_')
  df2 = df
  if (!(abs=='')) {
    df2[[power]] <- abs(df2[[power]])
  }
  if (removeOutlier) {
    df2 <- subset(df2, (get(phase_raw) > 45 & get(phase_raw) < 135) | (get(phase_raw) > 225 & get(phase_raw) < 315))
  }
  formula <- paste(response, ' ~ (', power, ' * ', phase, ') + (1|sub)', sep='')
  print(formula)
  m = lmer(as.formula(formula), data=df2)
  # summary(m)
  return(list(m, dim(df2)[1]))
}

plot_interaction <- function(band, response, removeOutlier=True, abs='') {
  power = paste(c(band, 'power'), collapse='_')
  power = paste(c(power, abs), collapse='')
  phase = paste(c(band, 'phase'), collapse='_')
  df2 = df
  df2[mapply(is.infinite, df2)] <- 0
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

plot_and_mm <- function(band, response, abs=FALSE, removeOutlier=FALSE) {
  abs <- if(abs == TRUE) '_abs' else ''
  plot_interaction(band, response, removeOutlier, abs=abs)
  print_mm(band, response, removeOutlier, abs=abs)
}

# hist(log(df$mep_size))
# hist(log(df$mep_size/mean(df$mep_size)))

# df$mep_size = df$mep_size / mean(df$mep_size)
# df$mep_size_log = log(df$mep_size)
# hist(df$mep_size/mean(df$mep_size))

hist(df$mep_size_log, breaks=40)
hist(df$mep_latency, breaks=40)
hist(df$mep_duration_log, breaks=40)
hist(df$mep_area_log, breaks=40)

y = 'mep_duration_log'
m = plot_and_mm('theta', y, abs=FALSE, removeOutlier=TRUE)
m = m[[1]]
summary(m)
qqp(resid(m), 'norm')
plot(residuals(m))
qqmath(m)
hist(resid(m))

get_estimations <- function(m) {
  return(list(summary(m)$coefficients[2,5], # p_power
  summary(m)$coefficients[2,1], # b_power
  summary(m)$coefficients[3,5], # p_phase
  summary(m)$coefficients[3,1], # b_phase
  summary(m)$coefficients[4,5], # p_inter
  summary(m)$coefficients[4,1])) # b_inter
}

DF <- data.frame(target=rep("", 5), band=rep("", 5), input=rep("", 5), p=rep(0, 5), b=rep(0, 5), no_of_obs=rep(0, 5), stringsAsFactors=FALSE)
i <- 1
for (variable in c("mep_size_log", "mep_latency", "mep_duration_log", "mep_area_log")) {
  for (band in c('theta', 'mu', 'beta', 'gamma', 'low_gamma', 'high_gamma')) {
    m = plot_and_mm(band, variable, abs=FALSE, removeOutlier=TRUE)
    no_of_obs = m[[2]]
    m = get_estimations(m[[1]])
    DF[i, ] <- list(variable, band, 'power', round(as.numeric(m[1]), 3), round(as.numeric(m[2]), 5), no_of_obs)
    i=i+1
    DF[i, ] <- list(variable, band, 'phase', round(as.numeric(m[3]), 3), round(as.numeric(m[4]), 5), no_of_obs)
    i=i+1
    DF[i, ] <- list(variable, band, 'interaction', round(as.numeric(m[5]), 3), round(as.numeric(m[6]), 5), no_of_obs)
    i=i+1
  }
}

# Linear Regression
model  <- lm(mep_size_log ~ theta_power + mu_power + beta_power + gamma_power, data = df)
summary(model)
confint(model)
