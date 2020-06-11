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


# Load file ----
# df = read_excel("136-clean-v1.xlsx")
# df = read_excel("138-all-v1.xlsx")
df = read_excel("137-all-v1-temp.xlsx")
# df = subset(df, df$rejected == FA137-all-v2-pburg.xlsxLSE)
df$subexp <- paste(df$sub, df$exp, sep='.')

# Remove multiple sessions and keep only one session for each subject. ----
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
# df <- subset(df, !(subexp %in% removearr))


# Transform some columns -----
# Convert to nominal.
df$sub = factor(df$sub)
df$exp = factor(df$exp)
df$trial_num = factor(df$trial_num)
# df$mep_by_cmap_log <- log10(df$mep_by_cmap)
# df$mep_size_log <- log10(df$mep_size)
df$subexp = factor(df$subexp)

df$mu_phase_bin <- cut(df$mu_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
df$mu_phase_bin <- cut(df$mu_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
# df$LTM1_mu_phase_bin <- cut(df$LTM1_mu_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
df$beta_phase_bin <- cut(df$beta_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
df$beta_phase_bin <- cut(df$beta_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
# df$LTM1_beta_phase_bin <- cut(df$LTM1_beta_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
df$gamma_phase_bin <- cut(df$gamma_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
df$gamma_phase_bin <- cut(df$gamma_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
# df$LTM1_gamma_phase_bin <- cut(df$LTM1_gamma_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
df$gamma_phase_bin <- cut(df$gamma_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
df$gamma_phase_bin <- cut(df$gamma_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
# df$LTM1_gamma_phase_bin <- cut(df$LTM1_gamma_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
df$high_gamma_phase_bin <- cut(df$high_gamma_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
df$high_gamma_phase_bin <- cut(df$high_gamma_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
# df$LTM1_high_gamma_phase_bin <- cut(df$LTM1_high_gamma_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
df$low_gamma_phase_bin <- cut(df$low_gamma_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
df$low_gamma_phase_bin <- cut(df$low_gamma_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
# df$LTM1_low_gamma_phase_bin <- cut(df$LTM1_low_gamma_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
df$theta_phase_bin <- cut(df$theta_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
df$theta_phase_bin <- cut(df$theta_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))
# df$LTM1_theta_phase_bin <- cut(df$LTM1_theta_phase, breaks=c(0,180,360), labels=c('peak', 'trough'))

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
df$mu_power_abs <- abs(df$mu_power)
df$beta_power_abs <- abs(df$beta_power)
df$gamma_power_abs <- abs(df$gamma_power)
df$theta_power_abs <- abs(df$theta_power)
df$mu_power_abs <- abs(df$mu_power)
df$beta_power_abs <- abs(df$beta_power)
df$gamma_power_abs <- abs(df$gamma_power)
df$theta_power_abs <- abs(df$theta_power)

df$mep_by_cmap_log <- log(df$mep_by_cmap)
df$mep_size <- abs(df$mep_size)
df$mep_size_log <- log(df$mep_size)
df$mep_latency_log <- log(df$mep_latency)
df$mep_duration_log <- log(df$mep_duration)

print_mm <- function(band, response, removeOutlier=TRUE, abs='') {
  power = paste(c(band, 'power'), collapse='_')
  power = paste(c(power, abs), collapse='')
  phase = paste(c(band, 'phase_bin'), collapse='_')
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
  summary(m)
  return(m)
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


df$mep_size = df$mep_size / mean(df$mep_size)
df$mep_size_log = log(df$mep_size)

# df$mep_latency2 = scale(df$mep_latency, center=TRUE, scale=TRUE)
# df$mep_latency2 = df$mep_latency2 / mean(df$mep_latency2)
# 
# df$b2m = df$mu_power / df$beta_power
# df$b2m = log(abs(df$b2m/mean(df$b2m)))
# df$b2m_power = df$b2m
# df$b2m_power_abs = abs(df$b2m_power)
# 
# df$b2g_power = df$beta_power / df$gamma_power
# df$b2g_power = log(abs(df$b2g_power/mean(df$b2g_power)))
# 
# df$m2g_power = df$mu_power / df$gamma_power
# df$m2g_power = log(abs(df$m2g_power/mean(df$m2g_power)))
# df$m2g_power_abs = abs(df$m2g_power)

hist(df$mep_size_log, breaks=40)
hist(df$mep_latency, breaks=40)
hist(df$mep_duration, breaks=40)

y = 'mep_latency'
m = plot_and_mm('mu', y, abs=FALSE, removeOutlier=TRUE)
summary(m)
qqp(resid(m), 'norm')
plot(residuals(m))
qqmath(m)

# --------------------------
# GLMM 
# --------------------------

contrasts(df$sub) <- "contr.sum"
contrasts(df$theta_phase_bin) <- "contr.sum"

df$mep_duration_dis = as.integer(normalize(df$mep_duration, method='range', range=c(1, 100)))
df$mep_latency_dis = as.integer(normalize(df$mep_latency, method='range', range=c(1, 100)))
df$mep_size_log_dis = as.integer(normalize(df$mep_size_log, method='range', range=c(1, 100)))
m = glmer(mep_duration_dis ~ (beta_power * beta_phase_bin) + (1|sub), data=df, family='poisson', nAGQ=0)
summary(m)
plot(df$beta_power, df$mep_duration_dis)

poisson <- fitdistr(resid(m), "Poisson")
qqp(residuals(m), 'pois', lambda=poisson$estimate)

qqnorm(residuals(m)); qqline(residuals(m))


# -----------
hist(df$mep_latency_dis, breaks=40)
m = glmer(mep_latency_dis ~ (gamma_power * gamma_phase_bin) + (1|sub), data=df, family='Gamma', nAGQ=0)
summary(m)
plot(df$gamma_power, df$mep_latency_dis)

poisson <- fitdistr(resid(m), "Poisson")
qqp(resid(m), 'pois', lambda=poisson$estimate)

# --------------------
hist(df$mep_size_log, breaks=40)
m = glmer(mep_size_log_dis ~ (beta_power * beta_phase_bin) + (1|sub), data=df, family='Gamma', nAGQ=0)
summary(m)
plot(df$beta_power, df$mep_size_log_dis)

poisson <- fitdistr(resid(m), "Poisson")
qqp(resid(m), 'pois', lambda=poisson$estimate)
# ---------------------------------

shapiro.test(resid(m))
hist(df$mep_duration, breaks=40)

df$mep_size_dis <- as.integer(normalize(df$mep_size, method='range', range=c(1, 100)))
df$mep_size_log_dis <- as.integer(normalize(df$mep_size_log, method='range', range=c(1, 100)))
df$mep_latency_dis <- as.integer(normalize(df$mep_latency, method='range', range=c(1, 100)))
df$mep_size_log = log(df$mep_size)
hist(df$mep_latency_log, breaks=40)
shapiro.test(df$mep_size_log)

# GLMM
df$mep_latency_dis = as.integer(normalize(df$mep_latency, method='range', range=c(1, 100)))
m = glmer(mep_latency_dis ~ (mu_power * mu_phase_bin) + (1|sub), data=df, family=poisson, nAGQ=0)
summary(m)
poisson <- fitdistr(resid(m), "Poisson")
qqp(resid(m), 'pois', lambda=poisson$estimate)
plot(df$mu_power, df$mep_size)

# Normality 2
x = df$mep_size_log
hist(x)
gf = goodfit(x, type= "poisson", method= "ML")
plot(gf, main="Count data vs Poisson distribution")
summary(gf)

# Normality

hist(df$mep_size_log+1)

x = log(df$mep_size)
qqnorm(x)
qqline(x, col='red')
ad.test(x)

qqnorm(rnorm(100,10,1)*10)
qqline(rnorm(100,10,1)*10, col = "red")


## Test distribution: https://ase.tufts.edu/gsc/gradresources/guidetomixedmodelsinr/mixed%20model%20guide.html

qqp(df$mep_size, "lnorm")

nbinom <- fitdistr(df$mep_size_log_dis, "Negative Binomial")
qqp(df$mep_size_log_dis, "nbinom", size = nbinom$estimate[[1]], mu = nbinom$estimate[[2]])

poisson <- fitdistr(df$mep_size_log_dis, "Poisson")
qqp(df$mep_size_log_dis, "pois", lambda=poisson$estimate)

gamma <- fitdistr(df$mep_size_dis, "gamma")
qqp(df$mep_size_dis, "gamma", shape = gamma$estimate[[1]], rate = gamma$estimate[[2]])

# Interaction plots -----
plot_bars <- function(df, column, title) {
  df = df[order(df$sub),]
  par(mar=c(8,5.5,4,4)+.1)
  boxplot(get(column) ~ sub, data=df, las=2, ylab=column)
  title(title)
}

plot_bars(df, 'mu_power', 'Power vs session')
plot_bars(df, 'mep_size', 'MEP size vs session')