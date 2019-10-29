# install.packages("tidyverse")
# install.packages("readxl")

# Load excel file.
library(readxl)
df = read_excel("114-phase-powers-v1.xlsx")
View(df)
names(df)

# Convert to nominal.
df$sub = factor(df$sub)
df$exp = factor(df$exp)
df$trial_num = factor(df$trial_num)

# View summary.
library(dplyr)
summary(select(df, LTM1_mu_power, LTM1_mu_phase, mep_by_cmap, trial_num, exp))

# Explore some data.
library(plyr)
ddply(df, ~ LTM1_mu_power * LTM1_mu_phase, function(data) summary(data$mep_by_cmap))
ddply(df, ~ LTM1_mu_power * LTM1_mu_phase, summarise, mep_by_cmap.mean=mean(mep_by_cmap), mep_by_cmap.sd=sd(mep_by_cmap))

