# install.packages("tidyverse")
# install.packages("readxl")

# Load excel file.
library(readxl)
df = read_excel("118-phase-powers-v3.xlsx")

# Convert to nominal.
df$sub = factor(df$sub)
df$exp = factor(df$exp)
df$trial_num = factor(df$trial_num)
View(df)
df$mep_by_cmap_log <- log10(df$mep_by_cmap)
df$mep_size_log <- log10(df$mep_size)
df$LTM1_beta_power_abs <- abs(df$LTM1_beta_power)
df$LTM1_mu_power_abs <- abs(df$LTM1_mu_power)
df$C3_mu_power_abs <- abs(df$C3_mu_power)
df$C3_beta_power_abs <- abs(df$C3_beta_power)
df$C4_mu_power_abs <- abs(df$C4_mu_power)
df$C4_beta_power_abs <- abs(df$C4_beta_power)

# Libraries for LMM.
library(lme4) # for lmer
library(lmerTest)
library(devtools)
library(car) # for Anova



# C3 Mu
# ----------------
# Remove outliers.
df2 = df
#df2 <- subset(df, (C3_mu_phase > 45 & C3_mu_phase < 135) | (C3_mu_phase > 225 & C3_mu_phase < 315))
# Main LMM
m = lmer(mep_size_log ~ (C3_mu_power * C3_mu_phase)/exp + (1|sub), data=df2)
Anova(m, type=3, test.statistic="F")

# C3 Mu Abs
# ----------------
# Remove outliers.
df2 = df
df2 <- subset(df, (C3_mu_phase > 45 & C3_mu_phase < 135) | (C3_mu_phase > 225 & C3_mu_phase < 315))
# Main LMM
m = lmer(mep_by_cmap_log ~ (C3_mu_power_abs * C3_mu_phase) + (1|sub), data=df2)
Anova(m, type=3, test.statistic="F")

# C3 Beta
# ----------------
# Remove outliers.
df2 = df
df2 <- subset(df, (C3_beta_phase > 45 & C3_beta_phase < 135) | (C3_beta_phase > 225 & C3_beta_phase < 315))
# Main LMM
m = lmer(mep_by_cmap_log ~ (C3_beta_power * C3_beta_phase) + (1|sub), data=df2)
Anova(m, type=3, test.statistic="F")

# C3 Beta Abs
# ----------------
# Remove outliers.
df2 = df
df2 <- subset(df, (C3_beta_phase > 45 & C3_beta_phase < 135) | (C3_beta_phase > 225 & C3_beta_phase < 315))
# Main LMM
m = lmer(mep_by_cmap_log ~ (C3_beta_power_abs * C3_beta_phase) + (1|sub), data=df2)
Anova(m, type=3, test.statistic="F")

# C4 Mu
# ----------------
# Remove outliers.
df2 = df
# df2 <- subset(df, (C4_mu_phase > 45 & C4_mu_phase < 135) | (C4_mu_phase > 225 & C4_mu_phase < 315))
# Main LMM
m = lmer(mep_size_log ~ (C4_mu_power * C4_mu_phase) + (1|sub), data=df2)
Anova(m, type=3, test.statistic="F")

# C4 Mu Abs
# ----------------
# Remove outliers.
df2 = df
df2 <- subset(df, (C4_mu_phase > 45 & C4_mu_phase < 135) | (C4_mu_phase > 225 & C4_mu_phase < 315))
# Main LMM
m = lmer(mep_by_cmap_log ~ (C4_mu_power_abs * C4_mu_phase) + (1|sub), data=df2)
Anova(m, type=3, test.statistic="F")

# C4 Beta
# ----------------
# Remove outliers.
df2 = df
df2 <- subset(df, (C4_beta_phase > 45 & C4_beta_phase < 135) | (C4_beta_phase > 225 & C4_beta_phase < 315))
# Main LMM
m = lmer(mep_by_cmap_log ~ (C4_beta_power * C4_beta_phase) + (1|sub), data=df2)
Anova(m, type=3, test.statistic="F")

# C4 Beta Abs
# ----------------
# Remove outliers.
df2 = df
df2 <- subset(df, (C4_beta_phase > 45 & C4_beta_phase < 135) | (C4_beta_phase > 225 & C4_beta_phase < 315))
# Main LMM
m = lmer(mep_by_cmap_log ~ (C4_beta_power_abs * C4_beta_phase) + (1|sub), data=df2)
Anova(m, type=3, test.statistic="F")





# LTM1 Mu
# ----------------
# Remove outliers.
df2 = df
df2 <- subset(df, (LTM1_mu_phase > 45 & LTM1_mu_phase < 135) | (LTM1_mu_phase > 225 & LTM1_mu_phase < 315))
# Main LMM
m = lmer(mep_by_cmap_log ~ (LTM1_mu_power * LTM1_mu_phase) + (1|sub), data=df2)
Anova(m, type=3, test.statistic="F")

# LTM1 Mu Abs
# ----------------
# Remove outliers.
df2 = df
df2 <- subset(df, (LTM1_mu_phase > 45 & LTM1_mu_phase < 135) | (LTM1_mu_phase > 225 & LTM1_mu_phase < 315))
# Main LMM
m = lmer(mep_by_cmap_log ~ (LTM1_mu_power_abs * LTM1_mu_phase) + (1|sub), data=df2)
Anova(m, type=3, test.statistic="F")

# LTM1 Beta
# ----------------
# Remove outliers.
df2 = df
df2 <- subset(df, (LTM1_beta_phase > 45 & LTM1_beta_phase < 135) | (LTM1_beta_phase > 225 & LTM1_beta_phase < 315))
# Main LMM
m = lmer(mep_by_cmap_log ~ (LTM1_beta_power * LTM1_beta_phase) + (1|sub), data=df2)
Anova(m, type=3, test.statistic="F")

# LTM1 Beta Abs
# ----------------
# Remove outliers.
df2 = df
df2 <- subset(df, (LTM1_beta_phase > 45 & LTM1_beta_phase < 135) | (LTM1_beta_phase > 225 & LTM1_beta_phase < 315))
# Main LMM
m = lmer(mep_by_cmap_log ~ (LTM1_beta_power_abs * LTM1_beta_phase) + (1|sub), data=df2)
Anova(m, type=3, test.statistic="F")