import pandas as pd
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest

# Load the dataset
data = pd.read_csv('train.csv')

# Display summary statistics of the dataset
print(data.describe())

# 1. One Sample t-Test: Income
# Before performing One Sample t-Test, perform Shapiro-Wilk test for normality to test wether Income data is normally distributed
# Shapiro-Wilk test for normality
# This tests the null hypothesis that the data was drawn from a normal distribution.
stat, p_value = stats.shapiro(data['Monthly_Inhand_Salary'])
print("Shapiro-Wilk Test for Income:")
print(f"statistic: {stat}, p-value: {p_value}")
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis - data is not normally distributed")
else:
    print("Do not reject the null hypothesis - data is normally distributed")
# Perform One Sample t-Test
# This tests whether the mean annual income of customers is significantly different from a hypothetical mean income of 30000.
t_stat, p_value = stats.ttest_1samp(data['Monthly_Inhand_Salary'], 1824, alternative='two-sided')  # Testing against a hypothetical mean income of 30000
print("One Sample t-Test for Monthly_Inhand_Salary:")
print(f"t-statistic: {t_stat}, p-value: {p_value}")
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis")
else:
    print("Do not reject the null hypothesis")



# 3. Hypothesis Test for Correlation: Income vs. Purchase Amount
# This tests whether there is a significant correlation between the income of customers and their purchase amount. 
r, p_value = stats.pearsonr(data['Annual_Income'], data['Num_Credit_Inquiries'], alternative='two-sided')
print("\nHypothesis Test for Correlation between Income and Purchase Amount:")
print(f"Correlation coefficient (r): {r}, p-value: {p_value}")
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis")
else:
    print("Do not reject the null hypothesis")
