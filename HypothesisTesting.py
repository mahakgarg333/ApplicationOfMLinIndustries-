import numpy as np
import scipy.stats as stats

# Create a dummy dataset of 10 year old children's weight for unpaired test
group1 = np.random.randint(20, 40, 10)
group2 = np.random.randint(20, 40, 10)

# Create a dummy dataset of 10 year old children's weight for paired test
before = np.random.randint(20, 40, 10)
after = np.random.randint(20, 40, 10)

# Define the null hypothesis
H0 = "The average weight of 10 year old children is 32kg."

# Define the alternative hypothesis
H1 = "The average weight of 10 year old children is more than 32kg."

# Population mean
population_mean = 32

# Perform unpaired t-test
t_stat_unpaired, p_value_unpaired = stats.ttest_ind(group1, group2)

# Perform paired t-test
t_stat_paired, p_value_paired = stats.ttest_rel(before, after)

# Calculate z-statistic and p-value for the z-test (assuming known population standard deviation)
mean_diff = np.mean(group1) - np.mean(group2)
population_std = 5  # Assuming known population standard deviation
n1, n2 = len(group1), len(group2)
z_stat_unpaired = mean_diff / (population_std * np.sqrt(1/n1 + 1/n2))
p_value_z_unpaired = stats.norm.cdf(z_stat_unpaired)  # Assuming two-tailed test

# Print the results for unpaired t-test
print("Unpaired t-test results:")
print("Null Hypothesis:", H0)
print("Alternate Hypothesis:", H1)
print("Population Mean:", population_mean)
print("T Statistics (unpaired):", t_stat_unpaired)
print("p-value (unpaired):", p_value_unpaired)
if p_value_unpaired < 0.05:
    print("Result: Reject the null hypothesis for unpaired t-test.")
else:
    print("Result: Fail to reject the null hypothesis for unpaired t-test.")

# Print the results for paired t-test
print("\nPaired t-test results:")
print("Null Hypothesis:", H0)
print("Alternate Hypothesis:", H1)
print("Population Mean:", population_mean)
print("T Statistics (paired):", t_stat_paired)
print("p-value (paired):", p_value_paired)
if p_value_paired < 0.05:
    print("Result: Reject the null hypothesis for paired t-test.")
else:
    print("Result: Fail to reject the null hypothesis for paired t-test.")

# Print the results for z-test
print("\nZ-test results:")
print("Null Hypothesis:", H0)
print("Alternate Hypothesis:", H1)
print("Population Mean:", population_mean)
print("Z Statistics:", z_stat_unpaired)
print("p-value (z-test):", p_value_z_unpaired)
if p_value_z_unpaired < 0.05:
    print("Result: Reject the null hypothesis for z-test.")
else:
    print("Result: Fail to reject the null hypothesis for z-test.")
