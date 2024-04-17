# Importing library
from scipy.stats import f_oneway

# Performance data for each group (engine oils)
performance1 = [89, 89, 88, 78, 79]
performance2 = [93, 92, 94, 89, 88]
performance3 = [89, 88, 89, 93, 90]
performance4 = [81, 78, 81, 92, 82]

# Conduct the one-way ANOVA
F, p_value = f_oneway(performance1, performance2, performance3, performance4)

# Output the results
if p_value < 0.05:
    print("There are significant differences in performance between the engine oils.")
else:
    print("There are no significant differences in performance between the engine oils.")

# Print the ANOVA results
print("ANOVA F-value:", F)
print("ANOVA p-value:", p_value)
