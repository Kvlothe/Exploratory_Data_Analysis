import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

data = pd.read_csv('churn_clean.csv')

# Relabel the columns listed as item1..item8 with appropriate questions
data.rename(columns={'Item1': 'Timely response',
                     'Item2': 'Timely fixes',
                     'Item3': 'Timely replacement',
                     'Item4': 'Reliability',
                     'Item5': 'Options',
                     'Item6': 'Respectful response',
                     'Item7': 'Courteous exchange',
                     'Item8': 'Evidence of active listening'},
            inplace=True)

# Convert 'MonthlyCharge' into, say, 4 bins of equal width
data['MonthlyCharge_bin'] = pd.cut(data['MonthlyCharge'], bins=4, labels=['Low', 'Medium', 'High', 'Very High'])

# Convert 'Bandwidth_GB_Year' into, say, 4 bins of equal width
data['Bandwidth_GB_Year_bin'] = pd.cut(data['Bandwidth_GB_Year'], bins=4, labels=['Low', 'Medium', 'High', 'Very High'])


columns_of_importance = ['Options', 'Courteous exchange', 'MonthlyCharge_bin', 'Timely replacement', 'Timely response',
                         'Bandwidth_GB_Year_bin', 'Respectful response', 'Evidence of active listening']

results = {}

for column in columns_of_importance:
    # Create a contingency table
    contingency_table = pd.crosstab(data[column], data['Churn'])

    # Perform the chi-squared test
    chi2, p, _, _ = chi2_contingency(contingency_table)

    results[column] = {'Chi-squared Value': chi2, 'p-value': p}

# Display the results
for column, values in results.items():
    print(f"Column: {column}")
    print(f"Chi-squared Value: {values['Chi-squared Value']}")
    print(f"P-value: {values['p-value']}\n")

########################
# contingency_table = pd.crosstab(data['Churn'], data['Bandwidth_GB_Year'])
# print(contingency_table)
#
# # Perform chi-squared test to get expected frequencies
# chi2, p, _, expected = chi2_contingency(contingency_table)
#
# # Calculate chi-squared residuals
# residuals = (contingency_table - expected) / np.sqrt(expected)
#
# print(f"Chi-squared Value = {chi2}")
# print(f"P-value = {p}")
# print(f"Expected Frequencies Table: \n{expected}")
##########################

# # Plot a heatmap
# plt.figure(figsize=(10, 7))
# sns.heatmap(contingency_table, annot=True, cmap="YlGnBu", fmt='g')
# plt.title('Contingency Table Heatmap')
# plt.show()

# # Plot the residuals using a heatmap
# plt.figure(figsize=(10, 7))
# sns.heatmap(residuals, annot=True, cmap="coolwarm", center=0)
# plt.title('Chi-squared Residuals Heatmap')
# plt.show()
