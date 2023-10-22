import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind

# Read in csv into a DF named data
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

# Create bins - originally used for continuous numeric columns in chi-squared
# Convert 'MonthlyCharge' into, say, 4 bins of equal width
data['MonthlyCharge_bin'] = pd.cut(data['MonthlyCharge'], bins=4, labels=['Low', 'Medium', 'High', 'Very High'])

# Convert 'Bandwidth_GB_Year' into, say, 4 bins of equal width
data['Bandwidth_GB_Year_bin'] = pd.cut(data['Bandwidth_GB_Year'], bins=4, labels=['Low', 'Medium', 'High', 'Very High'])

# Group categorical columns
categorical_columns = ['Options', 'Courteous exchange', 'Timely replacement', 'Timely response', 'Respectful response',
                       'Evidence of active listening', 'Reliability', 'Timely fixes']
# Group numeric columns
numeric_columns = ['MonthlyCharge', 'Bandwidth_GB_Year']
# Group numeric binned columns
numeric_bin_columns = ['MonthlyCharge_bin', 'Bandwidth_GB_Year_bin']

# Split data based on 'Churn'
group1 = data[data['Churn'] == 'Yes']
group2 = data[data['Churn'] == 'No']

# Plot histograms for categorical columns
for column in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=column, hue='Churn')
    plt.title(f"Distribution of {column} by Churn")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{column}_histogram.png", dpi=300)  # specify the desired resolution with dpi
    plt.show()

# Plot histograms for numeric columns
for column in numeric_bin_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x=column, hue='Churn', kde=True, bins=30)
    plt.title(f"Distribution of {column} by Churn")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{column}_histogram.png", dpi=300)
    plt.show()


results = {}

for column in categorical_columns:
    # Create a contingency table
    contingency_table = pd.crosstab(data[column], data['Churn'])

    # Perform the chi-squared test
    chi2, p, _, _ = chi2_contingency(contingency_table)

    results[column] = {'Chi-squared Value': chi2, 'p-value': p}

# Display the results
print('Chi-squared\n')
for column, values in results.items():
    print(f"Column: {column}")
    print(f"Chi-squared Value: {values['Chi-squared Value']}")
    print(f"P-value: {values['p-value']}\n")

print('T-Test\n')
for column in numeric_columns:
    t_stat, p_val = ttest_ind(group1[column], group2[column])
    print(f"Column: {column}")
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_val}\n")

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
