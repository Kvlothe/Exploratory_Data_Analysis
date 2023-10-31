import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind

# Read in csv into a DF named data
data = pd.read_csv('churn_clean.csv')

# Relabel the columns listed as item1...item8 with appropriate questions
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


# Histograms
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data['MonthlyCharge'], kde=True)
plt.title('Distribution of MonthlyCharge')

plt.subplot(1, 2, 2)
sns.histplot(data['Bandwidth_GB_Year'], kde=True)
plt.title('Distribution of Bandwidth_GB_Year')

plt.tight_layout()
plt.savefig('Histogram_continuous.png')
plt.show()

# Boxplot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=data['MonthlyCharge'])
plt.title('Boxplot of MonthlyCharge')

plt.subplot(1, 2, 2)
sns.boxplot(y=data['Bandwidth_GB_Year'])
plt.title('Boxplot of Bandwidth_GB_Year')

plt.tight_layout()
plt.savefig('Boxplot_continuous.png')
plt.show()

print(data[['MonthlyCharge', 'Bandwidth_GB_Year']].describe())
print("Skewness for MonthlyCharge:", data['MonthlyCharge'].skew())
print("Skewness for Bandwidth_GB_Year:", data['Bandwidth_GB_Year'].skew())

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x=data['Courteous exchange'])
plt.title('Distribution of Courteous exchange')

plt.subplot(1, 2, 2)
sns.countplot(x=data['Churn'])
plt.title('Distribution of Churn')

plt.tight_layout()
plt.savefig('Categorical_graph.png')
plt.show()

print(data['Courteous exchange'].value_counts())
print(data['Churn'].value_counts())

sns.scatterplot(x=data['MonthlyCharge'], y=data['Bandwidth_GB_Year'])
plt.title('Scatterplot of MonthlyCharge vs. Bandwidth_GB_Year')
plt.savefig('Scatterplot.jpg')
plt.show()

correlation = data[['MonthlyCharge', 'Bandwidth_GB_Year']].corr()
print(correlation)

cross_tab = pd.crosstab(data['Courteous exchange'], data['Churn'])

sns.heatmap(cross_tab, annot=True, cmap='Blues', fmt='g')
plt.title('Heatmap of Courteous exchange vs. Churn')
plt.savefig('Heatmap.jpg')
plt.show()
print(pd.crosstab(data['Courteous exchange'], data['Churn']))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x=data['Churn'], y=data['MonthlyCharge'])
plt.title('Boxplot of MonthlyCharge across Churn categories')

plt.subplot(1, 2, 2)
sns.boxplot(x=data['Churn'], y=data['Bandwidth_GB_Year'])
plt.title('Boxplot of Bandwidth_GB_Year across Churn categories')

plt.tight_layout()
plt.savefig('Boxplot2.jpg')
plt.show()

print(data.groupby('Churn')['MonthlyCharge'].mean())
print(data.groupby('Churn')['Bandwidth_GB_Year'].mean())
