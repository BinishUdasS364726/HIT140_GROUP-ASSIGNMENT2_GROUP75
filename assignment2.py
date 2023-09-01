import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Load the dataset without headers using the provided path
df = pd.read_csv(r"C:\Users\binis\OneDrive\Desktop\binish.conda\po1_data.csv", header=None)

 # Assign the provided column names
df.columns = [
    "Subject identifier", "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer(%)", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "Median Pitch", "Mean Pitch", "SD Pitch", "Min Pitch", "Max Pitch",
    "Number of Pulses", "Number of Periods", "Mean Period", "SD Period", "Fraction of Unvoiced Frames",
    "Number of Voice Breaks", "Degree of Voice Breaks", "UPDRS", "PD indicator"
]

print(df.head()) 


# Display basic statistics of the data and check for missing values
data_summary = df.describe(include='all').T
data_summary['missing_values'] = df.shape[0] - data_summary['count']

print(data_summary)

# Distributions of the Target variable
pd_distribution = df["PD indicator"].value_counts()


# Calculate the mean, median, and standard deviation for each group (PD and non-PD) for each column
mean_values = df.groupby('PD indicator').mean()
median_values = df.groupby('PD indicator').median()
std_dev_values = df.groupby('PD indicator').std()
percentile_25 = df.groupby('PD indicator').quantile(0.25)
percentile_75 = df.groupby('PD indicator').quantile(0.75)

print("Mean Values:\n", mean_values, "\n")
print("Median Values:\n", median_values, "\n")
print("Standard Deviation Values:\n", std_dev_values)
print("25th Percentile:\n", percentile_25, "\n")
print("75th Percentile:\n", percentile_75)

# Splitting the data into two groups based on the PD indicator
group_pd = df[df['PD indicator'] == 1]
group_no_pd = df[df['PD indicator'] == 0]

# Exclude 'Subject identifier' and 'PD indicator' columns for the t-test
columns_to_test = df.columns.difference(['Subject identifier', 'PD indicator'])

# Dictionary to store t-test results
t_test_results = {}

# Loop through each column and perform t-test
for column in columns_to_test:
    t_stat, p_val = ttest_ind(group_pd[column], group_no_pd[column])
    t_test_results[column] = {"T-Statistic": t_stat, "P-Value": p_val}

# Convert the results into a DataFrame for better visualization
t_test_df = pd.DataFrame.from_dict(t_test_results, orient='index')

# Adding a 'Change' column to the t-test results DataFrame
t_test_df['Change'] = ['Yes' if p < 0.05 else 'No' for p in t_test_df['P-Value']]

print(t_test_df)



# Identify variables with statistically significant differences (p < 0.05)
significant_columns = t_test_df[t_test_df["P-Value"] < 0.05].index.tolist()

# Print the results
print("Number of Variables with Statistically Significant Differences:", len(significant_columns))
print("\nVariables:")
for column in significant_columns:
    print(column)


# Loop through each significant variable and create a histogram
for column in significant_columns:
    plt.figure(figsize=(10, 6))
    
    # Plotting data for the PD group
    plt.hist(group_pd[column], bins=30, alpha=0.5, label='PD')
    
    # Plotting data for the non-PD group
    plt.hist(group_no_pd[column], bins=30, alpha=0.5, label='non-PD')
    
    plt.title(f'Histogram of {column} by Group')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()