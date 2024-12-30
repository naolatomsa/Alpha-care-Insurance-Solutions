# import pandas as pd
# from scipy.stats import ttest_ind, chi2_contingency
# import matplotlib.pyplot as plt
# import seaborn as sns
# from itertools import combinations

# # # Load your dataset (replace with your dataset path)
# # data = pd.read_csv('your_dataset.csv')
# # data = data.loc[:, ~data.columns.str.contains('^Unnamed')]  # Remove unnamed columns

# def calculate_margin(data):
#     """
#     Add a calculated margin column to the dataset.
#     Margin = TotalPremium - TotalClaims
#     """
#     data['Margin'] = data['TotalPremium'] - data['TotalClaims']
#     return data

# def segment_data_for_hypothesis(data, column):
#     """
#     Generates all combinations of two groups based on unique values in the column.

#     Args:
#         data (DataFrame): The input dataset.
#         column (str): The column used for segmentation.

#     Yields:
#         group_a (DataFrame): Data for Group A.
#         group_b (DataFrame): Data for Group B.
#         group_a_value (str): The value representing Group A.
#         group_b_value (str): The value representing Group B.
#     """
#     unique_values = data[column].dropna().unique()
#     if len(unique_values) < 2:
#         raise ValueError(f"Not enough unique values in column '{column}' to create two groups.")

#     for group_a_value, group_b_value in combinations(unique_values, 2):
#         group_a = data[data[column] == group_a_value]
#         group_b = data[data[column] == group_b_value]
#         return group_a, group_b

# def perform_ttest(group_a, group_b, metric):
#     """
#     Perform a t-test for numerical metrics.

#     Args:
#         group_a (DataFrame): Data for Group A.
#         group_b (DataFrame): Data for Group B.
#         metric (str): The column to perform the test on.

#     Returns:
#         t_stat (float): The t-statistic.
#         p_value (float): The p-value from the test.
#     """
#     t_stat, p_value = ttest_ind(group_a[metric], group_b[metric], nan_policy='omit')
#     return t_stat, p_value

# def perform_chi2_test(data, column, target):
#     """
#     Perform a chi-squared test for categorical metrics.

#     Args:
#         data (DataFrame): The input dataset.
#         column (str): The categorical column for grouping.
#         target (str): The target categorical column to compare.

#     Returns:
#         chi2 (float): The chi-squared statistic.
#         p_value (float): The p-value from the test.
#     """
#     contingency_table = pd.crosstab(data[column], data[target])
#     chi2, p_value, _, _ = chi2_contingency(contingency_table)
#     return chi2, p_value

# def analyze_p_value(p_value, alpha=0.05):
#     """
#     Analyze the p-value to determine statistical significance.

#     Args:
#         p_value (float): The p-value from a statistical test.
#         alpha (float): The significance level (default=0.05).

#     Returns:
#         str: Interpretation of the result.
#     """
#     if p_value < alpha:
#         return "Reject the null hypothesis: Statistically significant."
#     else:
#         return "Fail to reject the null hypothesis: Not statistically significant."

# def plot_metric_distribution(data, group_column, metric):
#     """
#     Plot the distribution of a metric grouped by a categorical column.

#     Args:
#         data (DataFrame): The input dataset.
#         group_column (str): The column to group data by.
#         metric (str): The numerical column to plot.
#     """
#     sns.boxplot(x=group_column, y=metric, data=data)
#     plt.title(f"Distribution of {metric} by {group_column}")
#     plt.xticks(rotation=45)
#     plt.show()

# # # Prepare dataset
# # data = calculate_margin(data)

# # # Hypothesis 1: Risk differences across provinces
# # print("\nHypothesis 1: Risk differences across provinces")
# # for group_a, group_b, group_a_value, group_b_value in segment_data_for_hypothesis(data, column='Province'):
# #     subset_data = data[(data['Province'] == group_a_value) | (data['Province'] == group_b_value)]
# #     chi2, p_value = perform_chi2_test(subset_data, column='Province', target='StatutoryRiskType')
# #     print(f"\nTesting {group_a_value} vs {group_b_value}")
# #     print(f"Chi-Squared Test: {analyze_p_value(p_value)}")

# # # Hypothesis 2: Risk differences between zip codes
# # print("\nHypothesis 2: Risk differences between zip codes")
# # for group_a, group_b, group_a_value, group_b_value in segment_data_for_hypothesis(data, column='PostalCode'):
# #     subset_data = data[(data['PostalCode'] == group_a_value) | (data['PostalCode'] == group_b_value)]
# #     chi2, p_value = perform_chi2_test(subset_data, column='PostalCode', target='StatutoryRiskType')
# #     print(f"\nTesting zip codes {group_a_value} vs {group_b_value}")
# #     print(f"Chi-Squared Test: {analyze_p_value(p_value)}")

# # # Hypothesis 3: Margin differences between zip codes
# # print("\nHypothesis 3: Margin differences between zip codes")
# # for group_a, group_b, group_a_value, group_b_value in segment_data_for_hypothesis(data, column='PostalCode'):
# #     t_stat, p_value = perform_ttest(group_a, group_b, metric='Margin')
# #     print(f"\nT-Test for Margin Differences between zip codes {group_a_value} and {group_b_value}")
# #     print(f"T-Test Result: {analyze_p_value(p_value)}")

# # # Hypothesis 4: Risk differences between Women and Men
# # print("\nHypothesis 4: Risk differences between Women and Men")
# # for group_a, group_b, group_a_value, group_b_value in segment_data_for_hypothesis(data, column='Gender'):
# #     subset_data = data[(data['Gender'] == group_a_value) | (data['Gender'] == group_b_value)]
# #     chi2, p_value = perform_chi2_test(subset_data, column='Gender', target='StatutoryRiskType')
# #     print(f"\nTesting {group_a_value} vs {group_b_value}")
# #     print(f"Chi-Squared Test: {analyze_p_value(p_value)}")

# # # Example plots
# # plot_metric_distribution(data, group_column='Province', metric='TotalClaims')
# # plot_metric_distribution(data, group_column='PostalCode', metric='Margin')
# # plot_metric_distribution(data, group_column='Gender', metric='TotalClaims')

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_margin(data):

    data['Margin'] = data['TotalPremium'] - data['TotalClaims']
    return data

def segment_randomly(data, column, random_seed):

    # Get unique values from the specified column
    unique_values = data[column].dropna().unique()

    # Shuffle the unique values
    if random_seed is not None:
        np.random.seed(random_seed)
    np.random.shuffle(unique_values)

    # Split the unique values into two groups
    mid_index = len(unique_values) // 2
    group_a_values = unique_values[:mid_index]
    group_b_values = unique_values[mid_index:]

    # Filter the original dataset into two groups
    group_a = data[data[column].isin(group_a_values)]
    group_b = data[data[column].isin(group_b_values)]

    return group_a, group_b


def segment_by_risk(data, risk_column, column_to_segement):

    # Aggregate total risk by province
    column_to_segement_risk = data.groupby(column_to_segement)[risk_column].sum().reset_index()

    # Sort provinces by risk
    column_to_segement_risk = column_to_segement_risk.sort_values(by=risk_column).reset_index(drop=True)

    # Divide provinces into two groups
    mid_index = len(column_to_segement_risk) // 2
    group_a_values = column_to_segement_risk[column_to_segement][:mid_index].tolist()  # Low risk provinces
    group_b_values = column_to_segement_risk[column_to_segement][mid_index:].tolist()  # High risk provinces

    # Filter the original data into two groups
    group_a = data[data[column_to_segement].isin(group_a_values)]
    group_b = data[data[column_to_segement].isin(group_b_values)]

    return group_a, group_b



def segment_gender(data, gender_column, male_label, female_label):

    # Filter data for males and females only
    filtered_data = data[data[gender_column].isin([male_label, female_label])]

    # Segment into male and female groups
    male_group = filtered_data[filtered_data[gender_column] == male_label]
    female_group = filtered_data[filtered_data[gender_column] == female_label]

    return male_group, female_group




def perform_ttest(group_a, group_b, metric):

    t_stat, p_value = ttest_ind(group_a[metric], group_b[metric], nan_policy='omit')
    return t_stat, p_value


def analyze_p_value(p_value, alpha=0.05):

    if p_value < alpha:
        return "Reject the null hypothesis: Statistically significant."
    else:
        return "Fail to reject the null hypothesis: Not statistically significant."
    
    
    
    


