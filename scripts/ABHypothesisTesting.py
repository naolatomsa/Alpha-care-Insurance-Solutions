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
    # print(f"The p_value value: {p_value:.2e}")

    return t_stat, p_value


def analyze_p_value(p_value, alpha=0.05):

    if p_value < alpha:
        return "Reject the null hypothesis: Statistically significant."
    else:
        return "Fail to reject the null hypothesis: Not statistically significant."
    
    
    
    


