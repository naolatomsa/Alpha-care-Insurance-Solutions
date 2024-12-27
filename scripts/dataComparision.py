
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cover_type_distribution(data, geography):
    """
    Plot the distribution of insurance cover types across geographical regions.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        geography (str): The column name for geographical grouping (e.g., 'Province').

    Returns:
        None: Displays a stacked bar chart.
    """
    # Calculate the percentage distribution of CoverType by geography
    cover_type_distribution = data.groupby([geography, 'CoverType']).size().unstack().fillna(0)
    cover_type_distribution = cover_type_distribution.div(cover_type_distribution.sum(axis=1), axis=0)

    # Plot stacked bar chart
    cover_type_distribution.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title(f'Insurance Cover Type Distribution by {geography}')
    plt.xlabel(geography)
    plt.ylabel('Percentage')
    plt.legend(title='CoverType')
    plt.show()

def plot_average_premium(data, geography):
    """
    Plot the average premium across geographical regions.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        geography (str): The column name for geographical grouping (e.g., 'Province').

    Returns:
        None: Displays a bar chart.
    """
    # Calculate average premium by geography
    average_premium = data.groupby(geography)['TotalPremium'].mean()

    # Plot bar chart
    average_premium.plot(kind='bar', figsize=(10, 6))
    plt.title(f'Average Premium by {geography}')
    plt.xlabel(geography)
    plt.ylabel('Average Premium')
    plt.show()

def plot_auto_make_distribution(data, geography):
    """
    Plot the distribution of auto makes across geographical regions.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        geography (str): The column name for geographical grouping (e.g., 'Province').

    Returns:
        None: Displays a stacked bar chart.
    """
    # Calculate the count distribution of Auto Makes by geography
    auto_make_distribution = data.groupby([geography, 'make']).size().unstack().fillna(0)

    # Plot stacked bar chart
    auto_make_distribution.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title(f'Auto Make Popularity by {geography}')
    plt.xlabel(geography)
    plt.ylabel('Count')
    plt.legend(title='Make', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def plot_new_vehicle_distribution(data, geography):
    """
    Plot the distribution of new vs. old vehicles across geographical regions.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        geography (str): The column name for geographical grouping (e.g., 'Province').

    Returns:
        None: Displays a bar chart.
    """
    new_vehicle_distribution = data.groupby([geography, 'NewVehicle']).size().unstack().fillna(0)
    new_vehicle_distribution = new_vehicle_distribution.div(new_vehicle_distribution.sum(axis=1), axis=0)

    # Plot stacked bar chart
    new_vehicle_distribution.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title(f'New vs. Old Vehicle Distribution by {geography}')
    plt.xlabel(geography)
    plt.ylabel('Percentage')
    plt.legend(title='NewVehicle')
    plt.show()

def plot_security_features_distribution(data, geography):
    """
    Plot the distribution of security features (AlarmImmobiliser, TrackingDevice) across geographical regions.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        geography (str): The column name for geographical grouping (e.g., 'Province').

    Returns:
        None: Displays multiple bar charts.
    """
    for feature in ['AlarmImmobiliser', 'TrackingDevice']:
        feature_distribution = data.groupby([geography, feature]).size().unstack().fillna(0)
        feature_distribution = feature_distribution.div(feature_distribution.sum(axis=1), axis=0)

        # Plot stacked bar chart
        feature_distribution.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title(f'{feature} Distribution by {geography}')
        plt.xlabel(geography)
        plt.ylabel('Percentage')
        plt.legend(title=feature)
        plt.show()


def correlation_by_geography(data, geography):
    """
    Calculate and display correlation between TotalPremium and TotalClaims by geography as a bar chart.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        geography (str): The column name for geographical grouping (e.g., 'Province').

    Returns:
        None: Displays a bar chart of correlations.
    """
    correlations = []

    for geo, group in data.groupby(geography):
        # Check if the required columns have valid data
        if group[['TotalPremium', 'TotalClaims']].notnull().all().all():
            corr = group[['TotalPremium', 'TotalClaims']].corr().iloc[0, 1]
            correlations.append((geo, corr))
        else:
            print(f"Skipping {geo} due to missing data.")

    # Create a DataFrame for the correlations
    correlation_df = pd.DataFrame(correlations, columns=[geography, 'Correlation']).sort_values(by='Correlation')

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(correlation_df[geography], correlation_df['Correlation'], color='skyblue')
    plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    plt.title(f'Correlation between TotalPremium and TotalClaims by {geography}')
    plt.xlabel(geography)
    plt.ylabel('Correlation')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()



