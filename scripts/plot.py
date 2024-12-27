import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
          
def univariate_analysis_histogram_plot(data):
    for column in data.columns:
        if data[column].dtype in ['int64', 'float64']:  # Numerical columns
            print(f"{column}")
            plt.figure(figsize=(8, 6))
            data[column].plot(kind='hist', bins=10, edgecolor='black')
            plt.title(f'Histogram for {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()

def univariate_analysis_bar_chart(data):
    for column in data.columns:
        if data[column].dtype.name == 'category':  # Categorical columns
            print(f"{column}")

            plt.figure(figsize=(8, 6))
            data[column].value_counts().plot(kind='bar')
            plt.title(f'Bar Chart for {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()

def plot_scatter_by_group(data, x, y, group):

    plt.figure(figsize=(10, 6))
    for grp in data[group].unique():
        subset = data[data[group] == grp]
        plt.scatter(subset[x], subset[y], label=f"{group}: {grp}", alpha=0.7)

    plt.title(f"Scatter Plot of {y} vs {x} by {group}")
    plt.xlabel(x)
    plt.ylabel(y)
    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, title=group, frameon=False)
    plt.grid(True)
    plt.show()

def plot_correlation_matrix(data, columns=None):

    if columns is None:
        columns = data.select_dtypes(include=['int64', 'float64']).columns

    corr_matrix = data[columns].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()


