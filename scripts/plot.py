import pandas as pd
import matplotlib.pyplot as plt
          
def univariate_analysis_histogram_plot(data):
    for column in data.columns:
        if data[column].dtype in ['int64', 'float64']:  # Numerical columns
            plt.figure(figsize=(8, 6))
            data[column].plot(kind='hist', bins=10, edgecolor='black')
            plt.title(f'Histogram for {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()

def univariate_analysis_bar_chart(data):
    for column in data.columns:
        if data[column].dtype.name == 'category':  # Categorical columns
            plt.figure(figsize=(8, 6))
            data[column].value_counts().plot(kind='bar')
            plt.title(f'Bar Chart for {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()

