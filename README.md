# Exploratory Data Analysis (EDA)
## Data Summarization:

    Calculate descriptive statistics for numerical features like TotalPremium and TotalClaim.

## Data Quality Assessment:

    Check for missing values and properly format categorical variables, dates, etc.

## Univariate Analysis:

    Visualize distributions of numerical columns (histograms) and categorical columns (bar charts).

## Bivariate/Multivariate Analysis:

    Use scatter plots and correlation matrices to explore relationships between variables.

## Trends Over Geography:

    Compare changes in insurance types, premiums, and auto makes.

## Outlier Detection:

    Use box plots to identify outliers.

## Visualization:

    Create at least 3 meaningful and visually appealing plots to highlight key insights.

# Data Version Control (DVC)
## Overview:

## DVC is used to track and version datasets, enabling reproducibility and collaboration.
## Steps:

    install DVC:

pip install dvc

Initialize DVC:

dvc init
git add .dvc
git commit -m "Initialize DVC"

Configure Local Remote Storage:

    Create storage directory:

mkdir dvc_storage

Add the remote storage:

    dvc remote add -d localstorage dvc_storage
    git add .dvc/config
    git commit -m "Configure local DVC remote storage"

Track and Push Data:

    Track the dataset:

dvc add data.csv
git add data.csv.dvc
git commit -m "Track dataset with DVC"

Push the data:

dvc push


# How to use this repo

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt




Python Libraries: pandas, matplotlib, seaborn
Version Control: Git, GitHub, DVC