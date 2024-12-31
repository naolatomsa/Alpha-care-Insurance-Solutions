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



# A/B Hypothesis Testing

## Objective: Accept or reject null hypotheses regarding risk and profit differences across various segments:
        Provinces
        Zip codes
        Gender
## Key Steps:
        Select Metrics: Identify KPIs for analysis.
        Data Segmentation: Create control (Group A) and test (Group B) groups.
        Statistical Testing: Perform chi-squared tests, t-tests, or z-tests.
        Analyze Results: Evaluate p-values to determine statistical significance.

# Statistical Modeling

## Objective: Build and evaluate predictive models for TotalPremium and TotalClaims.
## Key Steps:
        Data Preparation:
            Handle missing data.
            Create new features.
            Encode categorical data.
            Split data into training and testing sets.
        Model Building:
            Implement Linear Regression, Random Forests, and XGBoost.
        Model Evaluation:
            Use metrics like MAE, MSE, and R² to assess performance.
        Feature Importance:
            Analyze using SHAP or LIME for model interpretability.