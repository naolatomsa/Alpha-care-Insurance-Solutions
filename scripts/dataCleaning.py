def missing_percentage(data, thresholds):
    missing_percentage = (data.isnull().sum() / len(data)) * 100
    missing_columns_percentage = missing_percentage[missing_percentage > thresholds]
    return missing_columns_percentage;