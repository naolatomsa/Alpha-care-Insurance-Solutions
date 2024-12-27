def missing_percentage(data, thresholds):
    missing_percentage = (data.isnull().sum() / len(data)) * 100
    missing_columns_percentage = missing_percentage[missing_percentage > thresholds]
    return missing_columns_percentage;

def remove_missing_row(data, thresholds):
    missing_percentage = (data.isnull().sum()/ len(data)*100)
    missing_columns_percentage = missing_percentage[missing_percentage < thresholds].index
    
    data = data.dropna(subset=missing_columns_percentage)
    
    return data;
    
def fill_missing_values(data):
    data['Bank'] = data['Bank'].fillna(data['Bank'].mode()[0])
    data['NewVehicle'] = data['NewVehicle'].fillna(data['Bank'].mode()[0])
    data['AccountType'] = data['AccountType'].fillna(data['Bank'].mode()[0])
    
    return data

def drop_column(data, column_to_drop):
    data = data.drop(columns=column_to_drop)
    return data;


def identify_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

# Cap and floor outliers for all numeric columns
def cap_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    return data

